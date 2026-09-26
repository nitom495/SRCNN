# Real-Camera ×2 Super-Resolution Training Design

## Goal

Train a lightweight RGB ×2 super-resolution model for 1280×720 camera frames, producing 2560×1440 output. Training runs in FP32 on a single RTX 4090 24 GB GPU. The inference architecture remains suitable for a future INT8 FPGA implementation targeting 30 FPS.

The primary objective is improved perceptual fidelity on ordinary camera frames without hallucinated detail or unstable frame-to-frame textures.

## Current Baseline

The current model is a five-convolution ESPCN-style network:

- 3 → 16 channels
- 16 → 16 channels
- 16 → 16 channels
- 16 → 8 channels
- 8 → 12 channels
- PixelShuffle ×2

Training currently uses paired DIV2K bicubic ×2 images, 128×128 LR patches, 256×256 HR targets, L1 loss, Adam, cosine learning-rate decay, and batch size 64.

## Scope

This phase changes the data degradation, training schedule, validation, logging, and checkpointing. It does not change the deployed network architecture. Model widening or additional residual blocks will be considered only after measuring the present network on the target FPGA.

Quantization, GAN training, temporal video models, and FPGA implementation are outside this phase.

## Training Data

Use DF2K as the main HR source:

- DIV2K: 800 training images
- Flickr2K: 2,650 training images

Only HR images are required. LR samples are generated dynamically so that each HR crop can produce different camera-like degradations across iterations.

Use RealSR as the auxiliary real-paired validation dataset. DRealSR may be added as a secondary benchmark but does not replace RealSR or target-camera evaluation. These datasets are not treated as an exact match for the deployment camera.

## Synthetic Camera Degradation

For each HR crop, choose one of two paths:

- 20% probability: standard bicubic ×2 downsampling.
- 80% probability: camera-like synthetic degradation.

The camera-like path applies these operations in order:

1. Random isotropic or anisotropic Gaussian blur with kernel size selected from 7, 9, 11, or 13 and sigma in the range 0.2–1.5.
2. Downsample exactly ×2 using a random choice of area, bilinear, or bicubic interpolation.
3. Apply Gaussian noise with probability 0.5 and standard deviation 0–10 on the 0–255 scale, Poisson noise with probability 0.2 and scale 0.05–1.5, or no noise with probability 0.3.
4. Apply JPEG compression with probability 0.5 and quality in the range 60–100.
5. Convert the result to a three-channel RGB tensor in the range 0–1.

The first implementation intentionally avoids Real-ESRGAN's stronger second-order degradation. The intended input is a camera feed, not heavily restored or repeatedly compressed internet media. Degradation ranges can be widened later using visual evidence from representative camera frames.

Paired geometric augmentation must apply identical transforms to LR and HR. Use horizontal flips with probability 0.5. Do not use vertical flips or 90-degree rotations so that camera-scene orientation remains realistic.

## Training Schedule

Use LR patches of 128×128 and HR targets of 256×256.

Start with batch size 64. Benchmark batch sizes 64, 96, and 128 on the RTX 4090, then select the largest value that leaves enough memory for stable validation and does not reduce samples per second.

Train by optimizer steps instead of relying on a small epoch count:

- Stage 1: 120,000 steps with L1 loss.
- Stage 2: 30,000 steps with L1 loss plus a Sobel-gradient L1 loss weighted by 0.05.
- Total: 150,000 optimizer steps.

Use Adam with learning rate 2e-4 and betas 0.9 and 0.999. Apply cosine decay from 2e-4 to 1e-6 across the full schedule. Maintain an exponential moving average of model weights with decay 0.999 and use the EMA weights for validation and final export.

GAN and perceptual losses are excluded because the deployment target is live camera video. They can create plausible but incorrect texture and can increase frame-to-frame instability.

## RTX 4090 Input Pipeline

Keep FP32 training for this phase. Use the following data-loading configuration as an initial 4090 profile:

- 8 DataLoader workers
- pinned host memory
- persistent workers
- prefetch factor 2
- non-blocking host-to-device transfers
- cuDNN benchmark mode for the fixed patch size

Dataset and checkpoint storage should reside on a local SSD. Throughput is measured as samples per second after a warm-up period. Worker count and batch size are selected by measured throughput rather than GPU utilization alone.

## Validation

Validation runs every 2,000 optimizer steps using fixed, deterministic samples.

Maintain three validation views:

1. DIV2K validation with official bicubic ×2 pairs, measuring RGB and luminance PSNR and SSIM.
2. A fixed RealSR paired subset, measuring PSNR and SSIM after applying the dataset's required alignment and border handling. DRealSR can be reported separately if it is added later.
3. A small unpaired collection of representative 720p camera frames for consistent visual comparison.

Save separate checkpoints for:

- best DIV2K PSNR
- best real-paired validation score
- latest training state

Visual review checks fine edges, text, foliage, repeated patterns, noise amplification, ringing, color shifts, and consistency across adjacent camera frames.

## Checkpointing and Recovery

Save a complete checkpoint every 5,000 steps containing:

- model weights
- EMA weights
- optimizer state
- scheduler state
- current step
- random seeds or generator states needed for reproducibility
- training configuration

Training must resume from the latest complete checkpoint without resetting the learning-rate schedule.

Create the output directory before saving. A missing or corrupt checkpoint produces an explicit error rather than silently starting a new run.

## Error Handling

Dataset initialization fails clearly when no images are found or a paired validation image is missing. Training validates LR/HR scale and channel count before the first optimizer step.

The loop stops and saves diagnostic state if loss becomes NaN or infinite. CUDA out-of-memory errors are handled by reducing batch size and restarting from the latest checkpoint rather than changing multiple parameters at once.

## Verification

Before the full run:

1. Confirm one degraded pair has shapes 3×128×128 and 3×256×256 and remains spatially aligned.
2. Confirm model output shape equals the HR target shape.
3. Overfit one fixed batch and verify that its loss decreases substantially.
4. Run a short multi-worker training test on Windows and confirm that training starts only once.
5. Save and reload a checkpoint, then confirm identical validation output.
6. Run full-frame 1280×720 inference and confirm a 2560×1440 RGB output.

The full training run begins only after these checks pass.

## Success Criteria

The phase is successful when:

- validation loss and PSNR improve beyond the current bicubic-only baseline;
- real-paired validation improves without severe degradation of DIV2K performance;
- representative camera frames show sharper edges without objectionable ringing, noise amplification, or invented texture;
- the model architecture and inference operation set remain unchanged for later FPGA evaluation;
- training can resume reliably from checkpoints and complete the 150,000-step schedule.
