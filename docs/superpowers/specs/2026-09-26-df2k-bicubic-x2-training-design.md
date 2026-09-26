# DF2K Bicubic RGB x2 Super-Resolution Training Design

**Date:** 2026-09-26

## 1. Goal and Scope

Train the existing lightweight RGB x2 super-resolution network on standard bicubic data, using FP32 on a single RTX 4090 24 GB GPU.

The current stage optimizes reconstruction quality on standard bicubic degradation while preserving the small PixelShuffle architecture for a future INT8 FPGA deployment. Camera capture, camera-specific degradation, RealSR/DRealSR, video processing, INT8 quantization, and FPGA integration are explicitly deferred.

## 2. Model Contract

Keep the current network topology unchanged:

- input: RGB tensor with shape `N x 3 x H x W`;
- output: RGB tensor with shape `N x 3 x 2H x 2W`;
- five 3x3 convolution layers with channel progression `3 -> 16 -> 16 -> 16 -> 8 -> 12`;
- ReLU after the first four convolutions;
- `PixelShuffle(2)` as the upsampling operation;
- no normalization layers, attention blocks, GAN components, or pretrained feature extractor.

The unchanged architecture isolates gains from the data and training pipeline and keeps the deployed computation graph simple.

## 3. Training Data

Use DF2K, formed from two independent training sources:

- DIV2K training: 800 HR images and their official bicubic x2 LR images;
- Flickr2K training: 2,650 HR images and their bicubic x2 LR images;
- total: 3,450 paired training images.

The two datasets retain separate roots and pairing rules so overlapping numeric filenames cannot collide. Each LR image must have exactly one HR counterpart. Dataset construction fails immediately for missing files, duplicate samples, empty roots, or invalid dimensions.

All images are converted to RGB when loaded. An LR image and its HR partner must have an exact x2 spatial relationship. Images that do not satisfy the relationship are rejected rather than silently resized.

Dataset files remain excluded from Git.

## 4. Patch Sampling and Augmentation

For each training sample:

1. select a random `128 x 128` crop in LR coordinates;
2. take the aligned `256 x 256` crop from the HR image;
3. convert both crops to float tensors in `[0, 1]`;
4. apply the same geometric augmentation to LR and HR.

Allowed paired augmentation:

- horizontal flip with probability 0.5;
- vertical flip with probability 0.5;
- rotation by 0, 90, 180, or 270 degrees with equal probability.

No blur, noise, compression, sharpening, color jitter, or camera simulation is applied in this stage.

## 5. Validation Data and Metrics

Use the 100-image DIV2K validation set with official bicubic x2 LR images. Validation data is never used for training and receives no random augmentation.

Evaluate the full validation set using deterministic inference. Before metric calculation, shave a two-pixel border from each side of the SR and HR images to avoid scale-boundary effects.

Report:

- RGB PSNR;
- luminance PSNR using a documented RGB-to-Y conversion;
- SSIM using the same border handling and value range.

The primary model-selection metric is validation Y-channel PSNR. RGB PSNR, SSIM, and validation L1 loss remain diagnostic metrics.

## 6. Optimization

Training uses:

- `L1Loss` between predicted RGB SR and HR;
- Adam optimizer with `lr=2e-4` and `betas=(0.9, 0.999)`;
- cosine learning-rate decay from `2e-4` to `1e-6`;
- 150,000 optimizer steps;
- batch size 64 initially on the RTX 4090;
- FP32 parameters, activations, gradients, and optimizer state;
- no automatic mixed precision and no gradient scaler.

Training is step-based rather than epoch-based because random patch sampling makes optimizer steps the clearer schedule. If batch size changes, the default 150,000-step schedule remains unchanged unless an experiment explicitly changes it.

The implementation records the actual batch size, step count, learning rate, random seed, dataset sizes, and configuration in every run.

## 7. GPU Input Pipeline

The Windows data loader remains protected by `if __name__ == "__main__":`.

Use configurable loader settings with these RTX 4090 starting values:

- `num_workers=4`;
- `pin_memory=True`;
- `persistent_workers=True` when workers are enabled;
- asynchronous host-to-device copies with `non_blocking=True`;
- shuffled training batches and deterministic validation order.

Worker count and batch size are performance settings, not model-quality settings. They may be tuned after measuring GPU utilization and memory use.

## 8. Checkpoints and Recovery

Create the output directory before saving. Maintain:

- `latest.pth`: overwritten at each checkpoint interval;
- `best_y_psnr.pth`: overwritten only when validation Y PSNR improves;
- numbered checkpoints at a configurable interval for recovery and comparison.

A resumable checkpoint contains:

- model state;
- optimizer state;
- scheduler state;
- current global step;
- best validation metric;
- training configuration;
- random-number-generator states needed for practical reproducibility.

Also support exporting a weights-only state dictionary for later inference and FPGA conversion work.

## 9. Logging

Write concise console output and TensorBoard scalars for:

- training L1 loss;
- learning rate;
- validation L1 loss;
- RGB PSNR;
- Y-channel PSNR;
- SSIM;
- elapsed time and throughput.

Validation and checkpointing occur at fixed step intervals. Training must not print one line per batch by default.

## 10. Reproducibility and Failure Handling

Provide a configurable seed for Python and PyTorch. Save the resolved configuration beside checkpoints.

Before training starts, verify:

- CUDA is available;
- the selected device is an NVIDIA CUDA device;
- all required dataset directories exist;
- DIV2K and Flickr2K contain the expected number of paired samples;
- a sample LR/HR pair has the required RGB channels and x2 dimensions;
- model output shape matches the HR patch shape;
- the loss is finite.

Training stops with a clear error for invalid paths, mismatched pairs, non-finite loss, or incompatible tensor shapes.

## 11. Acceptance Criteria

The implementation is accepted when:

- DF2K contains 3,450 validated LR/HR pairs;
- a training batch has shapes `N x 3 x 128 x 128` and `N x 3 x 256 x 256`;
- one FP32 optimizer step succeeds on CUDA;
- the full DIV2K validation set produces finite RGB PSNR, Y PSNR, and SSIM;
- interrupted training can resume with the optimizer, scheduler, global step, and best metric restored;
- `latest.pth`, `best_y_psnr.pth`, numbered checkpoints, and weights-only export are created as specified;
- no camera-specific degradation or camera/video dependency appears in the active training path;
- dataset and generated training artifacts remain excluded from Git.

## 12. Deferred Work

The following work belongs to later phases:

- camera-specific synthetic degradation;
- validation on RealSR, DRealSR, or target-camera images;
- temporal consistency for live video;
- real-time 720p benchmarking;
- INT8 calibration or quantization-aware training;
- ONNX export and FPGA integration.
