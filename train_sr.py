from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from sr.checkpoint import load_checkpoint
from sr.config import TrainingConfig
from sr.data import (
    build_training_dataset,
    build_validation_dataset,
)
from sr.model import FastSRNet
from sr.trainer import (
    make_train_loader,
    make_validation_loader,
    run_training,
    seed_everything,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the FP32 DF2K "
            "bicubic RGB x2 SR model"
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/df2k_x2"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=150_000,
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5_000,
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1_337,
    )
    parser.add_argument(
        "--resume",
        type=Path,
    )

    return parser


def config_from_arguments(
    arguments: argparse.Namespace,
) -> TrainingConfig:
    return TrainingConfig.from_data_root(
        arguments.data_root,
        output_dir=arguments.output_dir,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        max_steps=arguments.max_steps,
        eval_interval=arguments.eval_interval,
        checkpoint_interval=(
            arguments.checkpoint_interval
        ),
        log_interval=arguments.log_interval,
        seed=arguments.seed,
    )


@torch.inference_mode()
def preflight_model(
    model: nn.Module,
    lr: torch.Tensor,
    hr: torch.Tensor,
    device: torch.device,
) -> None:
    lr_batch = lr.unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )
    hr_batch = hr.unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )

    sr_batch = model(lr_batch)

    if sr_batch.shape != hr_batch.shape:
        raise ValueError(
            f"model output "
            f"{tuple(sr_batch.shape)} "
            f"does not match HR "
            f"{tuple(hr_batch.shape)}"
        )

    loss = torch.nn.functional.l1_loss(
        sr_batch,
        hr_batch,
    )

    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"preflight produced non-finite loss: "
            f"{float(loss)}"
        )


def main() -> None:
    arguments = build_parser().parse_args()
    config = config_from_arguments(arguments)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for DF2K training "
            "but is not available"
        )

    device = torch.device("cuda:0")

    seed_everything(config.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_dataset = build_training_dataset(
        config
    )
    validation_dataset = (
        build_validation_dataset(config)
    )

    device_name = torch.cuda.get_device_name(
        device
    )

    config.save_json(
        config.output_dir / "config.json",
        metadata={
            "device": device_name,
            "train_pairs": len(train_dataset),
            "validation_pairs": len(
                validation_dataset
            ),
        },
    )

    print(
        f"device={device_name} "
        f"train_pairs={len(train_dataset)} "
        f"validation_pairs="
        f"{len(validation_dataset)} "
        f"batch_size={config.batch_size} "
        f"workers={config.num_workers}"
    )

    model = FastSRNet(
        config.scale
    ).to(
        device=device,
        dtype=torch.float32,
    )

    sample_lr, sample_hr = train_dataset[0]

    preflight_model(
        model,
        sample_lr,
        sample_hr,
        device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=config.max_steps,
            eta_min=(
                config.minimum_learning_rate
            ),
        )
    )

    start_step = 0
    best_y_psnr = float("-inf")

    if arguments.resume is not None:
        state = load_checkpoint(
            arguments.resume,
            model,
            optimizer,
            scheduler,
            device,
        )

        start_step = state.global_step
        best_y_psnr = state.best_y_psnr

        print(
            f"resumed={arguments.resume} "
            f"step={start_step} "
            f"best_y_psnr="
            f"{best_y_psnr:.4f}"
        )

    if start_step >= config.max_steps:
        raise ValueError(
            f"checkpoint step {start_step} "
            f"must be smaller than "
            f"max_steps {config.max_steps}"
        )

    train_loader = make_train_loader(
        train_dataset,
        config,
    )
    validation_loader = (
        make_validation_loader(
            validation_dataset,
            config,
        )
    )

    tensorboard_dir = (
        config.output_dir / "tensorboard"
    )

    with SummaryWriter(
        log_dir=str(tensorboard_dir)
    ) as writer:
        run_training(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            device=device,
            writer=writer,
            start_step=start_step,
            best_y_psnr=best_y_psnr,
        )


if __name__ == "__main__":
    main()