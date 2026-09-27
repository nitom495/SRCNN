from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer


from torch.utils.data import (
    DataLoader,
    Dataset,
)

from .metrics import (
    crop_border,
    psnr,
    rgb_to_y,
    ssim,
)

from .config import TrainingConfig


import random

import time
from collections.abc import Iterator

from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter

from .checkpoint import (
    export_weights,
    save_checkpoint,
)


def seed_everything(
    seed: int,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(
    worker_id: int,
) -> None:
    worker_seed = (
        torch.initial_seed()
        % (2**32)
    )

    random.seed(worker_seed)


def make_train_loader(
    dataset: Dataset,
    config: TrainingConfig,
) -> DataLoader:
    generator = torch.Generator() #控制打乱顺序
    generator.manual_seed(
        config.seed
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=(
            config.num_workers > 0
        ),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def make_validation_loader(
    dataset: Dataset,
    config: TrainingConfig,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=(
            config.num_workers > 0
        ),
    )
def train_step(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    lr: torch.Tensor,
    hr: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()

    lr = lr.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    hr = hr.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    optimizer.zero_grad(
        set_to_none=True,
    )

    sr = model(lr)

    if sr.shape != hr.shape:
        raise ValueError(
            f"model output {tuple(sr.shape)} "
            f"does not match HR {tuple(hr.shape)}"
        )

    loss = loss_fn(
        sr,
        hr,
    )

    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"non-finite training loss: "
            f"{float(loss.detach())}"
        )

    loss.backward()
    optimizer.step()

    return float(
        loss.detach()
    )


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scale: int,
) -> dict[str, float]:
    model.eval()

    totals = {
        "l1": 0.0,
        "rgb_psnr": 0.0,
        "y_psnr": 0.0,
        "ssim": 0.0,
    }
    image_count = 0

    for lr, hr in loader:
        lr = lr.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        hr = hr.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        sr = model(lr).clamp(
            0.0,
            1.0,
        )

        if sr.shape != hr.shape:
            raise ValueError(
                f"validation output "
                f"{tuple(sr.shape)} "
                f"does not match HR "
                f"{tuple(hr.shape)}"
            )

        sr = crop_border(
            sr,
            border=scale,
        )
        hr = crop_border(
            hr,
            border=scale,
        )

        sr_y = rgb_to_y(sr)
        hr_y = rgb_to_y(hr)

        batch_size = hr.shape[0]

        totals["l1"] += (
            float(
                torch.nn.functional.l1_loss(
                    sr,
                    hr,
                )
            )
            * batch_size
        )
        totals["rgb_psnr"] += (
            float(psnr(sr, hr))
            * batch_size
        )
        totals["y_psnr"] += (
            float(psnr(sr_y, hr_y))
            * batch_size
        )
        totals["ssim"] += (
            float(ssim(sr_y, hr_y))
            * batch_size
        )

        image_count += batch_size

    if image_count == 0:
        raise ValueError(
            "validation loader is empty"
        )

    return {
        name: total / image_count
        for name, total in totals.items()
    }


def _next_batch(
    iterator: Iterator[
        tuple[torch.Tensor, torch.Tensor]
    ],
    loader: DataLoader,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    Iterator[
        tuple[torch.Tensor, torch.Tensor]
    ],
]:
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)

    return batch, iterator


def run_training(
    *,
    config: TrainingConfig,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    start_step: int = 0,
    best_y_psnr: float = float("-inf"),
) -> None:
    loss_fn = nn.L1Loss()
    train_iterator = iter(train_loader)

    interval_start = time.perf_counter()
    interval_images = 0
    running_loss = 0.0
    running_steps = 0

    for global_step in range(
        start_step + 1,
        config.max_steps + 1,
    ):
        (lr, hr), train_iterator = _next_batch(
            train_iterator,
            train_loader,
        )

        loss = train_step(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr=lr,
            hr=hr,
            device=device,
        )

        scheduler.step()

        running_loss += loss
        running_steps += 1
        interval_images += lr.shape[0]

        if (
            global_step
            % config.log_interval
            == 0
        ):
            elapsed = (
                time.perf_counter()
                - interval_start
            )
            mean_loss = (
                running_loss
                / running_steps
            )
            throughput = (
                interval_images
                / elapsed
            )
            learning_rate = (
                optimizer
                .param_groups[0]["lr"]
            )

            writer.add_scalar(
                "train/l1",
                mean_loss,
                global_step,
            )
            writer.add_scalar(
                "train/learning_rate",
                learning_rate,
                global_step,
            )
            writer.add_scalar(
                "train/images_per_second",
                throughput,
                global_step,
            )
            writer.add_scalar(
                "train/interval_seconds",
                elapsed,
                global_step,
            )

            print(
                f"step={global_step}/"
                f"{config.max_steps} "
                f"loss={mean_loss:.6f} "
                f"lr={learning_rate:.8f} "
                f"images_per_second="
                f"{throughput:.1f}"
            )

            interval_start = time.perf_counter()
            interval_images = 0
            running_loss = 0.0
            running_steps = 0

        should_validate = (
            global_step
            % config.eval_interval
            == 0
            or global_step
            == config.max_steps
        )

        if should_validate:
            metrics = validate(
                model=model,
                loader=validation_loader,
                device=device,
                scale=config.scale,
            )

            for name, value in metrics.items():
                writer.add_scalar(
                    f"validation/{name}",
                    value,
                    global_step,
                )

            print(
                f"validation step={global_step} "
                f"l1={metrics['l1']:.6f} "
                f"rgb_psnr="
                f"{metrics['rgb_psnr']:.4f} "
                f"y_psnr="
                f"{metrics['y_psnr']:.4f} "
                f"ssim={metrics['ssim']:.6f}"
            )

            if (
                metrics["y_psnr"]
                > best_y_psnr
            ):
                best_y_psnr = (
                    metrics["y_psnr"]
                )

                save_checkpoint(
                    config.output_dir
                    / "best_y_psnr.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    best_y_psnr=best_y_psnr,
                    config=config.as_dict(),
                )

        should_checkpoint = (
            global_step
            % config.checkpoint_interval
            == 0
            or global_step
            == config.max_steps
        )

        if should_checkpoint:
            checkpoint_arguments = {
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "global_step": global_step,
                "best_y_psnr": best_y_psnr,
                "config": config.as_dict(),
            }

            save_checkpoint(
                config.output_dir
                / "latest.pth",
                **checkpoint_arguments,
            )

            save_checkpoint(
                config.output_dir
                / (
                    f"checkpoint_"
                    f"{global_step:06d}.pth"
                ),
                **checkpoint_arguments,
            )

            export_weights(
                config.output_dir
                / "model_weights.pth",
                model,
            )

            writer.flush()