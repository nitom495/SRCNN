from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


@dataclass(frozen=True)
class ResumeState:
    global_step: int
    best_y_psnr: float


def _atomic_torch_save(
    payload: Any,
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        payload,
        temporary_path,
    )

    os.replace(
        temporary_path,
        path,
    )


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def _restore_rng_state(
    state: dict[str, Any],
) -> None:
    random.setstate(
        state["python"]
    )
    torch.set_rng_state(
        state["torch"].cpu()
    )

    if (
        "cuda" in state
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            [
                cuda_state.cpu()
                for cuda_state in state["cuda"]
            ]
        )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    global_step: int,
    best_y_psnr: float,
    config: dict[str, Any],
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "best_y_psnr": best_y_psnr,
        "config": config,
        "rng_state": _rng_state(),
    }

    _atomic_torch_save(
        payload,
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
) -> ResumeState:
    payload = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        payload["model"]
    )
    optimizer.load_state_dict(
        payload["optimizer"]
    )
    scheduler.load_state_dict(
        payload["scheduler"]
    )

    _restore_rng_state(
        payload["rng_state"]
    )

    return ResumeState(
        global_step=int(
            payload["global_step"]
        ),
        best_y_psnr=float(
            payload["best_y_psnr"]
        ),
    )


def export_weights(
    path: Path,
    model: nn.Module,
) -> None:
    _atomic_torch_save(
        model.state_dict(),
        path,
    )
