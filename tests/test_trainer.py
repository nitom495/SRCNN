import pytest
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from sr.model import FastSRNet
from sr.trainer import (
    make_train_loader,
    make_validation_loader,
    run_training,
    seed_everything,
    train_step,
    validate,
)

import random

from sr.config import TrainingConfig

from torch.utils.tensorboard import SummaryWriter

def test_train_step_updates_fp32_model_parameters():
    torch.manual_seed(3)

    model = FastSRNet().float()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=2e-4,
    )

    lr = torch.rand(
        2,
        3,
        16,
        16,
    )
    hr = torch.rand(
        2,
        3,
        32,
        32,
    )

    before = (
        model.model[0]
        .weight
        .detach()
        .clone()
    )

    loss = train_step(
        model,
        optimizer,
        nn.L1Loss(),
        lr,
        hr,
        torch.device("cpu"),
    )

    assert loss > 0
    assert (
        model.model[0].weight.dtype
        == torch.float32
    )
    assert not torch.equal(
        before,
        model.model[0].weight,
    )


def test_train_step_rejects_non_finite_loss():
    model = FastSRNet().float()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=2e-4,
    )

    lr = torch.full(
        (1, 3, 16, 16),
        float("nan"),
    )
    hr = torch.rand(
        1,
        3,
        32,
        32,
    )

    with pytest.raises(
        FloatingPointError,
        match="non-finite",
    ):
        train_step(
            model,
            optimizer,
            nn.L1Loss(),
            lr,
            hr,
            torch.device("cpu"),
        )


def test_validate_reports_metrics_for_complete_images():
    torch.manual_seed(5)

    model = FastSRNet().float()
    lr = torch.rand(
        2,
        3,
        20,
        24,
    )

    with torch.no_grad():
        hr = torch.cat(
            [
                model(lr[index:index + 1]).clamp(
                    0.0,
                    1.0,
                )
                for index in range(lr.shape[0])
            ],
            dim=0,
        )
    loader = DataLoader(
        TensorDataset(lr, hr),
        batch_size=1,
        shuffle=False,
    )

    metrics = validate(
        model,
        loader,
        torch.device("cpu"),
        scale=2,
    )

    assert metrics["l1"] == pytest.approx(
        0.0,
        abs=1e-7,
    )
    assert metrics["rgb_psnr"] == float("inf")
    assert metrics["y_psnr"] == float("inf")
    assert metrics["ssim"] == pytest.approx(
        1.0,
        abs=1e-5,
    )

def test_seed_everything_repeats_python_and_torch_values():
    seed_everything(11)

    first_python = random.random()
    first_torch = torch.rand(4)

    seed_everything(11)

    second_python = random.random()
    second_torch = torch.rand(4)

    assert first_python == second_python
    assert torch.equal(
        first_torch,
        second_torch,
    )


def test_loaders_use_training_and_validation_settings(
    tmp_path,
):
    lr = torch.rand(
        5,
        3,
        8,
        8,
    )
    hr = torch.rand(
        5,
        3,
        16,
        16,
    )
    dataset = TensorDataset(lr, hr)

    config = TrainingConfig.from_data_root(
        tmp_path / "dataset",
        batch_size=2,
        num_workers=0,
    )

    train_loader = make_train_loader(
        dataset,
        config,
    )
    validation_loader = make_validation_loader(
        dataset,
        config,
    )

    assert train_loader.batch_size == 2
    assert train_loader.drop_last is True
    assert train_loader.pin_memory is True
    assert train_loader.persistent_workers is False

    assert validation_loader.batch_size == 1
    assert validation_loader.drop_last is False
    assert validation_loader.pin_memory is True
    assert validation_loader.persistent_workers is False

def test_run_training_writes_checkpoint_artifacts(
    tmp_path,
):
    torch.manual_seed(19)

    config = TrainingConfig.from_data_root(
        tmp_path / "dataset",
        output_dir=tmp_path / "output",
        batch_size=1,
        num_workers=0,
        max_steps=2,
        eval_interval=1,
        checkpoint_interval=1,
        log_interval=1,
    )

    model = FastSRNet().float()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.max_steps,
            eta_min=config.minimum_learning_rate,
        )
    )

    train_lr = torch.rand(
        2,
        3,
        12,
        12,
    )
    train_hr = torch.rand(
        2,
        3,
        24,
        24,
    )
    validation_lr = torch.rand(
        1,
        3,
        12,
        12,
    )
    validation_hr = torch.rand(
        1,
        3,
        24,
        24,
    )

    train_loader = DataLoader(
        TensorDataset(
            train_lr,
            train_hr,
        ),
        batch_size=1,
        shuffle=False,
    )
    validation_loader = DataLoader(
        TensorDataset(
            validation_lr,
            validation_hr,
        ),
        batch_size=1,
        shuffle=False,
    )

    with SummaryWriter(
        log_dir=str(
            tmp_path / "tensorboard"
        )
    ) as writer:
        run_training(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=torch.device("cpu"),
            writer=writer,
        )

    assert (
        config.output_dir / "latest.pth"
    ).is_file()
    assert (
        config.output_dir / "best_y_psnr.pth"
    ).is_file()
    assert (
        config.output_dir
        / "checkpoint_000001.pth"
    ).is_file()
    assert (
        config.output_dir
        / "checkpoint_000002.pth"
    ).is_file()
    assert (
        config.output_dir / "model_weights.pth"
    ).is_file()

    checkpoint = torch.load(
        config.output_dir / "latest.pth",
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["global_step"] == 2
    assert scheduler.last_epoch == 2