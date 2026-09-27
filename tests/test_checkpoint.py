from pathlib import Path

import torch

from sr.checkpoint import (
    export_weights,
    load_checkpoint,
    save_checkpoint,
)
from sr.model import FastSRNet

import random

def make_training_objects():
    model = FastSRNet()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=2e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=1e-6,
        )
    )

    return model, optimizer, scheduler


def test_checkpoint_restores_complete_training_state(
    tmp_path: Path,
):
    model, optimizer, scheduler = (
        make_training_objects()
    )

    input_tensor = torch.rand(
        1,
        3,
        8,
        8,
    )

    model(input_tensor).mean().backward()
    optimizer.step()
    scheduler.step()

    path = tmp_path / "nested" / "latest.pth"

    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=7,
        best_y_psnr=31.25,
        config={"scale": 2},
    )

    saved_parameters = {
        name: value.detach().clone()
        for name, value
        in model.state_dict().items()
    }

    restored_model, restored_optimizer, restored_scheduler = (
        make_training_objects()
    )

    state = load_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        torch.device("cpu"),
    )

    assert state.global_step == 7
    assert state.best_y_psnr == 31.25

    assert (
        restored_scheduler.last_epoch
        == scheduler.last_epoch
    )

    assert len(
        restored_optimizer.state_dict()["state"]
    ) == len(
        optimizer.state_dict()["state"]
    )

    for name, value in restored_model.state_dict().items():
        assert torch.equal(
            value,
            saved_parameters[name],
        )


def test_export_weights_writes_plain_state_dictionary(
    tmp_path: Path,
):
    model = FastSRNet()
    path = tmp_path / "weights" / "model.pth"

    export_weights(
        path,
        model,
    )

    loaded = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    assert set(loaded) == set(
        model.state_dict()
    )

def test_checkpoint_restores_random_number_states(
    tmp_path: Path,
):
    random.seed(123)
    torch.manual_seed(123)

    model, optimizer, scheduler = (
        make_training_objects()
    )

    path = tmp_path / "rng_state.pth"

    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=1,
        best_y_psnr=30.0,
        config={"scale": 2},
    )

    expected_python_value = random.random()
    expected_torch_value = torch.rand(4)

    random.seed(999)
    torch.manual_seed(999)

    load_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        torch.device("cpu"),
    )

    assert random.random() == expected_python_value
    assert torch.equal(
        torch.rand(4),
        expected_torch_value,
    )