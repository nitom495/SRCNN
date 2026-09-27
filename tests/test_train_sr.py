from pathlib import Path

import pytest
import torch

from sr.config import TrainingConfig
from sr.model import FastSRNet
from train_sr import (
    build_parser,
    preflight_model,
)


def test_parser_exposes_training_overrides():
    parser = build_parser()

    arguments = parser.parse_args(
        [
            "--data-root",
            "D:/datasets/DF2K",
            "--output-dir",
            "D:/runs/df2k",
            "--batch-size",
            "32",
            "--num-workers",
            "8",
            "--max-steps",
            "12",
        ]
    )

    assert arguments.data_root == Path(
        "D:/datasets/DF2K"
    )
    assert arguments.output_dir == Path(
        "D:/runs/df2k"
    )
    assert arguments.batch_size == 32
    assert arguments.num_workers == 8
    assert arguments.max_steps == 12


def test_preflight_accepts_matching_finite_output():
    model = FastSRNet()

    lr = torch.rand(
        3,
        128,
        128,
    )
    hr = torch.rand(
        3,
        256,
        256,
    )

    preflight_model(
        model,
        lr,
        hr,
        torch.device("cpu"),
    )


def test_preflight_rejects_mismatched_hr_shape():
    config = TrainingConfig.from_data_root(
        Path("dataset")
    )
    model = FastSRNet(
        config.scale
    )

    lr = torch.rand(
        3,
        128,
        128,
    )
    hr = torch.rand(
        3,
        255,
        256,
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        preflight_model(
            model,
            lr,
            hr,
            torch.device("cpu"),
        )