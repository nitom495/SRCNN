from pathlib import Path
from PIL import Image

import pytest
import torch

from sr.config import TrainingConfig

from sr.data import (
    PairSource,
    PairedImageDataset,
    PairedPatchTransform,
    build_training_dataset,
    build_validation_dataset,
    collect_pairs,
)

#创建测试图片
def save_rgb(
    path: Path,
    size: tuple[int, int],  #PIL:(width,height) Pytorch:(height,width)
    value: int = 128,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True) #创建图片目录
    image=Image.new("RGB",size,(value,value,value))
    image.save(path)

#创建假的配对图片
def make_source(
    root: Path,
    name: str,
    image_id: str,
) -> PairSource:
    lr_dir = root / name / "LR"
    hr_dir = root / name / "HR"

    save_rgb(
        lr_dir / f"{image_id}x2.png",
        size=(160, 144),
    )
    save_rgb(
        hr_dir / f"{image_id}.png",
        size=(320, 288),
    )

    return PairSource(
        name=name,
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        expected_count=1,
    )

def test_two_sources_keep_duplicate_numeric_ids_independent(
    tmp_path: Path,
):
    div2k = make_source(
        tmp_path,
        name="DIV2K",
        image_id="0001",
    )
    flickr2k = make_source(
        tmp_path,
        name="Flickr2K",
        image_id="0001",
    )

    pairs = collect_pairs(
        [div2k, flickr2k],
        scale=2,
    )

    assert len(pairs) == 2
    assert {
        pair.source
        for pair in pairs
    } == {"DIV2K", "Flickr2K"}

def test_missing_hr_pair_fails_during_discovery(
    tmp_path: Path,
):
    lr_dir = tmp_path / "LR"
    hr_dir = tmp_path / "HR"

    save_rgb(
        lr_dir / "0001x2.png",
        size=(160, 144),
    )
    hr_dir.mkdir()

    source = PairSource(
        name="broken",
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        expected_count=1,
    )

    with pytest.raises(
        FileNotFoundError,
        match="0001",
    ):
        collect_pairs([source], scale=2)


def test_wrong_scale_pair_fails_during_discovery(
    tmp_path: Path,
):
    lr_dir = tmp_path / "LR"
    hr_dir = tmp_path / "HR"

    save_rgb(
        lr_dir / "0001x2.png",
        size=(160, 144),
    )
    save_rgb(
        hr_dir / "0001.png",
        size=(319, 288),
    )

    source = PairSource(
        name="broken",
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        expected_count=1,
    )

    with pytest.raises(
        ValueError,
        match="exact x2 dimensions",
    ):
        collect_pairs([source], scale=2)


def test_unexpected_source_count_fails(
    tmp_path: Path,
):
    source = make_source(
        tmp_path,
        name="DIV2K",
        image_id="0001",
    )

    wrong_count = PairSource(
        name=source.name,
        lr_dir=source.lr_dir,
        hr_dir=source.hr_dir,
        expected_count=2,
    )

    with pytest.raises(
        ValueError,
        match="expected 2 pairs",
    ):
        collect_pairs([wrong_count], scale=2)

#检查训练时的对应裁剪
def test_training_dataset_returns_aligned_x2_rgb_patches(
    tmp_path: Path,
):
    source = make_source(
        tmp_path,
        name="DIV2K",
        image_id="0001",
    )

    pairs = collect_pairs(
        [source],
        scale=2,
    )

    dataset = PairedImageDataset(
        pairs=pairs,
        transform=PairedPatchTransform(
            lr_patch_size=128,
            scale=2,
        ),
    )

    lr, hr = dataset[0]

    assert lr.shape == (3, 128, 128)
    assert hr.shape == (3, 256, 256)
    assert lr.dtype == torch.float32
    assert hr.dtype == torch.float32
    assert 0.0 <= float(lr.min())
    assert float(lr.max()) <= 1.0
    assert 0.0 <= float(hr.min())
    assert float(hr.max()) <= 1.0

#检查验证时不随机裁剪，返回完整图像
def test_validation_dataset_returns_complete_images(
    tmp_path: Path,
):
    source = make_source(
        tmp_path,
        name="DIV2K_valid",
        image_id="0801",
    )

    pairs = collect_pairs(
        [source],
        scale=2,
    )

    dataset = PairedImageDataset(
        pairs=pairs,
    )

    lr, hr = dataset[0]

    assert lr.shape == (3, 144, 160)
    assert hr.shape == (3, 288, 320)
    assert lr.dtype == torch.float32
    assert hr.dtype == torch.float32

def test_dataset_factories_use_df2k_and_div2k_validation(
    tmp_path: Path,
    monkeypatch,
):
    config = TrainingConfig.from_data_root(
        tmp_path / "dataset"
    )
    calls = []

    def fake_collect_pairs(sources, scale):
        calls.append((list(sources), scale))
        return []

    monkeypatch.setattr(
        "sr.data.collect_pairs",
        fake_collect_pairs,
    )

    training_dataset = build_training_dataset(config)
    validation_dataset = build_validation_dataset(config)

    training_sources, training_scale = calls[0]
    validation_sources, validation_scale = calls[1]

    assert training_scale == 2
    assert validation_scale == 2

    assert [
        (source.name, source.expected_count)
        for source in training_sources
    ] == [
        ("DIV2K", 800),
        ("Flickr2K", 2_650),
    ]

    assert [
        (source.name, source.expected_count)
        for source in validation_sources
    ] == [
        ("DIV2K_valid", 100),
    ]

    assert isinstance(
        training_dataset.transform,
        PairedPatchTransform,
    )
    assert training_dataset.transform.lr_patch_size == 128
    assert validation_dataset.transform is None