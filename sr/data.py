from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


from PIL import Image

import random
from collections.abc import Callable, Sequence

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop

from .config import TrainingConfig


@dataclass(frozen=True)
class PairSource:
    name: str
    lr_dir: Path
    hr_dir: Path
    expected_count: int

@dataclass(frozen=True)
class ImagePair:
    source: str
    image_id: str
    lr_path: Path
    hr_path: Path

def collect_pairs(
    sources: Sequence[PairSource],
    scale: int,
) -> list[ImagePair]:
    collected: list[ImagePair] = []

    for source in sources:



        if not source.lr_dir.is_dir():
            raise FileNotFoundError(
                f"LR directory does not exist: {source.lr_dir}"
            )

        if not source.hr_dir.is_dir():
            raise FileNotFoundError(
                f"HR directory does not exist: {source.hr_dir}"
            )





        lr_paths = sorted(
            source.lr_dir.glob(f"*x{scale}.png")
        )

        if len(lr_paths) != source.expected_count:
            raise ValueError(
                f"{source.name} expected "
                f"{source.expected_count} pairs, "
                f"found {len(lr_paths)}"
            )




        seen_ids: set[str] = set()

        for lr_path in lr_paths:
            image_id = lr_path.stem.removesuffix(
                f"x{scale}"
            )

            if image_id in seen_ids:
                raise ValueError(
                    f"duplicate image id in "
                    f"{source.name}: {image_id}"
                )

            seen_ids.add(image_id)
            hr_path = source.hr_dir / f"{image_id}.png"

            if not hr_path.is_file():
                raise FileNotFoundError(
                    f"missing HR image for "
                    f"{source.name}/{image_id}: {hr_path}"
                )

            with Image.open(lr_path) as lr_image:
                lr_size = lr_image.size

            with Image.open(hr_path) as hr_image:
                hr_size = hr_image.size

            expected_hr_size = (
                lr_size[0] * scale,
                lr_size[1] * scale,
            )

            if hr_size != expected_hr_size:
                raise ValueError(
                    f"{source.name}/{image_id} must have "
                    f"exact x{scale} dimensions: "
                    f"LR={lr_size}, HR={hr_size}"
                )

            collected.append(
                ImagePair(
                    source=source.name,
                    image_id=image_id,
                    lr_path=lr_path,
                    hr_path=hr_path,
                )
            )

    if not collected:
        raise ValueError(
            "no paired images were found"
        )

    return collected

#对应变换
class PairedPatchTransform:
    def __init__(
        self,
        lr_patch_size: int = 128,
        scale: int = 2,
    ) -> None:
        self.lr_patch_size = lr_patch_size
        self.scale = scale

    def __call__(
        self,
        lr_image: Image.Image,
        hr_image: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            lr_image.width < self.lr_patch_size
            or lr_image.height < self.lr_patch_size
        ):
            raise ValueError(
                f"LR image {lr_image.size} is smaller "
                f"than patch size {self.lr_patch_size}"
            )

        top, left, height, width = (
            RandomCrop.get_params(
                lr_image,
                output_size=(
                    self.lr_patch_size,
                    self.lr_patch_size,
                ),
            )
        )

        lr_image = TF.crop(
            lr_image,
            top,
            left,
            height,
            width,
        )
        hr_image = TF.crop(
            hr_image,
            top * self.scale,
            left * self.scale,
            height * self.scale,
            width * self.scale,
        )

        lr = TF.to_tensor(lr_image)
        hr = TF.to_tensor(hr_image)

        if random.random() < 0.5:
            lr = torch.flip(lr, dims=(2,))
            hr = torch.flip(hr, dims=(2,))

        if random.random() < 0.5:
            lr = torch.flip(lr, dims=(1,))
            hr = torch.flip(hr, dims=(1,))

        rotations = random.randint(0, 3)

        lr = torch.rot90(
            lr,
            rotations,
            dims=(1, 2),
        )
        hr = torch.rot90(
            hr,
            rotations,
            dims=(1, 2),
        )

        return lr.contiguous(), hr.contiguous()


#获取对应的训练图片对
class PairedImageDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor]]
):
    def __init__(
        self,
        pairs: Sequence[ImagePair],
        transform: Callable[
            [Image.Image, Image.Image],
            tuple[torch.Tensor, torch.Tensor],
        ]
        | None = None,
    ) -> None:
        self.pairs = list(pairs)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[index]

        with Image.open(pair.lr_path) as image:
            lr_image = image.convert("RGB")

        with Image.open(pair.hr_path) as image:
            hr_image = image.convert("RGB")

        if self.transform is not None:
            return self.transform(
                lr_image,
                hr_image,
            )

        lr = TF.to_tensor(lr_image)
        hr = TF.to_tensor(hr_image)

        return lr, hr

# DIV2K 800 对
#       ├── 合并 → DF2K 3450 对 → 随机 128×128 LR 裁剪
# Flickr2K 2650 对

def build_training_dataset(
    config: TrainingConfig,
) -> PairedImageDataset:
    sources = [
        PairSource(
            name="DIV2K",
            lr_dir=config.div2k_train_lr,
            hr_dir=config.div2k_train_hr,
            expected_count=800,
        ),
        PairSource(
            name="Flickr2K",
            lr_dir=config.flickr2k_train_lr,
            hr_dir=config.flickr2k_train_hr,
            expected_count=2_650,
        ),
    ]

    pairs = collect_pairs(
        sources,
        scale=config.scale,
    )

    transform = PairedPatchTransform(
        lr_patch_size=config.lr_patch_size,
        scale=config.scale,
    )

    return PairedImageDataset(
        pairs=pairs,
        transform=transform,
    )


def build_validation_dataset(
    config: TrainingConfig,
) -> PairedImageDataset:
    source = PairSource(
        name="DIV2K_valid",
        lr_dir=config.div2k_valid_lr,
        hr_dir=config.div2k_valid_hr,
        expected_count=100,
    )

    pairs = collect_pairs(
        [source],
        scale=config.scale,
    )

    return PairedImageDataset(
        pairs=pairs,
    )