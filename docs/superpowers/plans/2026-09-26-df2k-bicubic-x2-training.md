# DF2K Bicubic RGB x2 Super-Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-file prototype with a tested FP32 training pipeline for the unchanged lightweight RGB x2 PixelShuffle model, using paired DF2K bicubic data and DIV2K validation.

**Architecture:** Keep the 7,124-parameter network unchanged and split the pipeline into focused modules for configuration, paired data, metrics, checkpoints, and training. The entry script validates CUDA and dataset layout, then performs step-based training with TensorBoard, deterministic validation, best/latest/numbered checkpoints, and resumable state.

**Tech Stack:** Python 3.11, PyTorch with CUDA 12.8 support, torchvision, Pillow, TensorBoard, pytest, NVIDIA RTX 4090.

---

## File Map

- Create `sr/__init__.py`: package exports.
- Create `sr/config.py`: immutable training configuration and JSON serialization.
- Create `sr/model.py`: unchanged lightweight x2 model.
- Create `sr/data.py`: DF2K pair discovery, validation, aligned patch sampling, and dataset factories.
- Create `sr/metrics.py`: border shave, RGB-to-Y conversion, PSNR, and SSIM.
- Create `sr/checkpoint.py`: resumable and weights-only checkpoint operations.
- Create `sr/trainer.py`: seeds, loader construction, FP32 training step, validation, and step-based loop.
- Replace `train_sr.py`: command-line entry point and CUDA/data preflight.
- Modify `.gitignore`: exclude generated checkpoints and TensorBoard runs.
- Create `README.md`: exact data layout and training commands.
- Create `tests/`: focused unit and integration tests for every module.

### Task 1: Package, configuration, and test foundation

**Files:**
- Create: `sr/__init__.py`
- Create: `sr/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing configuration tests**

Create `tests/test_config.py`:

```python
import json
from pathlib import Path

from sr.config import TrainingConfig


def test_training_config_has_quality_training_defaults(tmp_path: Path):
    config = TrainingConfig.from_data_root(tmp_path / "dataset")

    assert config.scale == 2
    assert config.lr_patch_size == 128
    assert config.batch_size == 64
    assert config.max_steps == 150_000
    assert config.learning_rate == 2e-4
    assert config.minimum_learning_rate == 1e-6
    assert config.div2k_train_hr == tmp_path / "dataset" / "DIV2K_train_HR"
    assert config.flickr2k_train_lr == tmp_path / "dataset" / "Flickr2K_train_LR_bicubic" / "X2"


def test_save_json_converts_paths_and_writes_resolved_config(tmp_path: Path):
    config = TrainingConfig.from_data_root(
        tmp_path / "dataset",
        output_dir=tmp_path / "output",
    )

    config.save_json(
        tmp_path / "resolved.json",
        metadata={"train_pairs": 3_450, "validation_pairs": 100},
    )
    saved = json.loads((tmp_path / "resolved.json").read_text(encoding="utf-8"))

    assert saved["data_root"] == str(tmp_path / "dataset")
    assert saved["output_dir"] == str(tmp_path / "output")
    assert saved["max_steps"] == 150_000
    assert saved["train_pairs"] == 3_450
    assert saved["validation_pairs"] == 100
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_config.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'sr'`.

- [ ] **Step 3: Add the package and configuration implementation**

Create `sr/__init__.py`:

```python
from .config import TrainingConfig

__all__ = ["TrainingConfig"]
```

Create `sr/config.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    data_root: Path
    output_dir: Path
    div2k_train_lr: Path
    div2k_train_hr: Path
    flickr2k_train_lr: Path
    flickr2k_train_hr: Path
    div2k_valid_lr: Path
    div2k_valid_hr: Path
    scale: int = 2
    lr_patch_size: int = 128
    batch_size: int = 64
    num_workers: int = 4
    max_steps: int = 150_000
    learning_rate: float = 2e-4
    minimum_learning_rate: float = 1e-6
    eval_interval: int = 1_000
    checkpoint_interval: int = 5_000
    log_interval: int = 100
    seed: int = 1_337

    @classmethod
    def from_data_root(
        cls,
        data_root: Path,
        output_dir: Path = Path("outputs/df2k_x2"),
        **overrides: Any,
    ) -> "TrainingConfig":
        data_root = Path(data_root)
        config = cls(
            data_root=data_root,
            output_dir=Path(output_dir),
            div2k_train_lr=data_root / "DIV2K_train_LR_bicubic" / "X2",
            div2k_train_hr=data_root / "DIV2K_train_HR",
            flickr2k_train_lr=data_root / "Flickr2K_train_LR_bicubic" / "X2",
            flickr2k_train_hr=data_root / "Flickr2K_HR",
            div2k_valid_lr=data_root / "DIV2K_valid_LR_bicubic" / "X2",
            div2k_valid_hr=data_root / "DIV2K_valid_HR",
        )
        return replace(config, **overrides)

    def as_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }

    def save_json(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.as_dict()
        if metadata is not None:
            payload.update(metadata)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
```

- [ ] **Step 4: Run the configuration tests**

Run:

```powershell
python -m pytest tests/test_config.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit the configuration foundation**

```powershell
git add sr/__init__.py sr/config.py tests/test_config.py
git commit -m "feat: add SR training configuration"
```

### Task 2: Preserve the lightweight x2 model as a tested module

**Files:**
- Create: `sr/model.py`
- Modify: `sr/__init__.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write failing model contract tests**

Create `tests/test_model.py`:

```python
import torch

from sr.model import FastSRNet


def test_model_outputs_rgb_at_twice_the_input_size():
    model = FastSRNet(scale=2)
    input_tensor = torch.rand(2, 3, 24, 32)

    output = model(input_tensor)

    assert output.shape == (2, 3, 48, 64)


def test_model_preserves_existing_parameter_count():
    model = FastSRNet(scale=2)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert parameter_count == 7_124


def test_model_rejects_unsupported_scale():
    try:
        FastSRNet(scale=3)
    except ValueError as error:
        assert "scale=2" in str(error)
    else:
        raise AssertionError("FastSRNet accepted an unsupported scale")
```

- [ ] **Step 2: Run the tests to verify the missing module failure**

Run:

```powershell
python -m pytest tests/test_model.py -v
```

Expected: collection fails because `sr.model` does not exist.

- [ ] **Step 3: Implement the unchanged model**

Create `sr/model.py`:

```python
import torch
from torch import nn


class FastSRNet(nn.Module):
    def __init__(self, scale: int = 2) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError("FastSRNet currently supports only scale=2")
        self.scale = scale
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3 * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor)
```

Replace `sr/__init__.py` with:

```python
from .config import TrainingConfig
from .model import FastSRNet

__all__ = ["FastSRNet", "TrainingConfig"]
```

- [ ] **Step 4: Run the model tests**

Run:

```powershell
python -m pytest tests/test_model.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit the model extraction**

```powershell
git add sr/__init__.py sr/model.py tests/test_model.py
git commit -m "refactor: extract lightweight x2 model"
```

### Task 3: Build and validate paired DF2K datasets

**Files:**
- Create: `sr/data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write failing pair, shape, and error tests**

Create `tests/test_data.py`:

```python
from pathlib import Path

import pytest
import torch
from PIL import Image

from sr.data import PairSource, PairedImageDataset, PairedPatchTransform, collect_pairs


def save_rgb(path: Path, size: tuple[int, int], value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (value, value, value)).save(path)


def make_source(root: Path, name: str, image_id: str) -> PairSource:
    lr_dir = root / name / "LR"
    hr_dir = root / name / "HR"
    save_rgb(lr_dir / f"{image_id}x2.png", (160, 144))
    save_rgb(hr_dir / f"{image_id}.png", (320, 288))
    return PairSource(name=name, lr_dir=lr_dir, hr_dir=hr_dir, expected_count=1)


def test_two_sources_keep_duplicate_numeric_ids_independent(tmp_path: Path):
    div2k = make_source(tmp_path, "DIV2K", "0001")
    flickr2k = make_source(tmp_path, "Flickr2K", "0001")

    pairs = collect_pairs([div2k, flickr2k], scale=2)

    assert len(pairs) == 2
    assert {pair.source for pair in pairs} == {"DIV2K", "Flickr2K"}


def test_training_dataset_returns_aligned_x2_rgb_patches(tmp_path: Path):
    source = make_source(tmp_path, "DIV2K", "0001")
    dataset = PairedImageDataset(
        collect_pairs([source], scale=2),
        transform=PairedPatchTransform(lr_patch_size=128, scale=2),
    )

    lr, hr = dataset[0]

    assert lr.shape == (3, 128, 128)
    assert hr.shape == (3, 256, 256)
    assert lr.dtype == torch.float32
    assert hr.dtype == torch.float32
    assert 0.0 <= float(lr.min()) <= float(lr.max()) <= 1.0


def test_validation_dataset_returns_complete_images(tmp_path: Path):
    source = make_source(tmp_path, "DIV2K_valid", "0801")
    dataset = PairedImageDataset(collect_pairs([source], scale=2))

    lr, hr = dataset[0]

    assert lr.shape == (3, 144, 160)
    assert hr.shape == (3, 288, 320)


def test_missing_hr_pair_fails_during_discovery(tmp_path: Path):
    lr_dir = tmp_path / "LR"
    hr_dir = tmp_path / "HR"
    save_rgb(lr_dir / "0001x2.png", (160, 144))
    hr_dir.mkdir()
    source = PairSource("broken", lr_dir, hr_dir, expected_count=1)

    with pytest.raises(FileNotFoundError, match="0001"):
        collect_pairs([source], scale=2)


def test_wrong_scale_pair_fails_during_discovery(tmp_path: Path):
    lr_dir = tmp_path / "LR"
    hr_dir = tmp_path / "HR"
    save_rgb(lr_dir / "0001x2.png", (160, 144))
    save_rgb(hr_dir / "0001.png", (319, 288))
    source = PairSource("broken", lr_dir, hr_dir, expected_count=1)

    with pytest.raises(ValueError, match="exact x2 dimensions"):
        collect_pairs([source], scale=2)


def test_unexpected_source_count_fails(tmp_path: Path):
    source = make_source(tmp_path, "DIV2K", "0001")
    wrong_count = PairSource(source.name, source.lr_dir, source.hr_dir, expected_count=2)

    with pytest.raises(ValueError, match="expected 2 pairs"):
        collect_pairs([wrong_count], scale=2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_data.py -v
```

Expected: collection fails because `sr.data` does not exist.

- [ ] **Step 3: Implement pair discovery and datasets**

Create `sr/data.py`:

```python
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torchvision.transforms.functional as TF
from PIL import Image
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


def collect_pairs(sources: Sequence[PairSource], scale: int) -> list[ImagePair]:
    collected: list[ImagePair] = []
    for source in sources:
        if not source.lr_dir.is_dir():
            raise FileNotFoundError(f"LR directory does not exist: {source.lr_dir}")
        if not source.hr_dir.is_dir():
            raise FileNotFoundError(f"HR directory does not exist: {source.hr_dir}")
        lr_paths = sorted(source.lr_dir.glob(f"*x{scale}.png"))
        if len(lr_paths) != source.expected_count:
            raise ValueError(
                f"{source.name} expected {source.expected_count} pairs, found {len(lr_paths)}"
            )
        seen_ids: set[str] = set()
        for lr_path in lr_paths:
            image_id = lr_path.stem.removesuffix(f"x{scale}")
            if image_id in seen_ids:
                raise ValueError(f"duplicate image id in {source.name}: {image_id}")
            seen_ids.add(image_id)
            hr_path = source.hr_dir / f"{image_id}.png"
            if not hr_path.is_file():
                raise FileNotFoundError(f"missing HR image for {source.name}/{image_id}: {hr_path}")
            with Image.open(lr_path) as lr_image, Image.open(hr_path) as hr_image:
                expected_hr_size = (lr_image.width * scale, lr_image.height * scale)
                if hr_image.size != expected_hr_size:
                    raise ValueError(
                        f"{source.name}/{image_id} must have exact x{scale} dimensions: "
                        f"LR={lr_image.size}, HR={hr_image.size}"
                    )
            collected.append(ImagePair(source.name, image_id, lr_path, hr_path))
    if not collected:
        raise ValueError("no paired images were found")
    return collected


class PairedPatchTransform:
    def __init__(self, lr_patch_size: int = 128, scale: int = 2) -> None:
        self.lr_patch_size = lr_patch_size
        self.scale = scale

    def __call__(self, lr_image: Image.Image, hr_image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if lr_image.width < self.lr_patch_size or lr_image.height < self.lr_patch_size:
            raise ValueError(
                f"LR image {lr_image.size} is smaller than patch size {self.lr_patch_size}"
            )
        top, left, height, width = RandomCrop.get_params(
            lr_image,
            output_size=(self.lr_patch_size, self.lr_patch_size),
        )
        lr_image = TF.crop(lr_image, top, left, height, width)
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
        lr = torch.rot90(lr, rotations, dims=(1, 2))
        hr = torch.rot90(hr, rotations, dims=(1, 2))
        return lr.contiguous(), hr.contiguous()


class PairedImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        pairs: Sequence[ImagePair],
        transform: Callable[[Image.Image, Image.Image], tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        self.pairs = list(pairs)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[index]
        with Image.open(pair.lr_path) as image:
            lr_image = image.convert("RGB")
        with Image.open(pair.hr_path) as image:
            hr_image = image.convert("RGB")
        if self.transform is not None:
            return self.transform(lr_image, hr_image)
        return TF.to_tensor(lr_image), TF.to_tensor(hr_image)


def build_training_dataset(config: TrainingConfig) -> PairedImageDataset:
    sources = [
        PairSource("DIV2K", config.div2k_train_lr, config.div2k_train_hr, 800),
        PairSource("Flickr2K", config.flickr2k_train_lr, config.flickr2k_train_hr, 2_650),
    ]
    return PairedImageDataset(
        collect_pairs(sources, config.scale),
        PairedPatchTransform(config.lr_patch_size, config.scale),
    )


def build_validation_dataset(config: TrainingConfig) -> PairedImageDataset:
    source = PairSource("DIV2K_valid", config.div2k_valid_lr, config.div2k_valid_hr, 100)
    return PairedImageDataset(collect_pairs([source], config.scale))
```

- [ ] **Step 4: Run the data tests**

Run:

```powershell
python -m pytest tests/test_data.py -v
```

Expected: `6 passed`.

- [ ] **Step 5: Commit paired DF2K data support**

```powershell
git add sr/data.py tests/test_data.py
git commit -m "feat: add validated DF2K paired datasets"
```

### Task 4: Implement reproducible SR validation metrics

**Files:**
- Create: `sr/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing metric tests**

Create `tests/test_metrics.py`:

```python
import math

import pytest
import torch

from sr.metrics import crop_border, psnr, rgb_to_y, ssim


def test_crop_border_removes_scale_pixels_from_every_side():
    image = torch.zeros(1, 3, 20, 30)

    cropped = crop_border(image, border=2)

    assert cropped.shape == (1, 3, 16, 26)


def test_rgb_to_y_uses_documented_bt601_conversion():
    black = torch.zeros(1, 3, 1, 1)
    white = torch.ones(1, 3, 1, 1)

    assert float(rgb_to_y(black)) == pytest.approx(16.0 / 255.0, abs=1e-6)
    assert float(rgb_to_y(white)) == pytest.approx(235.0 / 255.0, abs=1e-6)


def test_psnr_matches_known_error():
    prediction = torch.zeros(1, 3, 16, 16)
    target = torch.full_like(prediction, 0.1)

    value = psnr(prediction, target)

    assert float(value) == pytest.approx(20.0, abs=1e-5)


def test_ssim_is_one_for_identical_images_and_lower_for_different_images():
    generator = torch.Generator().manual_seed(7)
    target = torch.rand(2, 1, 32, 32, generator=generator)
    different = torch.clamp(target + 0.2, 0.0, 1.0)

    identical_value = ssim(target, target)
    different_value = ssim(different, target)

    assert math.isfinite(float(different_value))
    assert float(identical_value) == pytest.approx(1.0, abs=1e-5)
    assert float(different_value) < float(identical_value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_metrics.py -v
```

Expected: collection fails because `sr.metrics` does not exist.

- [ ] **Step 3: Implement border shave, PSNR, RGB-to-Y, and SSIM**

Create `sr/metrics.py`:

```python
import torch
import torch.nn.functional as F


def crop_border(image: torch.Tensor, border: int) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"expected NCHW tensor, got shape {tuple(image.shape)}")
    if border < 0:
        raise ValueError("border must be non-negative")
    if border == 0:
        return image
    if image.shape[-2] <= 2 * border or image.shape[-1] <= 2 * border:
        raise ValueError("border is too large for image dimensions")
    return image[..., border:-border, border:-border]


def rgb_to_y(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"expected NCHW RGB tensor, got shape {tuple(image.shape)}")
    coefficients = image.new_tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1)
    return (image * coefficients).sum(dim=1, keepdim=True) / 255.0 + 16.0 / 255.0


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    values = torch.where(
        mse == 0,
        torch.full_like(mse, float("inf")),
        10.0 * torch.log10(1.0 / mse),
    )
    return values.mean()


def _gaussian_window(channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coordinates = torch.arange(11, device=device, dtype=dtype) - 5
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, 11, 11).contiguous()


def ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if prediction.ndim != 4 or min(prediction.shape[-2:]) < 11:
        raise ValueError("SSIM expects NCHW images at least 11x11")
    channels = prediction.shape[1]
    window = _gaussian_window(channels, prediction.device, prediction.dtype)
    mean_prediction = F.conv2d(prediction, window, groups=channels)
    mean_target = F.conv2d(target, window, groups=channels)
    mean_prediction_sq = mean_prediction.square()
    mean_target_sq = mean_target.square()
    mean_product = mean_prediction * mean_target
    variance_prediction = F.conv2d(prediction.square(), window, groups=channels) - mean_prediction_sq
    variance_target = F.conv2d(target.square(), window, groups=channels) - mean_target_sq
    covariance = F.conv2d(prediction * target, window, groups=channels) - mean_product
    c1 = 0.01**2
    c2 = 0.03**2
    score = (
        (2 * mean_product + c1)
        * (2 * covariance + c2)
        / ((mean_prediction_sq + mean_target_sq + c1) * (variance_prediction + variance_target + c2))
    )
    return score.flatten(1).mean(dim=1).mean()
```

- [ ] **Step 4: Run the metric tests**

Run:

```powershell
python -m pytest tests/test_metrics.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit validation metrics**

```powershell
git add sr/metrics.py tests/test_metrics.py
git commit -m "feat: add SR validation metrics"
```

### Task 5: Add complete resumable checkpoints

**Files:**
- Create: `sr/checkpoint.py`
- Create: `tests/test_checkpoint.py`

- [ ] **Step 1: Write failing checkpoint round-trip tests**

Create `tests/test_checkpoint.py`:

```python
from pathlib import Path

import torch

from sr.checkpoint import export_weights, load_checkpoint, save_checkpoint
from sr.model import FastSRNet


def make_training_objects():
    model = FastSRNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    return model, optimizer, scheduler


def test_checkpoint_restores_model_optimizer_scheduler_and_metadata(tmp_path: Path):
    model, optimizer, scheduler = make_training_objects()
    input_tensor = torch.rand(1, 3, 8, 8)
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
    saved_parameters = {name: value.detach().clone() for name, value in model.state_dict().items()}

    restored_model, restored_optimizer, restored_scheduler = make_training_objects()
    state = load_checkpoint(path, restored_model, restored_optimizer, restored_scheduler, torch.device("cpu"))

    assert state.global_step == 7
    assert state.best_y_psnr == 31.25
    assert restored_scheduler.last_epoch == scheduler.last_epoch
    for name, value in restored_model.state_dict().items():
        assert torch.equal(value, saved_parameters[name])


def test_export_weights_writes_plain_state_dictionary(tmp_path: Path):
    model = FastSRNet()
    path = tmp_path / "weights" / "model.pth"

    export_weights(path, model)
    loaded = torch.load(path, map_location="cpu", weights_only=True)

    assert set(loaded) == set(model.state_dict())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_checkpoint.py -v
```

Expected: collection fails because `sr.checkpoint` does not exist.

- [ ] **Step 3: Implement atomic checkpoints and RNG capture**

Create `sr/checkpoint.py`:

```python
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


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


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
    _atomic_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "best_y_psnr": best_y_psnr,
            "config": config,
            "rng_state": _rng_state(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
) -> ResumeState:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    _restore_rng_state(payload["rng_state"])
    return ResumeState(
        global_step=int(payload["global_step"]),
        best_y_psnr=float(payload["best_y_psnr"]),
    )


def export_weights(path: Path, model: nn.Module) -> None:
    _atomic_torch_save(model.state_dict(), path)
```

- [ ] **Step 4: Run the checkpoint tests**

Run:

```powershell
python -m pytest tests/test_checkpoint.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit checkpoint support**

```powershell
git add sr/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: add resumable SR checkpoints"
```

### Task 6: Implement FP32 training, validation, and the step-based loop

**Files:**
- Create: `sr/trainer.py`
- Create: `tests/test_trainer.py`

- [ ] **Step 1: Write failing training and validation tests**

Create `tests/test_trainer.py`:

```python
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sr.config import TrainingConfig
from sr.model import FastSRNet
from sr.trainer import train_step, validate


def test_train_step_updates_fp32_model_parameters():
    torch.manual_seed(3)
    model = FastSRNet().float()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    lr = torch.rand(2, 3, 16, 16)
    hr = torch.rand(2, 3, 32, 32)
    before = model.model[0].weight.detach().clone()

    loss = train_step(model, optimizer, nn.L1Loss(), lr, hr, torch.device("cpu"))

    assert loss > 0
    assert model.model[0].weight.dtype == torch.float32
    assert not torch.equal(before, model.model[0].weight)


def test_train_step_rejects_non_finite_loss():
    model = FastSRNet().float()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    lr = torch.full((1, 3, 16, 16), float("nan"))
    hr = torch.rand(1, 3, 32, 32)

    with pytest.raises(FloatingPointError, match="non-finite"):
        train_step(model, optimizer, nn.L1Loss(), lr, hr, torch.device("cpu"))


def test_validate_reports_finite_metrics_for_complete_images():
    torch.manual_seed(5)
    model = FastSRNet().float()
    lr = torch.rand(2, 3, 20, 24)
    with torch.no_grad():
        hr = model(lr).clamp(0.0, 1.0)
    loader = DataLoader(TensorDataset(lr, hr), batch_size=1, shuffle=False)

    metrics = validate(model, loader, torch.device("cpu"), scale=2)

    assert metrics["l1"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["rgb_psnr"] == float("inf")
    assert metrics["y_psnr"] == float("inf")
    assert metrics["ssim"] == pytest.approx(1.0, abs=1e-5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_trainer.py -v
```

Expected: collection fails because `sr.trainer` does not exist.

- [ ] **Step 3: Implement data loaders, FP32 steps, validation, and training orchestration**

Create `sr/trainer.py`:

```python
from __future__ import annotations

import random
import time
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from .checkpoint import export_weights, save_checkpoint
from .config import TrainingConfig
from .metrics import crop_border, psnr, rgb_to_y, ssim


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def make_train_loader(dataset: Dataset, config: TrainingConfig) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def make_validation_loader(dataset: Dataset, config: TrainingConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
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
    lr = lr.to(device=device, dtype=torch.float32, non_blocking=True)
    hr = hr.to(device=device, dtype=torch.float32, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    sr = model(lr)
    if sr.shape != hr.shape:
        raise ValueError(f"model output {tuple(sr.shape)} does not match HR {tuple(hr.shape)}")
    loss = loss_fn(sr, hr)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite training loss: {float(loss.detach())}")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scale: int,
) -> dict[str, float]:
    model.eval()
    totals = {"l1": 0.0, "rgb_psnr": 0.0, "y_psnr": 0.0, "ssim": 0.0}
    image_count = 0
    for lr, hr in loader:
        lr = lr.to(device=device, dtype=torch.float32, non_blocking=True)
        hr = hr.to(device=device, dtype=torch.float32, non_blocking=True)
        sr = model(lr).clamp(0.0, 1.0)
        if sr.shape != hr.shape:
            raise ValueError(f"validation output {tuple(sr.shape)} does not match HR {tuple(hr.shape)}")
        sr = crop_border(sr, scale)
        hr = crop_border(hr, scale)
        sr_y = rgb_to_y(sr)
        hr_y = rgb_to_y(hr)
        batch_size = hr.shape[0]
        totals["l1"] += float(torch.nn.functional.l1_loss(sr, hr)) * batch_size
        totals["rgb_psnr"] += float(psnr(sr, hr)) * batch_size
        totals["y_psnr"] += float(psnr(sr_y, hr_y)) * batch_size
        totals["ssim"] += float(ssim(sr_y, hr_y)) * batch_size
        image_count += batch_size
    if image_count == 0:
        raise ValueError("validation loader is empty")
    return {name: total / image_count for name, total in totals.items()}


def _next_batch(
    iterator: Iterator[tuple[torch.Tensor, torch.Tensor]],
    loader: DataLoader,
) -> tuple[tuple[torch.Tensor, torch.Tensor], Iterator[tuple[torch.Tensor, torch.Tensor]]]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


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
    for global_step in range(start_step + 1, config.max_steps + 1):
        (lr, hr), train_iterator = _next_batch(train_iterator, train_loader)
        loss = train_step(model, optimizer, loss_fn, lr, hr, device)
        scheduler.step()
        running_loss += loss
        running_steps += 1
        interval_images += lr.shape[0]

        if global_step % config.log_interval == 0:
            elapsed = time.perf_counter() - interval_start
            mean_loss = running_loss / running_steps
            throughput = interval_images / elapsed
            learning_rate = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/l1", mean_loss, global_step)
            writer.add_scalar("train/learning_rate", learning_rate, global_step)
            writer.add_scalar("train/images_per_second", throughput, global_step)
            writer.add_scalar("train/interval_seconds", elapsed, global_step)
            print(
                f"step={global_step}/{config.max_steps} "
                f"loss={mean_loss:.6f} lr={learning_rate:.8f} "
                f"images_per_second={throughput:.1f}"
            )
            interval_start = time.perf_counter()
            interval_images = 0
            running_loss = 0.0
            running_steps = 0

        if global_step % config.eval_interval == 0 or global_step == config.max_steps:
            metrics = validate(model, validation_loader, device, config.scale)
            for name, value in metrics.items():
                writer.add_scalar(f"validation/{name}", value, global_step)
            print(
                f"validation step={global_step} l1={metrics['l1']:.6f} "
                f"rgb_psnr={metrics['rgb_psnr']:.4f} "
                f"y_psnr={metrics['y_psnr']:.4f} ssim={metrics['ssim']:.6f}"
            )
            if metrics["y_psnr"] > best_y_psnr:
                best_y_psnr = metrics["y_psnr"]
                save_checkpoint(
                    config.output_dir / "best_y_psnr.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    best_y_psnr=best_y_psnr,
                    config=config.as_dict(),
                )

        if global_step % config.checkpoint_interval == 0 or global_step == config.max_steps:
            checkpoint_arguments = {
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "global_step": global_step,
                "best_y_psnr": best_y_psnr,
                "config": config.as_dict(),
            }
            save_checkpoint(config.output_dir / "latest.pth", **checkpoint_arguments)
            save_checkpoint(
                config.output_dir / f"checkpoint_{global_step:06d}.pth",
                **checkpoint_arguments,
            )
            export_weights(config.output_dir / "model_weights.pth", model)
            writer.flush()
```

- [ ] **Step 4: Run the trainer tests**

Run:

```powershell
python -m pytest tests/test_trainer.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit the training engine**

```powershell
git add sr/trainer.py tests/test_trainer.py
git commit -m "feat: add FP32 step-based training engine"
```

### Task 7: Replace the prototype with a safe CUDA command-line entry point

**Files:**
- Replace: `train_sr.py`
- Create: `tests/test_train_sr.py`

- [ ] **Step 1: Write failing CLI and preflight tests**

Create `tests/test_train_sr.py`:

```python
from pathlib import Path

import pytest
import torch

from sr.config import TrainingConfig
from sr.model import FastSRNet
from train_sr import build_parser, preflight_model


def test_parser_exposes_step_and_loader_overrides():
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

    assert arguments.data_root == Path("D:/datasets/DF2K")
    assert arguments.output_dir == Path("D:/runs/df2k")
    assert arguments.batch_size == 32
    assert arguments.num_workers == 8
    assert arguments.max_steps == 12


def test_preflight_accepts_matching_finite_model_output():
    model = FastSRNet()
    lr = torch.rand(3, 128, 128)
    hr = torch.rand(3, 256, 256)

    preflight_model(model, lr, hr, torch.device("cpu"))


def test_preflight_rejects_mismatched_hr_shape():
    config = TrainingConfig.from_data_root(Path("dataset"))
    model = FastSRNet(config.scale)
    lr = torch.rand(3, 128, 128)
    hr = torch.rand(3, 255, 256)

    with pytest.raises(ValueError, match="does not match"):
        preflight_model(model, lr, hr, torch.device("cpu"))
```

- [ ] **Step 2: Run the tests to verify the old script has no CLI API**

Run:

```powershell
python -m pytest tests/test_train_sr.py -v
```

Expected: collection fails because the existing `train_sr.py` does not export `build_parser` and `preflight_model`.

- [ ] **Step 3: Replace `train_sr.py` with the complete entry point**

Replace the entire contents of `train_sr.py` with:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from sr.checkpoint import load_checkpoint
from sr.config import TrainingConfig
from sr.data import build_training_dataset, build_validation_dataset
from sr.model import FastSRNet
from sr.trainer import (
    make_train_loader,
    make_validation_loader,
    run_training,
    seed_everything,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the FP32 DF2K bicubic RGB x2 SR model")
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/df2k_x2"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=150_000)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--checkpoint-interval", type=int, default=5_000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument("--resume", type=Path)
    return parser


def config_from_arguments(arguments: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig.from_data_root(
        arguments.data_root,
        output_dir=arguments.output_dir,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        max_steps=arguments.max_steps,
        eval_interval=arguments.eval_interval,
        checkpoint_interval=arguments.checkpoint_interval,
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
    lr_batch = lr.unsqueeze(0).to(device=device, dtype=torch.float32)
    hr_batch = hr.unsqueeze(0).to(device=device, dtype=torch.float32)
    sr_batch = model(lr_batch)
    if sr_batch.shape != hr_batch.shape:
        raise ValueError(
            f"model output {tuple(sr_batch.shape)} does not match HR {tuple(hr_batch.shape)}"
        )
    loss = torch.nn.functional.l1_loss(sr_batch, hr_batch)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"preflight produced non-finite loss: {float(loss)}")


def main() -> None:
    arguments = build_parser().parse_args()
    config = config_from_arguments(arguments)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DF2K training but is not available")
    device = torch.device("cuda:0")
    seed_everything(config.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = build_training_dataset(config)
    validation_dataset = build_validation_dataset(config)
    config.save_json(
        config.output_dir / "config.json",
        metadata={
            "device": torch.cuda.get_device_name(device),
            "train_pairs": len(train_dataset),
            "validation_pairs": len(validation_dataset),
        },
    )
    print(
        f"device={torch.cuda.get_device_name(device)} "
        f"train_pairs={len(train_dataset)} validation_pairs={len(validation_dataset)} "
        f"batch_size={config.batch_size} workers={config.num_workers}"
    )

    model = FastSRNet(config.scale).to(device=device, dtype=torch.float32)
    preflight_model(model, *train_dataset[0], device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_steps,
        eta_min=config.minimum_learning_rate,
    )
    start_step = 0
    best_y_psnr = float("-inf")
    if arguments.resume is not None:
        state = load_checkpoint(arguments.resume, model, optimizer, scheduler, device)
        start_step = state.global_step
        best_y_psnr = state.best_y_psnr
        print(f"resumed={arguments.resume} step={start_step} best_y_psnr={best_y_psnr:.4f}")
    if start_step >= config.max_steps:
        raise ValueError(
            f"checkpoint step {start_step} must be smaller than max_steps {config.max_steps}"
        )

    train_loader = make_train_loader(train_dataset, config)
    validation_loader = make_validation_loader(validation_dataset, config)
    with SummaryWriter(log_dir=str(config.output_dir / "tensorboard")) as writer:
        run_training(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=device,
            writer=writer,
            start_step=start_step,
            best_y_psnr=best_y_psnr,
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI tests and verify importing the script does not start workers**

Run:

```powershell
python -m pytest tests/test_train_sr.py -v
python -c "import train_sr; print('import safe')"
```

Expected: `3 passed`, followed by `import safe` exactly once.

- [ ] **Step 5: Commit the CUDA entry point**

```powershell
git add train_sr.py tests/test_train_sr.py
git commit -m "feat: add safe DF2K CUDA training entry point"
```

### Task 8: Document data layout and exclude generated artifacts

**Files:**
- Create: `README.md`
- Modify: `.gitignore`
- Create: `tests/test_repository_contract.py`

- [ ] **Step 1: Write the repository contract test**

Create `tests/test_repository_contract.py`:

```python
from pathlib import Path


def test_generated_training_directories_are_ignored():
    ignored = Path(".gitignore").read_text(encoding="utf-8").splitlines()

    assert "dataset/" in ignored
    assert "outputs/" in ignored
    assert "models/" in ignored


def test_readme_documents_all_six_required_dataset_directories():
    readme = Path("README.md").read_text(encoding="utf-8")
    required = [
        "DIV2K_train_HR",
        "DIV2K_train_LR_bicubic/X2",
        "Flickr2K_HR",
        "Flickr2K_train_LR_bicubic/X2",
        "DIV2K_valid_HR",
        "DIV2K_valid_LR_bicubic/X2",
    ]

    for directory in required:
        assert directory in readme
```

- [ ] **Step 2: Run the repository contract test to verify it fails**

Run:

```powershell
python -m pytest tests/test_repository_contract.py -v
```

Expected: both tests fail because outputs/models are not ignored and `README.md` does not exist.

- [ ] **Step 3: Extend `.gitignore`**

Append these exact lines to `.gitignore`:

```gitignore
outputs/
models/
.pytest_cache/
```

- [ ] **Step 4: Add exact setup and training documentation**

Create `README.md`:

````markdown
# Lightweight RGB x2 Super-Resolution

This project trains a 7,124-parameter RGB x2 PixelShuffle model in FP32 on DF2K. Camera-specific degradation, video capture, INT8 quantization, and FPGA deployment are outside the current training stage.

## Dataset layout

Place the official bicubic x2 pairs under `dataset/`:

```text
dataset/
├── DIV2K_train_HR/                 # 800 PNG files
├── DIV2K_train_LR_bicubic/
│   └── X2/                         # 800 files named like 0001x2.png
├── Flickr2K_HR/                    # 2650 PNG files
├── Flickr2K_train_LR_bicubic/
│   └── X2/                         # 2650 files named like 000001x2.png
├── DIV2K_valid_HR/                 # 100 PNG files
└── DIV2K_valid_LR_bicubic/
    └── X2/                         # 100 files named like 0801x2.png
```

The loader stops before training if a directory, pair, count, or x2 dimension relationship is invalid. Dataset contents are ignored by Git.

## Environment check

Run these commands in the CUDA 12.8-compatible Python 3.11 environment:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m pytest -q
```

The CUDA check must print `True` and identify the NVIDIA GPU.

## Training

Default RTX 4090 run:

```powershell
python train_sr.py --data-root dataset --output-dir outputs/df2k_x2
```

If 24 GB VRAM is not being used effectively, measure batch sizes 64, 96, and 128 and keep the largest value that trains without an out-of-memory error. Batch size changes throughput and gradient statistics; keep the 150,000-step schedule unchanged for the first controlled comparison.

Resume an interrupted run:

```powershell
python train_sr.py --data-root dataset --output-dir outputs/df2k_x2 --resume outputs/df2k_x2/latest.pth
```

Inspect TensorBoard:

```powershell
tensorboard --logdir outputs/df2k_x2/tensorboard
```

Generated files include `latest.pth`, `best_y_psnr.pth`, numbered checkpoints, `model_weights.pth`, `config.json`, and TensorBoard event files.
````

- [ ] **Step 5: Run repository contract tests**

Run:

```powershell
python -m pytest tests/test_repository_contract.py -v
```

Expected: `2 passed`.

- [ ] **Step 6: Commit documentation and ignore rules**

```powershell
git add .gitignore README.md tests/test_repository_contract.py
git commit -m "docs: add DF2K training setup"
```

### Task 9: Perform unit, CUDA, dataset, resume, and artifact verification

**Files:**
- Verify: all files created or modified in Tasks 1-8

- [ ] **Step 1: Run the entire test suite from a clean process**

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass with no failures or errors.

- [ ] **Step 2: Verify formatting hazards and repository state**

Run:

```powershell
git diff --check
git status --short
```

Expected: `git diff --check` prints nothing. `git status --short` prints nothing after the preceding task commits.

- [ ] **Step 3: Verify CUDA and FP32 behavior on the RTX 4090**

Run:

```powershell
python -c "import torch; from sr.model import FastSRNet; d=torch.device('cuda:0'); m=FastSRNet().to(d); x=torch.rand(1,3,128,128,device=d); y=m(x); print(torch.cuda.get_device_name(d)); print(x.dtype, next(m.parameters()).dtype, y.dtype, tuple(y.shape))"
```

Expected: the first line contains `NVIDIA GeForce RTX 4090`; the second line contains `torch.float32 torch.float32 torch.float32 (1, 3, 256, 256)`.

- [ ] **Step 4: Verify the real dataset contract before the long run**

Run:

```powershell
python -c "from pathlib import Path; from sr.config import TrainingConfig; from sr.data import build_training_dataset, build_validation_dataset; c=TrainingConfig.from_data_root(Path('dataset')); t=build_training_dataset(c); v=build_validation_dataset(c); print(len(t), len(v), t[0][0].shape, t[0][1].shape)"
```

Expected: `3450 100 torch.Size([3, 128, 128]) torch.Size([3, 256, 256])`.

- [ ] **Step 5: Run a two-step end-to-end CUDA smoke test**

Run:

```powershell
python train_sr.py --data-root dataset --output-dir outputs/smoke --batch-size 2 --num-workers 0 --max-steps 2 --eval-interval 1 --checkpoint-interval 1 --log-interval 1
```

Expected: two finite training losses, two validation reports, and exit code 0. `outputs/smoke/` contains `latest.pth`, `best_y_psnr.pth`, `checkpoint_000001.pth`, `checkpoint_000002.pth`, `model_weights.pth`, and `config.json`.

- [ ] **Step 6: Verify resume from the first smoke checkpoint**

Run:

```powershell
python train_sr.py --data-root dataset --output-dir outputs/smoke_resume --batch-size 2 --num-workers 0 --max-steps 2 --eval-interval 1 --checkpoint-interval 1 --log-interval 1 --resume outputs/smoke/checkpoint_000001.pth
```

Expected: output reports `step=1`, runs only step 2, validates successfully, and creates `outputs/smoke_resume/latest.pth`.

- [ ] **Step 7: Start the full configured run only after smoke verification**

Run:

```powershell
python train_sr.py --data-root dataset --output-dir outputs/df2k_x2 --batch-size 64 --num-workers 4 --max-steps 150000
```

Expected startup output: RTX 4090 device name, `train_pairs=3450`, `validation_pairs=100`, `batch_size=64`, and `workers=4`. Training then reports one aggregate line per 100 steps.

- [ ] **Step 8: Commit any verification-only correction, if one was required**

If verification required a correction, stage only the corrected source and its regression test, then run:

```powershell
git commit -m "fix: correct DF2K training verification issue"
```

If no correction was required, do not create an empty commit.

## Completion Conditions

- All tests pass.
- The repository has no unintended uncommitted changes.
- Real data validation reports exactly 3,450 training pairs and 100 validation pairs.
- The CUDA smoke run is FP32 and produces finite loss and metrics.
- Resume continues from the stored optimizer step and scheduler state.
- Best, latest, numbered, and weights-only artifacts exist.
- No active training code contains camera degradation, RealSR/DRealSR, video capture, AMP, or INT8 logic.
- The full 150,000-step run starts only after all preceding checks pass.
