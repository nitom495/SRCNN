# Real-Camera ×2 Super-Resolution Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the existing FP32 RGB ×2 training pipeline for camera footage with synthetic camera degradation, reproducible 150,000-step training, EMA, validation, logging, and recoverable checkpoints while preserving the deployed network architecture.

**Architecture:** Split the current single script into focused `sr` modules for configuration, model, degradation, datasets, losses, metrics, EMA/checkpoints, and training operations. `train_sr.py` remains the command-line entry point and coordinates these units. All image degradation and augmentation runs in DataLoader workers on CPU; model forward, loss, and backward run on the RTX 4090.

**Tech Stack:** Python 3.11, PyTorch with CUDA, Torchvision, Pillow, TorchMetrics, TensorBoard, pytest.

---

## File Map

- Create `requirements.txt`: non-PyTorch runtime and test dependencies.
- Create `sr/__init__.py`: package exports.
- Create `sr/config.py`: validated training configuration.
- Create `sr/model.py`: unchanged FastESPCN ×2 inference architecture.
- Create `sr/degradation.py`: synthetic camera degradation.
- Create `sr/datasets.py`: HR-only training data and paired validation data.
- Create `sr/losses.py`: Sobel-gradient loss.
- Create `sr/ema.py`: exponential moving average weights.
- Create `sr/checkpoint.py`: atomic save and complete resume.
- Create `sr/metrics.py`: RGB/Y PSNR and SSIM.
- Create `sr/trainer.py`: one-step training and deterministic validation.
- Modify `train_sr.py`: CLI, DataLoaders, scheduling, logging, validation, and checkpoints.
- Create `tests/`: unit and integration tests for each boundary.
- Create `README.md`: environment, datasets, training, resume, and verification commands.

### Task 1: Configuration and dependency baseline

**Files:**
- Create: `requirements.txt`
- Create: `sr/__init__.py`
- Create: `sr/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing configuration tests**

```python
# tests/test_config.py
from pathlib import Path

import pytest

from sr.config import TrainConfig


def test_train_config_accepts_valid_values(tmp_path: Path):
    config = TrainConfig(
        train_hr_dirs=(tmp_path / "div2k", tmp_path / "flickr2k"),
        div2k_val_lr_dir=tmp_path / "val_lr",
        div2k_val_hr_dir=tmp_path / "val_hr",
        output_dir=tmp_path / "outputs",
    )

    assert config.scale == 2
    assert config.lr_patch_size == 128
    assert config.total_steps == 150_000
    assert config.stage1_steps == 120_000


def test_train_config_rejects_invalid_stage_boundary(tmp_path: Path):
    with pytest.raises(ValueError, match="stage1_steps"):
        TrainConfig(
            train_hr_dirs=(tmp_path / "div2k",),
            div2k_val_lr_dir=tmp_path / "val_lr",
            div2k_val_hr_dir=tmp_path / "val_hr",
            output_dir=tmp_path / "outputs",
            stage1_steps=200,
            total_steps=100,
        )


def test_train_config_rejects_non_x2_scale(tmp_path: Path):
    with pytest.raises(ValueError, match="scale must be 2"):
        TrainConfig(
            train_hr_dirs=(tmp_path / "div2k",),
            div2k_val_lr_dir=tmp_path / "val_lr",
            div2k_val_hr_dir=tmp_path / "val_hr",
            output_dir=tmp_path / "outputs",
            scale=4,
        )
```

- [ ] **Step 2: Run the tests and verify the expected import failure**

Run:

```powershell
python -m pytest tests/test_config.py -v
```

Expected: FAIL because `sr.config` does not exist.

- [ ] **Step 3: Add dependencies and the minimal validated configuration**

```text
# requirements.txt
Pillow>=10.0
tensorboard>=2.16
torchmetrics>=1.4
pytest>=8.0
```

```python
# sr/__init__.py
from .config import TrainConfig

__all__ = ["TrainConfig"]
```

```python
# sr/config.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    train_hr_dirs: tuple[Path, ...]
    div2k_val_lr_dir: Path
    div2k_val_hr_dir: Path
    output_dir: Path
    real_val_lr_dir: Path | None = None
    real_val_hr_dir: Path | None = None
    camera_dir: Path | None = None
    scale: int = 2
    lr_patch_size: int = 128
    batch_size: int = 64
    num_workers: int = 8
    total_steps: int = 150_000
    stage1_steps: int = 120_000
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-6
    edge_weight: float = 0.05
    ema_decay: float = 0.999
    val_interval: int = 2_000
    checkpoint_interval: int = 5_000
    seed: int = 20_260_926

    def __post_init__(self) -> None:
        if not self.train_hr_dirs:
            raise ValueError("train_hr_dirs cannot be empty")
        if self.scale != 2:
            raise ValueError("scale must be 2")
        if self.lr_patch_size <= 0:
            raise ValueError("lr_patch_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 < self.stage1_steps < self.total_steps:
            raise ValueError("stage1_steps must be between 0 and total_steps")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be between 0 and 1")
```

- [ ] **Step 4: Run the configuration tests**

Run:

```powershell
python -m pytest tests/test_config.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit the baseline**

```powershell
git add requirements.txt sr/__init__.py sr/config.py tests/test_config.py
git commit -m "chore: add validated training configuration"
```

### Task 2: Preserve the deployed FastESPCN architecture

**Files:**
- Create: `sr/model.py`
- Create: `tests/test_model.py`
- Modify: `train_sr.py:82-98`

- [ ] **Step 1: Write model shape and architecture tests**

```python
# tests/test_model.py
import torch

from sr.model import FastESPCN


def test_fast_espcn_doubles_rgb_spatial_dimensions():
    model = FastESPCN()
    x = torch.rand(2, 3, 32, 48)

    y = model(x)

    assert y.shape == (2, 3, 64, 96)


def test_fast_espcn_preserves_current_parameter_count():
    model = FastESPCN()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert parameter_count == 7_124
```

- [ ] **Step 2: Run the model tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_model.py -v
```

Expected: FAIL because `sr.model` does not exist.

- [ ] **Step 3: Move the existing network into `sr/model.py` without changing layers**

```python
# sr/model.py
import torch
from torch import nn


class FastESPCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 12, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
```

Update the package exports at the same time:

```python
# sr/__init__.py
from .config import TrainConfig
from .model import FastESPCN

__all__ = ["FastESPCN", "TrainConfig"]
```

Delete the old `NN` class from `train_sr.py`; Task 9 will import `FastESPCN` from `sr.model`.

- [ ] **Step 4: Run the model tests**

Run:

```powershell
python -m pytest tests/test_model.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit the architecture extraction**

```powershell
git add sr/model.py sr/__init__.py tests/test_model.py train_sr.py
git commit -m "refactor: isolate unchanged FastESPCN model"
```

### Task 3: Synthetic camera degradation

**Files:**
- Create: `sr/degradation.py`
- Create: `tests/test_degradation.py`

- [ ] **Step 1: Write degradation shape, range, clean-path, and repeatability tests**

```python
# tests/test_degradation.py
import torch

from sr.degradation import CameraDegradation, DegradationConfig


def test_degradation_produces_x2_rgb_pair_in_range():
    degradation = CameraDegradation(DegradationConfig(clean_probability=0.0))
    hr = torch.linspace(0, 1, 3 * 256 * 256).reshape(3, 256, 256)

    lr = degradation(hr, generator=torch.Generator().manual_seed(7))

    assert lr.shape == (3, 128, 128)
    assert 0.0 <= float(lr.min()) <= float(lr.max()) <= 1.0


def test_clean_path_matches_bicubic_downsample():
    degradation = CameraDegradation(DegradationConfig(clean_probability=1.0))
    hr = torch.rand(3, 256, 256)
    generator = torch.Generator().manual_seed(11)

    actual = degradation(hr, generator=generator)
    expected = torch.nn.functional.interpolate(
        hr.unsqueeze(0),
        size=(128, 128),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).squeeze(0).clamp(0, 1)

    torch.testing.assert_close(actual, expected)


def test_degradation_is_repeatable_for_a_fixed_generator_seed():
    degradation = CameraDegradation(DegradationConfig(clean_probability=0.0))
    hr = torch.rand(3, 256, 256)

    first = degradation(hr, generator=torch.Generator().manual_seed(19))
    second = degradation(hr, generator=torch.Generator().manual_seed(19))

    torch.testing.assert_close(first, second)
```

- [ ] **Step 2: Run the degradation tests and verify the import failure**

Run:

```powershell
python -m pytest tests/test_degradation.py -v
```

Expected: FAIL because `sr.degradation` does not exist.

- [ ] **Step 3: Implement the specified degradation pipeline**

```python
# sr/degradation.py
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg


@dataclass(frozen=True)
class DegradationConfig:
    scale: int = 2
    clean_probability: float = 0.2
    gaussian_noise_probability: float = 0.5
    poisson_noise_probability: float = 0.2
    jpeg_probability: float = 0.5
    gaussian_sigma_max: float = 10.0
    poisson_scale_min: float = 0.05
    poisson_scale_max: float = 1.5
    jpeg_quality_min: int = 60
    jpeg_quality_max: int = 100


class CameraDegradation:
    def __init__(self, config: DegradationConfig | None = None) -> None:
        self.config = config or DegradationConfig()
        if self.config.scale != 2:
            raise ValueError("CameraDegradation supports scale=2 only")

    @staticmethod
    def _rand(generator: torch.Generator | None) -> float:
        return float(torch.rand((), generator=generator))

    @staticmethod
    def _uniform(
        low: float,
        high: float,
        generator: torch.Generator | None,
    ) -> float:
        return low + (high - low) * CameraDegradation._rand(generator)

    def _downsample(self, image: torch.Tensor, mode: str) -> torch.Tensor:
        size = (image.shape[-2] // 2, image.shape[-1] // 2)
        if mode == "area":
            return F.interpolate(image.unsqueeze(0), size=size, mode=mode).squeeze(0)
        return F.interpolate(
            image.unsqueeze(0),
            size=size,
            mode=mode,
            align_corners=False,
            antialias=True,
        ).squeeze(0)

    def _jpeg(self, image: torch.Tensor, quality: int) -> torch.Tensor:
        uint8_image = image.mul(255).round().to(torch.uint8).cpu()
        encoded = encode_jpeg(uint8_image, quality=quality)
        decoded = decode_jpeg(encoded, mode=ImageReadMode.RGB)
        return decoded.float().div(255)

    def __call__(
        self,
        hr: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if hr.device.type != "cpu":
            raise ValueError("degradation must run on CPU")
        if hr.ndim != 3 or hr.shape[0] != 3:
            raise ValueError("expected a CHW RGB tensor")
        if hr.shape[-2] % 2 or hr.shape[-1] % 2:
            raise ValueError("HR dimensions must be divisible by 2")

        if self._rand(generator) < self.config.clean_probability:
            return self._downsample(hr, "bicubic").clamp(0, 1)

        kernel_sizes = (7, 9, 11, 13)
        kernel_index = int(
            torch.randint(len(kernel_sizes), (), generator=generator)
        )
        kernel_size = kernel_sizes[kernel_index]
        sigma_x = self._uniform(0.2, 1.5, generator)
        sigma_y = sigma_x
        if self._rand(generator) < 0.5:
            sigma_y = self._uniform(0.2, 1.5, generator)
        degraded = TF.gaussian_blur(
            hr,
            kernel_size=[kernel_size, kernel_size],
            sigma=[sigma_x, sigma_y],
        )

        modes = ("area", "bilinear", "bicubic")
        mode_index = int(torch.randint(len(modes), (), generator=generator))
        degraded = self._downsample(degraded, modes[mode_index])

        noise_choice = self._rand(generator)
        if noise_choice < self.config.gaussian_noise_probability:
            sigma = self._uniform(0.0, self.config.gaussian_sigma_max, generator) / 255.0
            noise = torch.randn(
                degraded.shape,
                generator=generator,
                dtype=degraded.dtype,
            )
            degraded = degraded + noise * sigma
        elif noise_choice < (
            self.config.gaussian_noise_probability
            + self.config.poisson_noise_probability
        ):
            scale = self._uniform(
                self.config.poisson_scale_min,
                self.config.poisson_scale_max,
                generator,
            )
            peak = 255.0 * scale
            degraded = torch.poisson(
                degraded.clamp(0, 1) * peak,
                generator=generator,
            ) / peak

        degraded = degraded.clamp(0, 1)
        if self._rand(generator) < self.config.jpeg_probability:
            quality = int(
                torch.randint(
                    self.config.jpeg_quality_min,
                    self.config.jpeg_quality_max + 1,
                    (),
                    generator=generator,
                )
            )
            degraded = self._jpeg(degraded, quality)

        return degraded.clamp(0, 1)
```

- [ ] **Step 4: Run the degradation tests**

Run:

```powershell
python -m pytest tests/test_degradation.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit camera degradation**

```powershell
git add sr/degradation.py tests/test_degradation.py
git commit -m "feat: add synthetic camera degradation"
```

### Task 4: HR-only training and paired validation datasets

**Files:**
- Create: `sr/datasets.py`
- Create: `tests/test_datasets.py`

- [ ] **Step 1: Write dataset discovery, shape, pairing, and error tests**

```python
# tests/test_datasets.py
from pathlib import Path

import pytest
import torch
from PIL import Image

from sr.datasets import HRTrainingDataset, PairedValidationDataset
from sr.degradation import CameraDegradation, DegradationConfig


def save_rgb(path: Path, width: int, height: int, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(value, value, value)).save(path)


def test_hr_training_dataset_returns_x2_pair(tmp_path: Path):
    save_rgb(tmp_path / "hr" / "0001.png", 320, 320, 128)
    dataset = HRTrainingDataset(
        hr_dirs=(tmp_path / "hr",),
        lr_patch_size=128,
        degradation=CameraDegradation(DegradationConfig(clean_probability=1.0)),
    )

    lr, hr, path = dataset[0]

    assert lr.shape == (3, 128, 128)
    assert hr.shape == (3, 256, 256)
    assert path.endswith("0001.png")


def test_hr_training_dataset_rejects_empty_directories(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(RuntimeError, match="No HR images"):
        HRTrainingDataset(
            hr_dirs=(empty,),
            lr_patch_size=128,
            degradation=CameraDegradation(),
        )


def test_paired_validation_dataset_matches_div2k_suffix(tmp_path: Path):
    save_rgb(tmp_path / "lr" / "0801x2.png", 128, 128, 64)
    save_rgb(tmp_path / "hr" / "0801.png", 256, 256, 64)
    dataset = PairedValidationDataset(
        lr_dir=tmp_path / "lr",
        hr_dir=tmp_path / "hr",
        lr_suffix="x2",
    )

    lr, hr, key = dataset[0]

    assert lr.shape == (3, 128, 128)
    assert hr.shape == (3, 256, 256)
    assert key == "0801"


def test_paired_validation_dataset_rejects_missing_hr(tmp_path: Path):
    save_rgb(tmp_path / "lr" / "0801x2.png", 128, 128, 64)
    (tmp_path / "hr").mkdir()

    with pytest.raises(FileNotFoundError, match="Missing HR pair"):
        PairedValidationDataset(
            lr_dir=tmp_path / "lr",
            hr_dir=tmp_path / "hr",
            lr_suffix="x2",
        )
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run:

```powershell
python -m pytest tests/test_datasets.py -v
```

Expected: FAIL because `sr.datasets` does not exist.

- [ ] **Step 3: Implement both datasets**

```python
# sr/datasets.py
from collections.abc import Sequence
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop

from .degradation import CameraDegradation


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return TF.pil_to_tensor(image.convert("RGB")).float().div(255)


class HRTrainingDataset(Dataset):
    def __init__(
        self,
        hr_dirs: Sequence[Path],
        lr_patch_size: int,
        degradation: CameraDegradation,
    ) -> None:
        self.hr_patch_size = lr_patch_size * 2
        self.degradation = degradation
        self.paths = sorted(
            path
            for directory in hr_dirs
            for path in Path(directory).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise RuntimeError("No HR images found")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        path = self.paths[index]
        hr = load_rgb_tensor(path)
        if hr.shape[-2] < self.hr_patch_size or hr.shape[-1] < self.hr_patch_size:
            raise ValueError(f"HR image is too small: {path}")
        top, left, height, width = RandomCrop.get_params(
            hr,
            output_size=(self.hr_patch_size, self.hr_patch_size),
        )
        hr = TF.crop(hr, top, left, height, width)
        if float(torch.rand(())) < 0.5:
            hr = torch.flip(hr, dims=[2])
        lr = self.degradation(hr)
        return lr, hr, str(path)


class PairedValidationDataset(Dataset):
    def __init__(
        self,
        lr_dir: Path,
        hr_dir: Path,
        lr_suffix: str = "x2",
    ) -> None:
        self.samples: list[tuple[Path, Path, str]] = []
        for lr_path in sorted(Path(lr_dir).rglob("*.png")):
            key = lr_path.stem.removesuffix(lr_suffix)
            hr_path = Path(hr_dir) / f"{key}.png"
            if not hr_path.exists():
                raise FileNotFoundError(f"Missing HR pair: {hr_path}")
            self.samples.append((lr_path, hr_path, key))
        if not self.samples:
            raise RuntimeError("No validation pairs found")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        lr_path, hr_path, key = self.samples[index]
        lr = load_rgb_tensor(lr_path)
        hr = load_rgb_tensor(hr_path)
        if hr.shape[-2:] != (lr.shape[-2] * 2, lr.shape[-1] * 2):
            raise ValueError(f"Invalid ×2 pair: {key}")
        return lr, hr, key
```

- [ ] **Step 4: Run the dataset tests**

Run:

```powershell
python -m pytest tests/test_datasets.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit datasets**

```powershell
git add sr/datasets.py tests/test_datasets.py
git commit -m "feat: add HR training and paired validation datasets"
```

### Task 5: Edge-aware stage-two loss

**Files:**
- Create: `sr/losses.py`
- Create: `tests/test_losses.py`

- [ ] **Step 1: Write Sobel loss tests**

```python
# tests/test_losses.py
import torch

from sr.losses import SobelGradientLoss


def test_sobel_gradient_loss_is_zero_for_identical_images():
    loss_fn = SobelGradientLoss()
    image = torch.rand(2, 3, 32, 32)

    loss = loss_fn(image, image)

    assert float(loss) == 0.0


def test_sobel_gradient_loss_detects_edge_difference():
    loss_fn = SobelGradientLoss()
    target = torch.zeros(1, 3, 32, 32)
    prediction = target.clone()
    prediction[:, :, :, 16:] = 1.0

    loss = loss_fn(prediction, target)

    assert float(loss) > 0.0
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run:

```powershell
python -m pytest tests/test_losses.py -v
```

Expected: FAIL because `sr.losses` does not exist.

- [ ] **Step 3: Implement channel-wise Sobel gradient L1 loss**

```python
# sr/losses.py
import torch
import torch.nn.functional as F
from torch import nn


class SobelGradientLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        )
        kernel_y = kernel_x.t()
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3))

    def _gradient(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        channels = image.shape[1]
        kernel_x = self.kernel_x.expand(channels, 1, 3, 3)
        kernel_y = self.kernel_y.expand(channels, 1, 3, 3)
        gradient_x = F.conv2d(image, kernel_x, padding=1, groups=channels)
        gradient_y = F.conv2d(image, kernel_y, padding=1, groups=channels)
        return gradient_x, gradient_y

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_x, prediction_y = self._gradient(prediction)
        target_x, target_y = self._gradient(target)
        return F.l1_loss(prediction_x, target_x) + F.l1_loss(
            prediction_y,
            target_y,
        )
```

- [ ] **Step 4: Run the loss tests**

Run:

```powershell
python -m pytest tests/test_losses.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit the loss**

```powershell
git add sr/losses.py tests/test_losses.py
git commit -m "feat: add Sobel edge loss"
```

### Task 6: EMA and atomic resumable checkpoints

**Files:**
- Create: `sr/ema.py`
- Create: `sr/checkpoint.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Write EMA update and checkpoint round-trip tests**

```python
# tests/test_state.py
from pathlib import Path

import torch

from sr.checkpoint import load_checkpoint, save_checkpoint
from sr.ema import ModelEMA
from sr.model import FastESPCN


def test_ema_moves_toward_updated_model():
    model = FastESPCN()
    ema = ModelEMA(model, decay=0.5)
    before = next(ema.model.parameters()).detach().clone()
    with torch.no_grad():
        next(model.parameters()).add_(2.0)

    ema.update(model)

    after = next(ema.model.parameters()).detach()
    torch.testing.assert_close(after, before + 1.0)


def test_checkpoint_restores_training_state(tmp_path: Path):
    model = FastESPCN()
    ema = ModelEMA(model, decay=0.999)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=1e-6,
    )
    path = tmp_path / "latest.pth"
    save_checkpoint(
        path=path,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        step=37,
        best_scores={"div2k": 31.2, "real": 28.4},
        config={"scale": 2},
    )

    restored_model = FastESPCN()
    restored_ema = ModelEMA(restored_model, decay=0.999)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=2e-4)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer,
        T_max=100,
        eta_min=1e-6,
    )
    state = load_checkpoint(
        path,
        restored_model,
        restored_ema,
        restored_optimizer,
        restored_scheduler,
        map_location="cpu",
    )

    assert state["step"] == 37
    assert state["best_scores"] == {"div2k": 31.2, "real": 28.4}
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(expected, actual)
```

- [ ] **Step 2: Run state tests and verify the import failure**

Run:

```powershell
python -m pytest tests/test_state.py -v
```

Expected: FAIL because `sr.ema` and `sr.checkpoint` do not exist.

- [ ] **Step 3: Implement EMA**

```python
# sr/ema.py
from copy import deepcopy

import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.model = deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        for name, value in self.model.state_dict().items():
            source_value = source[name].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
            else:
                value.copy_(source_value)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict)
```

- [ ] **Step 4: Implement atomic checkpoint save and complete resume**

```python
# sr/checkpoint.py
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .ema import ModelEMA


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    step: int,
    best_scores: dict[str, float],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "best_scores": best_scores,
            "config": config,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    map_location: str | torch.device,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    random.setstate(checkpoint["python_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return checkpoint
```

- [ ] **Step 5: Run state tests**

Run:

```powershell
python -m pytest tests/test_state.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit EMA and checkpoints**

```powershell
git add sr/ema.py sr/checkpoint.py tests/test_state.py
git commit -m "feat: add EMA and resumable checkpoints"
```

### Task 7: RGB/Y validation metrics

**Files:**
- Create: `sr/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write metric tests**

```python
# tests/test_metrics.py
import torch

from sr.metrics import evaluate_pair, rgb_to_y


def test_rgb_to_y_returns_single_channel():
    rgb = torch.rand(2, 3, 16, 16)

    y = rgb_to_y(rgb)

    assert y.shape == (2, 1, 16, 16)


def test_identical_images_have_near_perfect_metrics():
    image = torch.rand(1, 3, 32, 32)

    metrics = evaluate_pair(image, image, border=2)

    assert metrics["rgb_psnr"] > 100.0
    assert metrics["y_psnr"] > 100.0
    assert metrics["rgb_ssim"] > 0.999
    assert metrics["y_ssim"] > 0.999


def test_metrics_drop_for_different_images():
    prediction = torch.zeros(1, 3, 32, 32)
    target = torch.ones(1, 3, 32, 32)

    metrics = evaluate_pair(prediction, target, border=2)

    assert metrics["rgb_psnr"] < 1.0
    assert metrics["rgb_ssim"] < 0.1
```

- [ ] **Step 2: Run metric tests and verify the import failure**

Run:

```powershell
python -m pytest tests/test_metrics.py -v
```

Expected: FAIL because `sr.metrics` does not exist.

- [ ] **Step 3: Implement deterministic RGB/Y PSNR and SSIM**

```python
# sr/metrics.py
import torch
from torchmetrics.functional.image import structural_similarity_index_measure


def rgb_to_y(rgb: torch.Tensor) -> torch.Tensor:
    coefficients = rgb.new_tensor([0.256788, 0.504129, 0.097906]).view(1, 3, 1, 1)
    return (rgb * coefficients).sum(dim=1, keepdim=True) + 16.0 / 255.0


def _crop_border(image: torch.Tensor, border: int) -> torch.Tensor:
    if border == 0:
        return image
    return image[..., border:-border, border:-border]


def _psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((prediction - target) ** 2).clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def evaluate_pair(
    prediction: torch.Tensor,
    target: torch.Tensor,
    border: int = 2,
) -> dict[str, float]:
    prediction = _crop_border(prediction.clamp(0, 1), border)
    target = _crop_border(target.clamp(0, 1), border)
    prediction_y = rgb_to_y(prediction)
    target_y = rgb_to_y(target)
    return {
        "rgb_psnr": _psnr(prediction, target),
        "y_psnr": _psnr(prediction_y, target_y),
        "rgb_ssim": float(
            structural_similarity_index_measure(prediction, target, data_range=1.0)
        ),
        "y_ssim": float(
            structural_similarity_index_measure(
                prediction_y,
                target_y,
                data_range=1.0,
            )
        ),
    }
```

- [ ] **Step 4: Run metric tests**

Run:

```powershell
python -m pytest tests/test_metrics.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit metrics**

```powershell
git add sr/metrics.py tests/test_metrics.py
git commit -m "feat: add RGB and luminance SR metrics"
```

### Task 8: Tested training and validation operations

**Files:**
- Create: `sr/trainer.py`
- Create: `tests/test_trainer.py`

- [ ] **Step 1: Write one-step, stage-two, non-finite, and validation tests**

```python
# tests/test_trainer.py
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sr.ema import ModelEMA
from sr.losses import SobelGradientLoss
from sr.model import FastESPCN
from sr.trainer import train_step, validate


def make_batch() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    lr = torch.rand(2, 3, 16, 16)
    hr = torch.rand(2, 3, 32, 32)
    return lr, hr, ["a.png", "b.png"]


def test_train_step_updates_model_and_ema():
    model = FastESPCN()
    ema = ModelEMA(model, decay=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    before = next(model.parameters()).detach().clone()

    metrics = train_step(
        model=model,
        ema=ema,
        optimizer=optimizer,
        batch=make_batch(),
        device=torch.device("cpu"),
        use_edge_loss=False,
        edge_weight=0.05,
        edge_loss_fn=SobelGradientLoss(),
    )

    assert metrics["total_loss"] > 0.0
    assert metrics["edge_loss"] == 0.0
    assert not torch.equal(before, next(model.parameters()).detach())


def test_stage_two_reports_edge_loss():
    model = FastESPCN()
    ema = ModelEMA(model, decay=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    metrics = train_step(
        model=model,
        ema=ema,
        optimizer=optimizer,
        batch=make_batch(),
        device=torch.device("cpu"),
        use_edge_loss=True,
        edge_weight=0.05,
        edge_loss_fn=SobelGradientLoss(),
    )

    assert metrics["edge_loss"] > 0.0
    assert metrics["total_loss"] > metrics["l1_loss"]


def test_train_step_rejects_non_finite_loss():
    model = FastESPCN()
    ema = ModelEMA(model, decay=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    lr, hr, paths = make_batch()
    hr.fill_(float("nan"))

    with pytest.raises(FloatingPointError, match="Non-finite loss"):
        train_step(
            model=model,
            ema=ema,
            optimizer=optimizer,
            batch=(lr, hr, paths),
            device=torch.device("cpu"),
            use_edge_loss=False,
            edge_weight=0.05,
            edge_loss_fn=SobelGradientLoss(),
        )


def test_fixed_batch_can_be_overfit():
    torch.manual_seed(5)
    model = FastESPCN()
    ema = ModelEMA(model, decay=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    lr = torch.rand(1, 3, 8, 8)
    hr = F.interpolate(
        lr,
        scale_factor=2,
        mode="bicubic",
        align_corners=False,
    ).clamp(0, 1)
    batch = (lr, hr, ["fixed.png"])
    with torch.no_grad():
        initial_loss = F.l1_loss(model(lr), hr).item()

    for _ in range(300):
        train_step(
            model=model,
            ema=ema,
            optimizer=optimizer,
            batch=batch,
            device=torch.device("cpu"),
            use_edge_loss=False,
            edge_weight=0.05,
            edge_loss_fn=SobelGradientLoss(),
        )

    with torch.no_grad():
        final_loss = F.l1_loss(model(lr), hr).item()
    assert final_loss < initial_loss * 0.25


def test_validate_returns_averaged_metrics():
    model = FastESPCN()
    lr = torch.rand(2, 3, 16, 16)
    hr = torch.rand(2, 3, 32, 32)
    dataset = TensorDataset(lr, hr, torch.arange(2))
    loader = DataLoader(dataset, batch_size=1)

    metrics = validate(model, loader, torch.device("cpu"))

    assert set(metrics) == {"rgb_psnr", "y_psnr", "rgb_ssim", "y_ssim"}
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
```

- [ ] **Step 2: Run trainer tests and verify the import failure**

Run:

```powershell
python -m pytest tests/test_trainer.py -v
```

Expected: FAIL because `sr.trainer` does not exist.

- [ ] **Step 3: Implement one training step and validation**

```python
# sr/trainer.py
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

from .ema import ModelEMA
from .losses import SobelGradientLoss
from .metrics import evaluate_pair


def train_step(
    model: nn.Module,
    ema: ModelEMA,
    optimizer: Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor, object],
    device: torch.device,
    use_edge_loss: bool,
    edge_weight: float,
    edge_loss_fn: SobelGradientLoss,
) -> dict[str, float]:
    lr, hr, _ = batch
    lr = lr.to(device, non_blocking=True)
    hr = hr.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(lr)
    l1_loss = F.l1_loss(prediction, hr)
    edge_loss = edge_loss_fn(prediction, hr) if use_edge_loss else l1_loss.new_zeros(())
    total_loss = l1_loss + edge_weight * edge_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError("Non-finite loss detected")
    total_loss.backward()
    optimizer.step()
    ema.update(model)
    return {
        "total_loss": float(total_loss.detach()),
        "l1_loss": float(l1_loss.detach()),
        "edge_loss": float(edge_loss.detach()),
    }


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {"rgb_psnr": 0.0, "y_psnr": 0.0, "rgb_ssim": 0.0, "y_ssim": 0.0}
    count = 0
    for lr, hr, _ in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        prediction = model(lr).clamp(0, 1)
        metrics = evaluate_pair(prediction, hr, border=2)
        batch_size = lr.shape[0]
        for name, value in metrics.items():
            totals[name] += value * batch_size
        count += batch_size
    if count == 0:
        raise RuntimeError("Validation loader is empty")
    return {name: value / count for name, value in totals.items()}
```

- [ ] **Step 4: Run trainer tests**

Run:

```powershell
python -m pytest tests/test_trainer.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit training operations**

```powershell
git add sr/trainer.py tests/test_trainer.py
git commit -m "feat: add tested training and validation steps"
```

### Task 9: Replace the entry point with step-based FP32 training

**Files:**
- Modify: `train_sr.py:1-174`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write CLI parsing and help tests**

```python
# tests/test_cli.py
import subprocess
import sys

from train_sr import parse_args


def test_parse_args_builds_required_paths():
    args = parse_args(
        [
            "--train-hr",
            "D:/data/DIV2K_train_HR",
            "--train-hr",
            "D:/data/Flickr2K_HR",
            "--div2k-val-lr",
            "D:/data/DIV2K_valid_LR_bicubic/X2",
            "--div2k-val-hr",
            "D:/data/DIV2K_valid_HR",
            "--output-dir",
            "D:/runs/srcnn",
        ]
    )

    assert len(args.train_hr) == 2
    assert args.batch_size == 64
    assert args.total_steps == 150_000


def test_cli_help_exits_successfully():
    result = subprocess.run(
        [sys.executable, "train_sr.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--train-hr" in result.stdout
    assert "--resume" in result.stdout
```

- [ ] **Step 2: Run CLI tests and verify failure against the old script**

Run:

```powershell
python -m pytest tests/test_cli.py -v
```

Expected: FAIL because the old `train_sr.py` has no `parse_args` function and starts training when imported.

- [ ] **Step 3: Replace `train_sr.py` with the new orchestration entry point**

```python
# train_sr.py
import argparse
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sr.checkpoint import load_checkpoint, save_checkpoint
from sr.config import TrainConfig
from sr.datasets import HRTrainingDataset, PairedValidationDataset
from sr.degradation import CameraDegradation
from sr.ema import ModelEMA
from sr.losses import SobelGradientLoss
from sr.model import FastESPCN
from sr.trainer import train_step, validate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FP32 RGB ×2 super-resolution")
    parser.add_argument("--train-hr", action="append", required=True, type=Path)
    parser.add_argument("--div2k-val-lr", required=True, type=Path)
    parser.add_argument("--div2k-val-hr", required=True, type=Path)
    parser.add_argument("--real-val-lr", type=Path)
    parser.add_argument("--real-val-hr", type=Path)
    parser.add_argument("--camera-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=150_000)
    parser.add_argument("--stage1-steps", type=int, default=120_000)
    parser.add_argument("--val-interval", type=int, default=2_000)
    parser.add_argument("--checkpoint-interval", type=int, default=5_000)
    return parser.parse_args(argv)


def make_config(args: argparse.Namespace) -> TrainConfig:
    if (args.real_val_lr is None) != (args.real_val_hr is None):
        raise ValueError("--real-val-lr and --real-val-hr must be provided together")
    return TrainConfig(
        train_hr_dirs=tuple(args.train_hr),
        div2k_val_lr_dir=args.div2k_val_lr,
        div2k_val_hr_dir=args.div2k_val_hr,
        real_val_lr_dir=args.real_val_lr,
        real_val_hr_dir=args.real_val_hr,
        camera_dir=args.camera_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        total_steps=args.total_steps,
        stage1_steps=args.stage1_steps,
        val_interval=args.val_interval,
        checkpoint_interval=args.checkpoint_interval,
    )


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, seed: int):
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
        "generator": torch.Generator().manual_seed(seed),
    }
    if workers > 0:
        options["prefetch_factor"] = 2
    return DataLoader(**options)


def serializable_config(config: TrainConfig) -> dict:
    values = asdict(config)
    for name, value in values.items():
        if isinstance(value, Path):
            values[name] = str(value)
        elif isinstance(value, tuple) and value and isinstance(value[0], Path):
            values[name] = tuple(str(path) for path in value)
    return values


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = make_config(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda:0")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = HRTrainingDataset(
        hr_dirs=config.train_hr_dirs,
        lr_patch_size=config.lr_patch_size,
        degradation=CameraDegradation(),
    )
    div2k_validation = PairedValidationDataset(
        config.div2k_val_lr_dir,
        config.div2k_val_hr_dir,
        lr_suffix="x2",
    )
    train_loader = make_loader(
        train_dataset,
        config.batch_size,
        config.num_workers,
        True,
        config.seed,
    )
    div2k_loader = make_loader(div2k_validation, 1, 2, False, config.seed)
    real_loader = None
    if config.real_val_lr_dir is not None and config.real_val_hr_dir is not None:
        real_validation = PairedValidationDataset(
            config.real_val_lr_dir,
            config.real_val_hr_dir,
            lr_suffix="x2",
        )
        real_loader = make_loader(real_validation, 1, 2, False, config.seed)

    model = FastESPCN().to(device)
    ema = ModelEMA(model, config.ema_decay)
    edge_loss_fn = SobelGradientLoss().to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.total_steps,
        eta_min=config.min_learning_rate,
    )
    step = 0
    best_scores = {"div2k": float("-inf"), "real": float("-inf")}
    if args.resume is not None:
        state = load_checkpoint(
            args.resume,
            model,
            ema,
            optimizer,
            scheduler,
            map_location=device,
        )
        step = int(state["step"])
        best_scores = dict(state["best_scores"])

    writer = SummaryWriter(config.output_dir / "tensorboard")
    config_dict = serializable_config(config)
    writer.add_text("config", str(config_dict), global_step=step)
    train_iterator = iter(train_loader)
    started_at = time.perf_counter()

    try:
        while step < config.total_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            step += 1
            metrics = train_step(
                model=model,
                ema=ema,
                optimizer=optimizer,
                batch=batch,
                device=device,
                use_edge_loss=step > config.stage1_steps,
                edge_weight=config.edge_weight,
                edge_loss_fn=edge_loss_fn,
            )
            scheduler.step()
            for name, value in metrics.items():
                writer.add_scalar(f"train/{name}", value, step)
            writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], step)

            if step % 100 == 0:
                elapsed = time.perf_counter() - started_at
                samples_per_second = step * config.batch_size / elapsed
                print(
                    f"step={step}/{config.total_steps} "
                    f"loss={metrics['total_loss']:.6f} "
                    f"samples/s={samples_per_second:.1f}"
                )

            if step % config.val_interval == 0:
                div2k_metrics = validate(ema.model, div2k_loader, device)
                for name, value in div2k_metrics.items():
                    writer.add_scalar(f"val_div2k/{name}", value, step)
                if div2k_metrics["rgb_psnr"] > best_scores["div2k"]:
                    best_scores["div2k"] = div2k_metrics["rgb_psnr"]
                    torch.save(ema.state_dict(), config.output_dir / "best_div2k.pth")
                if real_loader is not None:
                    real_metrics = validate(ema.model, real_loader, device)
                    for name, value in real_metrics.items():
                        writer.add_scalar(f"val_real/{name}", value, step)
                    if real_metrics["rgb_psnr"] > best_scores["real"]:
                        best_scores["real"] = real_metrics["rgb_psnr"]
                        torch.save(ema.state_dict(), config.output_dir / "best_real.pth")

            if step % config.checkpoint_interval == 0:
                save_checkpoint(
                    config.output_dir / "latest.pth",
                    model,
                    ema,
                    optimizer,
                    scheduler,
                    step,
                    best_scores,
                    config_dict,
                )
    except FloatingPointError:
        save_checkpoint(
            config.output_dir / "non_finite_loss.pth",
            model,
            ema,
            optimizer,
            scheduler,
            step,
            best_scores,
            config_dict,
        )
        raise
    finally:
        writer.close()

    save_checkpoint(
        config.output_dir / "latest.pth",
        model,
        ema,
        optimizer,
        scheduler,
        step,
        best_scores,
        config_dict,
    )
    torch.save(ema.state_dict(), config.output_dir / "model.pth")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI tests**

Run:

```powershell
python -m pytest tests/test_cli.py -v
```

Expected: 2 passed and importing `train_sr` does not start training.

- [ ] **Step 5: Run the full unit suite**

Run:

```powershell
python -m pytest tests -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the entry point**

```powershell
git add train_sr.py tests/test_cli.py
git commit -m "feat: add step-based FP32 training entry point"
```

### Task 10: Camera visual exports and documentation

**Files:**
- Create: `sr/visual.py`
- Create: `tests/test_visual.py`
- Create: `README.md`
- Modify: `train_sr.py`

- [ ] **Step 1: Write a camera export test**

```python
# tests/test_visual.py
from pathlib import Path

from PIL import Image

from sr.model import FastESPCN
from sr.visual import export_camera_samples


def test_export_camera_samples_writes_x2_images(tmp_path: Path):
    input_dir = tmp_path / "camera"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    Image.new("RGB", (24, 16), color=(80, 120, 160)).save(input_dir / "frame.png")

    written = export_camera_samples(
        model=FastESPCN(),
        input_dir=input_dir,
        output_dir=output_dir,
        device="cpu",
    )

    assert written == [output_dir / "frame_x2.png"]
    with Image.open(written[0]) as image:
        assert image.size == (48, 32)
```

- [ ] **Step 2: Run the visual test and verify the import failure**

Run:

```powershell
python -m pytest tests/test_visual.py -v
```

Expected: FAIL because `sr.visual` does not exist.

- [ ] **Step 3: Implement deterministic camera sample export**

```python
# sr/visual.py
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import nn

from .datasets import IMAGE_SUFFIXES


@torch.inference_mode()
def export_camera_samples(
    model: nn.Module,
    input_dir: Path,
    output_dir: Path,
    device: str | torch.device,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    written: list[Path] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        with Image.open(path) as image:
            tensor = TF.pil_to_tensor(image.convert("RGB")).float().div(255)
        prediction = model(tensor.unsqueeze(0).to(device)).clamp(0, 1)[0].cpu()
        output_path = output_dir / f"{path.stem}_x2.png"
        TF.to_pil_image(prediction).save(output_path)
        written.append(output_path)
    return written
```

- [ ] **Step 4: Run the visual test**

Run:

```powershell
python -m pytest tests/test_visual.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Call visual export after each validation interval**

Add this import to `train_sr.py`:

```python
from sr.visual import export_camera_samples
```

At the end of the `if step % config.val_interval == 0:` block, add:

```python
                if config.camera_dir is not None:
                    export_camera_samples(
                        ema.model,
                        config.camera_dir,
                        config.output_dir / "visual" / f"step_{step:06d}",
                        device,
                    )
```

- [ ] **Step 6: Document installation, data layout, training, and resume**

````markdown
# SRCNN Camera Super-Resolution

Lightweight RGB ×2 super-resolution for 1280×720 camera input and future FPGA deployment.

## Environment

Install the CUDA 12.8 PyTorch and Torchvision wheels, then install project dependencies:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## Data

```text
dataset/
├── DIV2K_train_HR/
├── Flickr2K_HR/
├── DIV2K_valid_HR/
├── DIV2K_valid_LR_bicubic/
│   └── X2/
├── RealSR_valid_HR/
│   └── 0001.png
├── RealSR_valid_LR/
│   └── 0001x2.png
└── camera_samples/
```

`dataset/` is ignored by Git. Training requires only HR images; LR training patches are degraded dynamically.

RealSR validation data must be pre-aligned and arranged so that an LR file such as `0001x2.png` has the corresponding HR file `0001.png`.

## Tests

```powershell
python -m pytest tests -v
```

## Training on RTX 4090

```powershell
python train_sr.py `
  --train-hr dataset/DIV2K_train_HR `
  --train-hr dataset/Flickr2K_HR `
  --div2k-val-lr dataset/DIV2K_valid_LR_bicubic/X2 `
  --div2k-val-hr dataset/DIV2K_valid_HR `
  --real-val-lr dataset/RealSR_valid_LR `
  --real-val-hr dataset/RealSR_valid_HR `
  --camera-dir dataset/camera_samples `
  --output-dir runs/camera_x2 `
  --batch-size 64 `
  --num-workers 8
```

## Resume

```powershell
python train_sr.py `
  --train-hr dataset/DIV2K_train_HR `
  --train-hr dataset/Flickr2K_HR `
  --div2k-val-lr dataset/DIV2K_valid_LR_bicubic/X2 `
  --div2k-val-hr dataset/DIV2K_valid_HR `
  --output-dir runs/camera_x2 `
  --resume runs/camera_x2/latest.pth
```

## TensorBoard

```powershell
tensorboard --logdir runs/camera_x2/tensorboard
```
````

- [ ] **Step 7: Ignore generated runs while keeping exported release weights explicit**

Append to `.gitignore`:

```text
runs/
```

- [ ] **Step 8: Run all tests**

Run:

```powershell
python -m pytest tests -v
```

Expected: all tests pass with no warnings from project code.

- [ ] **Step 9: Commit visual validation and documentation**

```powershell
git add sr/visual.py tests/test_visual.py train_sr.py README.md .gitignore
git commit -m "docs: add camera validation and training guide"
```

### Task 11: RTX 4090 smoke test, overfit check, and full-run gate

**Files:**
- No code changes expected unless a failing check identifies a defect.

- [ ] **Step 1: Install dependencies in the laboratory environment**

Run:

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

Expected: the command identifies the RTX 4090 and reports a CUDA-enabled PyTorch build.

- [ ] **Step 2: Run the complete test suite**

Run:

```powershell
python -m pytest tests -v
```

Expected: all tests pass.

- [ ] **Step 3: Run a 200-step smoke training**

Run:

```powershell
python train_sr.py `
  --train-hr dataset/DIV2K_train_HR `
  --train-hr dataset/Flickr2K_HR `
  --div2k-val-lr dataset/DIV2K_valid_LR_bicubic/X2 `
  --div2k-val-hr dataset/DIV2K_valid_HR `
  --real-val-lr dataset/RealSR_valid_LR `
  --real-val-hr dataset/RealSR_valid_HR `
  --camera-dir dataset/camera_samples `
  --output-dir runs/smoke `
  --batch-size 64 `
  --num-workers 8 `
  --total-steps 200 `
  --stage1-steps 150 `
  --val-interval 100 `
  --checkpoint-interval 100
```

Expected:

- one training process starts;
- losses remain finite;
- `runs/smoke/latest.pth` and `runs/smoke/model.pth` exist;
- TensorBoard event files exist;
- two validation reports are written;
- camera sample outputs are 2× their input dimensions.

- [ ] **Step 4: Verify checkpoint resume**

Run the same command with:

```powershell
--total-steps 220 --resume runs/smoke/latest.pth
```

Expected: training resumes from step 200 and stops at step 220 without resetting the learning rate.

- [ ] **Step 5: Perform the fixed-batch overfit check**

Run:

```powershell
python -m pytest tests/test_trainer.py::test_fixed_batch_can_be_overfit -v
```

Expected: PASS; the fixed-batch L1 loss after 300 updates is lower than 25% of its initial value. If this fails, stop before the full run and investigate model/data alignment.

- [ ] **Step 6: Benchmark batches 64, 96, and 128**

Run the 200-step smoke command three times with `--batch-size 64`, `96`, and `128`, using separate output directories.

Expected: select the batch size with the highest stable samples/second, finite loss, and no CUDA out-of-memory error. Do not select by GPU utilization alone.

- [ ] **Step 7: Start the full 150,000-step run**

Run the documented training command with the selected batch size and default step counts.

Expected: checkpoints every 5,000 steps, validation every 2,000 steps, stage-two edge loss after step 120,000, and final EMA weights in `model.pth`.

- [ ] **Step 8: Record the verified baseline**

Append a `Verified RTX 4090 baseline` section to `README.md`. Copy the exact PyTorch/CUDA version, selected batch size, measured samples per second, best DIV2K RGB/Y PSNR and SSIM, and best RealSR RGB PSNR and SSIM from the completed run. Link the local artifact names `runs/camera_x2/latest.pth` and `runs/camera_x2/model.pth`. Do not enter estimated values.

- [ ] **Step 9: Commit only the measured documentation update**

```powershell
git add README.md
git commit -m "docs: record verified RTX 4090 baseline"
git push origin main
```
