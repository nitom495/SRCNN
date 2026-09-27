from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

#基本参数config
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
        #路径
        config = cls(
            data_root=data_root,
            output_dir=Path(output_dir),
            div2k_train_lr=(
                data_root / "DIV2K_train_LR_bicubic" / "X2"
            ),
            div2k_train_hr=data_root / "DIV2K_train_HR",
            flickr2k_train_lr=(
                data_root / "Flickr2K_train_LR_bicubic" / "X2"
            ),
            flickr2k_train_hr=data_root / "Flickr2K_HR",
            div2k_valid_lr=(
                data_root / "DIV2K_valid_LR_bicubic" / "X2"
            ),
            div2k_valid_hr=data_root / "DIV2K_valid_HR",
        )

        return replace(config, **overrides)

    def as_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }

    def save_json(
        self,
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = self.as_dict()

        if metadata is not None:
            payload.update(metadata)

        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )