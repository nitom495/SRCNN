import json
from pathlib import Path

from sr.config import TrainingConfig

#检查基本配置
def test_training_config_has_quality_training_defaults(tmp_path: Path):
    config = TrainingConfig.from_data_root(tmp_path / "dataset")

    assert config.scale == 2
    assert config.lr_patch_size == 128
    assert config.batch_size == 64
    assert config.max_steps == 150_000
    assert config.learning_rate == 2e-4
    assert config.minimum_learning_rate == 1e-6
    assert config.div2k_train_hr == tmp_path / "dataset" / "DIV2K_train_HR"
    assert (
        config.flickr2k_train_lr
        == tmp_path / "dataset" / "Flickr2K_train_LR_bicubic" / "X2"
    )

#检查保存路径
def test_save_json_converts_paths_and_writes_resolved_config(tmp_path: Path):
    config = TrainingConfig.from_data_root(
        tmp_path / "dataset",
        output_dir=tmp_path / "output",
    )

    config.save_json(
        tmp_path / "resolved.json",
        metadata={"train_pairs": 3_450, "validation_pairs": 100},
    )
    saved = json.loads(
        (tmp_path / "resolved.json").read_text(encoding="utf-8")
    )

    assert saved["data_root"] == str(tmp_path / "dataset")
    assert saved["output_dir"] == str(tmp_path / "output")
    assert saved["max_steps"] == 150_000
    assert saved["train_pairs"] == 3_450
    assert saved["validation_pairs"] == 100
