from pathlib import Path


def test_generated_training_directories_are_ignored():
    ignored = (
        Path(".gitignore")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    assert "dataset/" in ignored
    assert "outputs/" in ignored
    assert "models/" in ignored
    assert ".pytest_cache/" in ignored


def test_readme_documents_required_dataset_directories():
    readme = Path("README.md").read_text(
        encoding="utf-8"
    )

    required_directories = [
        "DIV2K_train_HR",
        "DIV2K_train_LR_bicubic/X2",
        "Flickr2K_HR",
        "Flickr2K_train_LR_bicubic/X2",
        "DIV2K_valid_HR",
        "DIV2K_valid_LR_bicubic/X2",
    ]

    for directory in required_directories:
        assert directory in readme