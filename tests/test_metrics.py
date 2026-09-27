import math

import pytest
import torch

from sr.metrics import (
    crop_border,
    psnr,
    rgb_to_y,
    ssim,
)

#border=2 会从上下左右各裁掉两个像素
def test_crop_border_removes_scale_pixels_from_every_side():
    image = torch.zeros(1, 3, 20, 30)

    cropped = crop_border(
        image,
        border=2,
    )

    assert cropped.shape == (1, 3, 16, 26)

#RGB 转 Y 使用 BT.601 标准范围
def test_rgb_to_y_uses_documented_bt601_conversion():
    black = torch.zeros(1, 3, 1, 1)
    white = torch.ones(1, 3, 1, 1)

    black_y = rgb_to_y(black)
    white_y = rgb_to_y(white)

    assert float(black_y) == pytest.approx(
        16.0 / 255.0,
        abs=1e-6,
    )
    assert float(white_y) == pytest.approx(
        235.0 / 255.0,
        abs=1e-6,
    )

#均方误差为 0.01 时，PSNR 应为 20 dB
def test_psnr_matches_known_error():
    prediction = torch.zeros(1, 3, 16, 16)
    target = torch.full_like(
        prediction,
        0.1,
    )

    value = psnr(
        prediction,
        target,
    )

    assert float(value) == pytest.approx(
        20.0,
        abs=1e-5,
    )

#相同图像的 SSIM 应接近 1
def test_ssim_is_one_for_identical_images_and_lower_for_different_images():
    generator = torch.Generator().manual_seed(7)
    target = torch.rand(
        2,
        1,
        32,
        32,
        generator=generator,
    )
    different = torch.clamp(
        target + 0.2,
        0.0,
        1.0,
    )

    identical_value = ssim(
        target,
        target,
    )
    different_value = ssim(
        different,
        target,
    )

    assert math.isfinite(
        float(different_value)
    )
    assert float(identical_value) == pytest.approx(
        1.0,
        abs=1e-5,
    )
    assert float(different_value) < float(
        identical_value
    )