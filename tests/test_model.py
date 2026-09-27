import torch
import pytest

import sr

from sr.model import FastSRNet

#检查输出图片通道和尺寸NCHW
def test_model_outputs_rgb_at_twice_the_input_size():
    model = FastSRNet(scale=2)
    input_tensor = torch.rand(2, 3, 24, 32)

    output = model(input_tensor)

    assert output.shape == (2, 3, 48, 64)

#检查模型参数量
def test_model_preserves_existing_parameter_count():
    model = FastSRNet(scale=2)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert parameter_count == 7_124

#检查放大尺寸
def test_model_rejects_unsupported_scale():
    with pytest.raises(ValueError, match="scale=2"):
        FastSRNet(scale=3)

#导出fastsrnet
def test_package_exports_model():
    assert sr.FastSRNet is FastSRNet
    assert "FastSRNet" in sr.__all__
