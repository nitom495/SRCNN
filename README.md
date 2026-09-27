# 轻量级 RGB ×2 超分辨率

本项目使用 DF2K 数据集，以 FP32 训练一个具有 7,124 个参数的轻量级 RGB ×2 PixelShuffle 超分模型。

当前阶段只训练标准 bicubic ×2 超分模型，不包含摄像头退化、视频采集、INT8 量化和 FPGA 部署。这些内容将在后续阶段处理。

## 数据集目录

请将标准 bicubic ×2 配对数据放入 `dataset/`：

```text
dataset/
├── DIV2K_train_HR/                 # 800 张 HR 训练图片
├── DIV2K_train_LR_bicubic/
│   └── X2/                         # 800 张 LR 图片，例如 0001x2.png
├── Flickr2K_HR/                    # 2650 张 HR 训练图片
├── Flickr2K_train_LR_bicubic/
│   └── X2/                         # 2650 张 LR 图片，例如 000001x2.png
├── DIV2K_valid_HR/                 # 100 张 HR 验证图片
└── DIV2K_valid_LR_bicubic/
    └── X2/                         # 100 张 LR 图片，例如 0801x2.png
```

程序需要以下六个目录：

- `DIV2K_train_HR`
- `DIV2K_train_LR_bicubic/X2`
- `Flickr2K_HR`
- `Flickr2K_train_LR_bicubic/X2`
- `DIV2K_valid_HR`
- `DIV2K_valid_LR_bicubic/X2`

程序会在训练开始前检查目录、图片数量、LR/HR 配对关系以及 ×2 尺寸关系。如果检查失败，程序会停止并显示具体错误。数据集内容已通过 Git 忽略规则排除，不会上传到仓库。

## 环境检查

请在支持 CUDA 12.8 的 Python 3.11 PyTorch 环境中运行：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m pytest -q
```

CUDA 检查应输出 `True`，并显示 NVIDIA GPU 名称。测试命令应全部通过。

## 开始训练

RTX 4090 的默认训练命令：

```powershell
python train_sr.py --data-root dataset --output-dir outputs/df2k_x2
```

默认配置：

- 放大倍数：×2
- LR 训练图块：`128×128`
- HR 标签图块：`256×256`
- batch size：64
- 训练步数：150,000
- 初始学习率：`2e-4`
- 最低学习率：`1e-6`
- 数据加载进程数：4
- 训练精度：FP32

如果 RTX 4090 的 24 GB 显存利用率较低，可以依次测试 batch size 64、96 和 128，并选择不会发生显存不足的最大值。第一次对比时保持 150,000 步训练计划不变。

## 断点续训

如果训练中断，可以从 `latest.pth` 恢复：

```powershell
python train_sr.py --data-root dataset --output-dir outputs/df2k_x2 --resume outputs/df2k_x2/latest.pth
```

恢复内容包括：

- 模型参数；
- 优化器状态；
- 学习率调度器状态；
- 当前训练步数；
- 历史最佳 Y 通道 PSNR；
- 随机数生成器状态。

## 查看 TensorBoard

运行：

```powershell
tensorboard --logdir outputs/df2k_x2/tensorboard
```

TensorBoard 会记录训练 L1 损失、学习率、吞吐率、验证集 RGB PSNR、Y 通道 PSNR 和 SSIM。

## 训练输出

默认输出目录中会生成：

- `latest.pth`：最新的完整训练断点；
- `best_y_psnr.pth`：Y 通道 PSNR 最好的完整断点；
- `checkpoint_XXXXXX.pth`：定期保存的编号断点；
- `model_weights.pth`：仅包含模型参数的推理权重；
- `config.json`：本次训练使用的完整配置；
- `tensorboard/`：TensorBoard 日志。
