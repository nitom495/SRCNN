import torch
import torch.nn.functional as F


def crop_border(
    image: torch.Tensor,
    border: int,
) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(
            f"expected NCHW tensor, "
            f"got shape {tuple(image.shape)}"
        )

    if border < 0:
        raise ValueError(
            "border must be non-negative"
        )

    if border == 0:
        return image

    if (
        image.shape[-2] <= 2 * border
        or image.shape[-1] <= 2 * border
    ):
        raise ValueError(
            "border is too large for image dimensions"
        )

    return image[
        ...,
        border:-border,
        border:-border,
    ]


def rgb_to_y(
    image: torch.Tensor,
) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(
            f"expected NCHW RGB tensor, "
            f"got shape {tuple(image.shape)}"
        )

    coefficients = image.new_tensor(
        [65.481, 128.553, 24.966]
    ).view(1, 3, 1, 1)

    y = (
        (image * coefficients).sum(
            dim=1,
            keepdim=True,
        )
        / 255.0
        + 16.0 / 255.0
    )

    return y


def psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target shapes must match"
        )

    mse = (
        (prediction - target)
        .square()
        .flatten(1)
        .mean(dim=1)
    )

    values = torch.where(
        mse == 0,
        torch.full_like(
            mse,
            float("inf"),
        ),
        10.0 * torch.log10(1.0 / mse),
    )

    return values.mean()


def _gaussian_window(
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = (
        torch.arange(
            11,
            device=device,
            dtype=dtype,
        )
        - 5
    )

    kernel_1d = torch.exp(
        -coordinates.square()
        / (2 * 1.5**2)
    )
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_2d = torch.outer(
        kernel_1d,
        kernel_1d,
    )

    return (
        kernel_2d
        .expand(channels, 1, 11, 11)
        .contiguous()
    )


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target shapes must match"
        )

    if (
        prediction.ndim != 4
        or min(prediction.shape[-2:]) < 11
    ):
        raise ValueError(
            "SSIM expects NCHW images "
            "at least 11x11"
        )

    channels = prediction.shape[1]

    window = _gaussian_window(
        channels=channels,
        device=prediction.device,
        dtype=prediction.dtype,
    )

    mean_prediction = F.conv2d(
        prediction,
        window,
        groups=channels,
    )
    mean_target = F.conv2d(
        target,
        window,
        groups=channels,
    )

    mean_prediction_sq = mean_prediction.square()
    mean_target_sq = mean_target.square()
    mean_product = mean_prediction * mean_target

    variance_prediction = (
        F.conv2d(
            prediction.square(),
            window,
            groups=channels,
        )
        - mean_prediction_sq
    )
    variance_target = (
        F.conv2d(
            target.square(),
            window,
            groups=channels,
        )
        - mean_target_sq
    )
    covariance = (
        F.conv2d(
            prediction * target,
            window,
            groups=channels,
        )
        - mean_product
    )

    c1 = 0.01**2
    c2 = 0.03**2

    numerator = (
        (2 * mean_product + c1)
        * (2 * covariance + c2)
    )
    denominator = (
        mean_prediction_sq
        + mean_target_sq
        + c1
    ) * (
        variance_prediction
        + variance_target
        + c2
    )

    score = numerator / denominator

    return (
        score
        .flatten(1)
        .mean(dim=1)
        .mean()
    )