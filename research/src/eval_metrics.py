"""Torch-native PSNR and SSIM for DiffVax evaluation scripts.

These operate directly on (B,C,H,W) or (1,C,H,W) float tensors in [-1, 1]
range and return Python floats, making them drop-in for the eval scripts.
"""

import torch
import torch.nn.functional as F


def psnr(img_a: torch.Tensor, img_b: torch.Tensor, max_val: float = 2.0) -> float:
    """Compute PSNR between two (B,C,H,W) tensors in [-1, 1].

    max_val defaults to 2.0 because the signal range is [-1, 1] → span of 2.
    """
    mse = F.mse_loss(img_a.float(), img_b.float())
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(max_val**2 / mse)).item()


def _gaussian_kernel(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """1-D Gaussian kernel broadcast to (channels, 1, window_size, window_size)."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss /= gauss.sum()
    kernel_2d = gauss.unsqueeze(0) * gauss.unsqueeze(1)  # (W, W)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)       # (1, 1, W, W)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Compute mean SSIM between two (B,C,H,W) tensors in [-1, 1].

    Returns a Python float (mean SSIM over the batch).
    """
    a = img_a.float()
    b = img_b.float()
    B, C, H, W = a.shape

    kernel = _gaussian_kernel(window_size, sigma, C).to(a.device)
    pad = window_size // 2

    mu_a = F.conv2d(a, kernel, padding=pad, groups=C)
    mu_b = F.conv2d(b, kernel, padding=pad, groups=C)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a = F.conv2d(a * a, kernel, padding=pad, groups=C) - mu_a_sq
    sigma_b = F.conv2d(b * b, kernel, padding=pad, groups=C) - mu_b_sq
    sigma_ab = F.conv2d(a * b, kernel, padding=pad, groups=C) - mu_ab

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a + sigma_b + c2)
    ssim_map = numerator / denominator
    return ssim_map.mean().item()
