"""JPEG augmentation for compression-robust immunization training.

Applies JPEG compression to immunized images during training so that the
immunization network learns to place perturbation energy in DCT frequency
bands that survive JPEG re-compression (q=70-75, typical for Instagram/Twitter).

Uses the Straight-Through Estimator (STE) to allow gradient flow through
the non-differentiable JPEG operation:
  - Forward pass: uses JPEG-compressed image (realistic)
  - Backward pass: gradients flow as if JPEG were identity (trainable)

Reference: DCT-Shield (ICCV 2025, arXiv:2504.17894)

Social media compression context:
  - Instagram: ~q=75 JPEG equivalent on upload
  - Twitter/X:  strong JPEG re-compression (quality not disclosed, ~q=70)
  - Implication: perturbations must survive q=70-75 to be useful in production
"""

import io
import random
from typing import Union

import torch
import torchvision.transforms.functional as TF
from PIL import Image


def jpeg_compress_tensor(
    img: torch.Tensor,
    quality: int,
) -> torch.Tensor:
    """Apply JPEG compression to a (1, 3, H, W) tensor in [-1, 1].

    Non-differentiable. Use jpeg_augment_ste for training.

    Args:
        img: (1, 3, H, W) float tensor in [-1, 1].
        quality: JPEG quality 1–95.

    Returns:
        JPEG-compressed tensor of same shape in [-1, 1].
    """
    pil = TF.to_pil_image((img.squeeze(0).float().cpu().clamp(-1, 1) + 1) / 2)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed = TF.to_tensor(Image.open(buf).convert("RGB"))
    return (compressed.unsqueeze(0) * 2 - 1).to(img.device, img.dtype)


def jpeg_augment_ste(
    img_immunized: torch.Tensor,
    quality: Union[int, None] = None,
    quality_range: tuple = (70, 85),
) -> torch.Tensor:
    """JPEG augmentation with Straight-Through Estimator for training.

    Forward: returns JPEG-compressed image (forces JPEG robustness)
    Backward: gradients flow through as if compression were identity

    Args:
        img_immunized: (1, 3, H, W) float tensor in [-1, 1], requires_grad-able.
        quality: fixed JPEG quality. If None, samples uniformly from quality_range.
        quality_range: (min_q, max_q) for random quality sampling.

    Returns:
        Tensor with same shape. Forward value = JPEG compressed; gradient = STE.
    """
    if quality is None:
        quality = random.randint(quality_range[0], quality_range[1])

    with torch.no_grad():
        compressed = jpeg_compress_tensor(img_immunized, quality=quality)

    # STE: forward uses compressed, backward flows through img_immunized
    return compressed + (img_immunized - img_immunized.detach())


def should_apply_jpeg(prob: float) -> bool:
    """Return True with probability `prob`."""
    return random.random() < prob
