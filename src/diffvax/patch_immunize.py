"""Patch-based inference for high-resolution image immunization.

Enables a NestedUNet immunization model trained at 512x512 to produce
coherent immunizations on arbitrarily-sized images (e.g. 1088x1088) via
overlapping patch inference with Gaussian-weighted blending.

The NestedUNet is fully convolutional -- no positional encodings -- so it
generalises to patch sizes other than the training resolution. Blending
with a smooth Gaussian weight window eliminates hard boundary artefacts.

Usage::

    from diffvax.patch_immunize import patch_immunize
    from diffvax.model import NestedUNet

    model = NestedUNet(num_classes=3).cuda()
    model.load_state_dict(torch.load("checkpoints/diffvax_trained.pth"))
    model.training = False  # set inference mode

    img = ...  # (1, 3, 1088, 1088) tensor in [-1, 1]
    mask = ...  # (1, 1, 1088, 1088) tensor in [0, 1]
    immunized = patch_immunize(model, img, mask, patch_size=512, stride=256)  # 50% overlap required at 1088px
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional


def _gaussian_window(size: int, sigma: float = None, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Create a 2-D Gaussian weight window of shape (1, 1, size, size).

    The window peaks at 1.0 in the centre and falls off to ~0 at the edges,
    producing smooth blending when patches overlap.

    Args:
        size: window side length (must match patch_size).
        sigma: Gaussian std-dev. Defaults to size / 6 (covers +/-3 sigma).
        device: tensor device.
        dtype: tensor dtype.
    """
    if sigma is None:
        sigma = size / 6.0

    half = size // 2
    x = torch.arange(size, device=device, dtype=dtype) - half
    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
    gauss_2d = gauss_2d / gauss_2d.max()  # normalise to [0, 1]
    return gauss_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def patch_immunize(
    model: torch.nn.Module,
    image: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int = 512,
    stride: int = 256,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
    dtype: torch.dtype = torch.float32,
    sigma: Optional[float] = None,
) -> torch.Tensor:
    """Apply immunization model to a large image via overlapping patch inference.

    The model is run in inference mode (no gradient tracking). The caller is
    responsible for putting the model into non-training mode before calling this
    function (model.training = False, or torch.inference_mode context).

    Args:
        model: trained NestedUNet (or compatible) immunization model, on CUDA.
        image: input image tensor in [-1, 1], shape (1, 3, H, W). On CUDA.
        mask: edit mask tensor in [0, 1], shape (1, 1, H, W). On CUDA.
            1 = edit region (perturbation NOT applied here).
        patch_size: size of each square patch (default 512, matching training res).
        stride: stride between patch top-left corners. Smaller = more overlap =
            smoother blends, but slower. Recommended: stride = patch_size * 0.75.
        clamp_min: minimum pixel value.
        clamp_max: maximum pixel value.
        dtype: computation dtype.
        sigma: Gaussian window sigma override. Default: patch_size / 6.

    Returns:
        Immunized image tensor in [-1, 1], shape (1, 3, H, W).
    """
    B, C, H, W = image.shape
    assert B == 1, "patch_immunize currently supports batch_size=1 only."

    device = image.device
    img = image.to(dtype=dtype)
    msk = mask.to(dtype=dtype)

    # Accumulation buffers
    perturb_sum = torch.zeros(1, C, H, W, device=device, dtype=dtype)
    weight_sum = torch.zeros(1, 1, H, W, device=device, dtype=dtype)

    weight_window = _gaussian_window(patch_size, sigma=sigma, device=device, dtype=dtype)

    # Reflect-pad the image so all patches are valid
    pad_h = max((math.ceil((H - patch_size) / stride) * stride + patch_size) - H, 0)
    pad_w = max((math.ceil((W - patch_size) / stride) * stride + patch_size) - W, 0)

    img_padded = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
    msk_padded = F.pad(msk, (0, pad_w, 0, pad_h), mode="reflect")

    ph = img_padded.shape[2]
    pw = img_padded.shape[3]

    # Padded accumulation buffers
    ps_pad = torch.zeros(1, C, ph, pw, device=device, dtype=dtype)
    ws_pad = torch.zeros(1, 1, ph, pw, device=device, dtype=dtype)

    with torch.no_grad():
        for y in range(0, ph - patch_size + 1, stride):
            for x in range(0, pw - patch_size + 1, stride):
                patch = img_padded[:, :, y:y + patch_size, x:x + patch_size]
                patch_mask = msk_padded[:, :, y:y + patch_size, x:x + patch_size]

                with torch.autocast("cuda", dtype=dtype):
                    patch_perturb = model(patch)  # (1, 3, patch_size, patch_size)

                # Zero out perturbation inside edit region (mask=1)
                patch_perturb = patch_perturb * (1 - patch_mask)

                w = weight_window
                ps_pad[:, :, y:y + patch_size, x:x + patch_size] += patch_perturb * w
                ws_pad[:, :, y:y + patch_size, x:x + patch_size] += w.expand(1, 1, -1, -1)

    # Crop back to original size
    ps_crop = ps_pad[:, :, :H, :W]
    ws_crop = ws_pad[:, :, :H, :W].clamp(min=1e-8)

    perturbation = ps_crop / ws_crop  # (1, 3, H, W)
    immunized = torch.clamp(img + perturbation, clamp_min, clamp_max)
    return immunized.to(image.dtype)
