"""Differentiable Expectation over Transformations (EoT) augmentation.

Inserts a differentiable augmentation pipeline between perturbation application
and the attack model forward pass so that gradients flow through transforms
back to the NestedUNet. Simulates real-world preprocessing pipelines (social
media, API compression, resize-on-upload) that would otherwise nullify the
immunization.
"""

import math
import random

import torch
import torch.nn.functional as F
from torch import Tensor


class DifferentiableEoT:
    """Differentiable augmentation pipeline for EoT-based immunization training.

    Constructor parameters (from config['eot'] dict):
        jpeg_quality_range: [int, int]   – uniform sample quality for DiffJPEG
        resize_range:       [float, float] – random scale factor (resize then back)
        blur_sigma_range:   [float, float] – Gaussian blur sigma (0 = no blur)
        noise_std_range:    [float, float] – additive Gaussian noise sigma
        p_jpeg:   float – probability of applying JPEG
        p_resize: float – probability of applying resize
        p_blur:   float – probability of applying blur
        p_noise:  float – probability of applying noise
        enabled:  bool  – master toggle (False = passthrough)

    All operations preserve gradient flow. JPEG uses kornia when available,
    falls back to a passthrough if the library is not installed.
    Pipeline order: resize → JPEG → blur → noise → clamp to [-1, 1].
    """

    def __init__(self, config: dict):
        cfg = config.get("eot", {})
        self.enabled = cfg.get("enabled", True)
        self.jpeg_quality_range = cfg.get("jpeg_quality_range", [75, 95])
        self.resize_range = cfg.get("resize_range", [0.5, 2.0])
        self.blur_sigma_range = cfg.get("blur_sigma_range", [0.0, 2.0])
        self.noise_std_range = cfg.get("noise_std_range", [0.0, 0.03])
        self.p_jpeg = float(cfg.get("p_jpeg", 0.8))
        self.p_resize = cfg.get("p_resize", 0.5)
        self.p_blur = cfg.get("p_blur", 0.3)
        self.p_noise = cfg.get("p_noise", 0.3)

        # Fail loudly if JPEG is requested but kornia is missing. A silent
        # passthrough would train a model believed to be JPEG-robust but isn't.
        if self.p_jpeg > 0:
            try:
                import kornia.enhance  # noqa: F401
            except ImportError:
                raise ImportError(
                    "DifferentiableEoT: p_jpeg > 0 requires kornia. "
                    "Install it with: pip install kornia\n"
                    "Or set p_jpeg: 0 in the eot config to disable JPEG augmentation."
                )
    def _apply_jpeg(self, x: Tensor) -> Tensor:
        """Apply differentiable JPEG compression via kornia.

        Falls back to identity if kornia is not installed.
        Expects input in [-1, 1]; converts to [0, 1] internally.
        Quality is sampled per call as a per-image tensor of shape (B,).
        """
        try:
            from kornia.enhance import jpeg_codec_differentiable
        except ImportError:
            return x

        quality = random.randint(
            int(self.jpeg_quality_range[0]), int(self.jpeg_quality_range[1])
        )
        x_01 = (x.float() + 1.0) / 2.0
        quality_t = torch.full(
            (x.shape[0],), float(quality), dtype=torch.float32, device=x.device
        )
        x_jpg = jpeg_codec_differentiable(x_01, quality_t)
        return ((x_jpg * 2.0) - 1.0).to(x.dtype)

    def _apply_resize(self, x: Tensor) -> Tensor:
        """Scale down and back to original size (round-trip differentiable resize)."""
        scale = random.uniform(float(self.resize_range[0]), float(self.resize_range[1]))
        h, w = x.shape[2], x.shape[3]
        new_h = max(16, int(h * scale))
        new_w = max(16, int(w * scale))

        x_float = x.float()
        x_small = F.interpolate(
            x_float, (new_h, new_w), mode="bilinear", align_corners=False
        )
        x_back = F.interpolate(
            x_small, (h, w), mode="bilinear", align_corners=False
        )
        return x_back.to(x.dtype)

    def _apply_blur(self, x: Tensor) -> Tensor:
        """Apply differentiable Gaussian blur using depthwise convolution."""
        sigma = random.uniform(
            float(self.blur_sigma_range[0]), float(self.blur_sigma_range[1])
        )
        if sigma < 1e-5:
            return x

        kernel_size = int(2 * math.ceil(3.0 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)

        coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device)
        coords = coords - kernel_size // 2
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        g = g / g.sum()

        kernel_2d = g.outer(g)  # (k, k)
        # Depthwise: one kernel per input channel
        kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(
            x.shape[1], 1, kernel_size, kernel_size
        )

        pad = kernel_size // 2
        x_float = x.float()
        x_padded = F.pad(x_float, [pad, pad, pad, pad], mode="reflect")
        x_blur = F.conv2d(x_padded, kernel.to(x_float.dtype), groups=x.shape[1])
        return x_blur.to(x.dtype)

    def _apply_noise(self, x: Tensor) -> Tensor:
        """Add differentiable Gaussian noise (straight-through via additive path)."""
        std = random.uniform(
            float(self.noise_std_range[0]), float(self.noise_std_range[1])
        )
        if std < 1e-8:
            return x
        noise = torch.randn_like(x.float()) * std
        return (x.float() + noise).to(x.dtype)

    def __call__(self, img_adv: Tensor) -> Tensor:
        """Apply a random subset of transforms in fixed pipeline order.

        Order: resize → JPEG → blur → noise → clamp to [-1, 1].
        Each transform is applied independently with its own probability.

        Args:
            img_adv: Adversarial image tensor in [-1, 1], shape (B, C, H, W).

        Returns:
            Augmented tensor with gradient flow intact, clamped to [-1, 1].
        """
        if not self.enabled:
            return img_adv

        x = img_adv

        if random.random() < self.p_resize:
            x = self._apply_resize(x)
        if random.random() < self.p_jpeg:
            x = self._apply_jpeg(x)
        if random.random() < self.p_blur:
            x = self._apply_blur(x)
        if random.random() < self.p_noise:
            x = self._apply_noise(x)

        return torch.clamp(x, -1.0, 1.0)
