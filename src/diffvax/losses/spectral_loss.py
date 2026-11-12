"""Frequency-domain perturbation concentration loss (H5).

Penalises low-frequency energy in the adversarial perturbation
δ = img_adv − img_orig via a 2-D real FFT magnitude penalty.

At high resolution (1088 × 1088) the L-inf ε = 32/255 budget is spread
over 4× more pixels than at 512 px.  L1 loss2 penalises all frequencies
equally.  Human visual sensitivity peaks in the mid-frequency band
(~2–10 cycles/degree) and drops sharply below that (smooth colour shifts
are the most conspicuous perturbation type).  Concentrating perturbation
energy in high-frequency bands achieves the same disruption effect at
higher SSIM / PSNR.

Basis:
  DDAP (arXiv:2407.20141) — frequency-aware adversarial perturbations
    outperform pixel-domain attacks on imperceptibility metrics.
  AdvAD (NeurIPS 2024) — frequency concentration improves SSIM by
    0.02–0.05 at the same epsilon budget.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class SpectralLoss:
    """DCT/FFT frequency-domain perturbation concentration loss.

    Discourages the perturbation network from placing energy in the
    low-frequency (visible) band of the perturbation.  Uses
    ``torch.fft.rfft2`` which is natively hardware-accelerated and
    equivalent in frequency-localisation to a 2-D DCT for this
    regularisation purpose.

    Config keys under ``spectral_loss``:
        enabled (bool):           Gate — set to true to activate.
        low_freq_radius (float):  Normalised radius below which components
                                  are penalised.  0.1 = central 10% of
                                  normalised frequency space (default).
                                  Increase to 0.2 for stronger push toward
                                  high frequencies.
        weight (float):           Scalar multiplier applied by LossComposer
                                  (not used internally).

    Args:
        config: Full training config dict.
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("spectral_loss", {})
        self.low_freq_radius = float(cfg.get("low_freq_radius", 0.1))
        # Pre-computed masks are cached by (H, W, device) to avoid
        # rebuilding on every forward pass.
        self._mask_cache: dict[tuple[int, int, str], Tensor] = {}

    def _low_freq_mask(self, H: int, W: int, device: torch.device) -> Tensor:
        """Return a (1, 1, H, W//2+1) boolean mask for low-frequency components.

        ``torch.fft.rfft2`` output layout:
          - Rows: DC at index 0, positive frequencies 1…H//2, then negative
            frequencies H//2+1…H−1 (mirrored, same energy).
          - Cols: DC at index 0, positive real-only frequencies 1…W//2
            (rfft2 exploits Hermitian symmetry, so no negative-frequency cols).

        We normalise row and column indices into [0, 1] and fold the row
        axis so that both DC and the corresponding negative-frequency rows
        map to low distance values.
        """
        key = (H, W, str(device))
        if key in self._mask_cache:
            return self._mask_cache[key]

        freq_w = W // 2 + 1

        # Normalise BOTH axes against their true Nyquist-relative range so
        # dist is comparable across axes and low_freq_radius means what it
        # says. fy correctly divides by H (the full row-axis FFT length, so
        # bins span the full [0,1) circle before folding). fx must divide by
        # W (the original width), NOT freq_w = W//2+1 — dividing by freq_w
        # was a bug: it stretched fx to span ~[0,1) instead of the correct
        # [0,0.5] (rfft2's column axis only ever holds non-negative
        # frequencies 0..W//2, i.e. bin k is frequency k/W of the sampling
        # rate, capping at (W//2)/W ≈ 0.5, not 1). The bug inflated fx by
        # ~2x, shrinking the effective captured region to roughly half the
        # configured radius. Verified via tests/test_spectral_loss_mask.py:
        # for W=100, radius=0.1, the correct boundary is bin 9 (last bin
        # with k/100 < 0.1); the buggy version put it at bin 5.
        fy = torch.arange(H, device=device).float() / H
        fx = torch.arange(freq_w, device=device).float() / W

        # Fold fy: negative-frequency rows are symmetric with positive ones.
        fy_centered = torch.min(fy, 1.0 - fy)

        fy_grid, fx_grid = torch.meshgrid(fy_centered, fx, indexing="ij")
        dist = (fy_grid ** 2 + fx_grid ** 2).sqrt()          # (H, W//2+1)

        mask = (dist < self.low_freq_radius).float()
        mask = mask.unsqueeze(0).unsqueeze(0)                 # (1, 1, H, W//2+1)

        self._mask_cache[key] = mask
        return mask

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        img_orig: Tensor,
        img_adv: Tensor,
        **kwargs,  # accept (and ignore) img_out, prompts passed by LossComposer
    ) -> Tensor:
        """Compute the spectral concentration loss.

        Args:
            img_orig: Clean image (B, 3, H, W), values in [-1, 1].
            img_adv:  Adversarial image (B, 3, H, W), values in [-1, 1].
            **kwargs: Extra arguments forwarded by LossComposer (ignored).

        Returns:
            Scalar tensor: mean low-frequency magnitude of the perturbation.
            Minimising this loss pushes perturbation energy into high-frequency
            bands.
        """
        # Compute perturbation; upcast to float32 — rfft2 requires real input
        # and operates incorrectly on float16.
        delta = (img_adv - img_orig).float()   # (B, 3, H, W)

        H, W = delta.shape[-2], delta.shape[-1]

        # 2-D real FFT with orthonormal normalisation so magnitude is
        # independent of resolution.
        delta_fft = torch.fft.rfft2(delta, norm="ortho")      # (B, 3, H, W//2+1)
        delta_mag = delta_fft.abs()

        mask = self._low_freq_mask(H, W, delta.device)        # (1, 1, H, W//2+1)

        # Mean low-frequency magnitude across batch, channels, and spatial dims.
        return (delta_mag * mask).mean()

    def __call__(
        self,
        img_orig: Tensor,
        img_adv: Tensor,
        **kwargs,
    ) -> Tensor:
        return self.forward(img_orig, img_adv, **kwargs)
