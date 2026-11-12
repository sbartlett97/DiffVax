"""Regression test for SpectralLoss._low_freq_mask()'s frequency normalization.

Bug: the column-frequency axis (fx) divided by freq_w = W//2+1 instead of
the true Nyquist normalization W. rfft2's column axis only holds
non-negative frequencies 0..W//2 (bin k = frequency k/W of the sampling
rate, capping at (W//2)/W ~= 0.5) — dividing by freq_w instead inflated fx
values by ~2x, shrinking the effective captured region to roughly half of
whatever low_freq_radius was configured to mean.

This test picks W=100 (round numbers, easy to reason about exactly) and
checks the mask boundary along the DC row (fy_centered=0, so dist==fx)
lands exactly where low_freq_radius=0.1 says it should: bin 9 is the last
included bin (9/100 = 0.09 < 0.1), bin 10 is the first excluded bin
(10/100 = 0.10, not < 0.1). The pre-fix implementation put this boundary at
bin 5 instead (5/51 ~= 0.098 < 0.1, but 6/51 ~= 0.118 excluded) — this test
would have failed against it.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffvax.losses.spectral_loss import SpectralLoss  # noqa: E402


def test_low_freq_mask_boundary_matches_configured_radius():
    sl = SpectralLoss({"spectral_loss": {"low_freq_radius": 0.1}})
    mask = sl._low_freq_mask(H=100, W=100, device=torch.device("cpu"))

    # DC row (row index 0, fy_centered=0) -> dist along this row equals fx
    # directly, so the mask boundary here reads out the fx normalization
    # exactly.
    dc_row = mask[0, 0, 0, :]

    assert dc_row[9] == 1.0, "bin 9 (freq=0.09) should be inside radius 0.1"
    assert dc_row[10] == 0.0, "bin 10 (freq=0.10) should be outside radius 0.1 (not <)"
    assert dc_row[5] == 1.0, (
        "bin 5 (freq=0.05) should be inside radius 0.1 — the pre-fix bug put "
        "the boundary here instead of bin 9/10, roughly halving the intended radius"
    )


def test_low_freq_mask_always_includes_dc():
    """The zero-frequency (DC) bin must always be included regardless of
    resolution — it's the (0,0) corner where both axes are exactly 0."""
    for H, W in [(512, 512), (768, 768), (1024, 1024), (1088, 1088), (300, 500)]:
        sl = SpectralLoss({"spectral_loss": {"low_freq_radius": 0.1}})
        mask = sl._low_freq_mask(H, W, torch.device("cpu"))
        assert mask[0, 0, 0, 0] == 1.0, f"DC bin missing at H={H}, W={W}"


def test_low_freq_mask_fraction_is_resolution_independent():
    """A correctly-normalized mask should capture roughly the same FRACTION
    of the frequency grid regardless of resolution, since low_freq_radius is
    defined in normalized (resolution-independent) frequency space."""
    fractions = []
    for size in [256, 512, 768, 1024]:
        sl = SpectralLoss({"spectral_loss": {"low_freq_radius": 0.1}})
        mask = sl._low_freq_mask(size, size, torch.device("cpu"))
        fractions.append(mask.float().mean().item())

    # All fractions should be close to each other (within a small tolerance
    # for discretization effects at low resolution).
    assert max(fractions) - min(fractions) < 0.01, (
        f"mask fraction should be ~resolution-independent, got {fractions}"
    )
