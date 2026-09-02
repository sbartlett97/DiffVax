"""Regression test for _weighted_l1()'s channel normalization.

Bug: loss1/loss2 multiplied a (B, 3, H, W) image diff by a (B, 1, H, W)
weight mask, then divided the resulting L1 sum by weight.sum() — which counts
spatial positions only (1 channel), not the 3 channels actually summed over.
That inflated both losses by exactly the channel count (3 for RGB), matching
the observed loss1=3.000 plateau with a noise target: true mean abs diff
~= 1.0, x3 bug -> 3.000.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffvax.immunization.diffvax_immunization import _weighted_l1  # noqa: E402


def test_weighted_l1_uniform_weight_matches_plain_mean_abs():
    diff = torch.full((1, 3, 4, 4), 2.0)
    weight = torch.ones(1, 1, 4, 4)
    assert torch.isclose(_weighted_l1(diff, weight), torch.tensor(2.0))


def test_weighted_l1_is_exactly_a_third_of_the_pre_fix_buggy_formula():
    diff = torch.full((1, 3, 4, 4), 1.0)
    weight = torch.ones(1, 1, 4, 4)
    correct = _weighted_l1(diff, weight)
    buggy = (diff * weight).norm(p=1) / weight.sum()  # pre-fix formula
    assert torch.isclose(buggy, correct * 3)


def test_weighted_l1_restricts_to_weighted_region():
    diff = torch.zeros(1, 3, 4, 4)
    diff[:, :, :, :2] = 4.0  # nonzero only in the left half
    weight = torch.zeros(1, 1, 4, 4)
    weight[:, :, :, :2] = 1.0  # weight matches the nonzero region exactly
    assert torch.isclose(_weighted_l1(diff, weight), torch.tensor(4.0))


def test_weighted_l1_floors_a_tiny_weighted_region():
    """A real face mask can be as small as ~0.02% of the frame. Without a
    floor, weight.sum() shrinks with the region and 1/weight.sum() amplifies
    that region's gradient by up to ~5000x relative to full-image weighting
    -- large enough to overflow fp16 during backward (observed directly: a
    real training run's GradScaler collapsed to ~0 from very early on after
    perturbation_mask_gating started producing small-area weight tensors).
    The floor caps the amplification at 1/min_area_frac regardless of how
    small the real region is.
    """
    total = 100 * 100
    diff = torch.ones(1, 3, 100, 100)
    weight = torch.zeros(1, 1, 100, 100)
    weight.view(-1)[:1] = 1.0  # 1 pixel out of 10000 -> 0.01% of the frame

    min_area_frac = 0.01
    result = _weighted_l1(diff, weight, min_area_frac=min_area_frac)
    unfloored = (diff * weight).norm(p=1) / (weight.sum() * 3)

    # Floored result must match clamping weight.sum() at min_area_frac*total.
    expected = (diff * weight).norm(p=1) / (min_area_frac * total * 3)
    assert torch.isclose(result, expected)
    # And it must be far smaller than what the unfloored formula would give
    # (the whole point: capping the amplification, not just reducing it).
    assert result < unfloored / 50
