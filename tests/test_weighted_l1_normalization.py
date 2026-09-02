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
