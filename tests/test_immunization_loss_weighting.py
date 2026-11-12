"""Unit test for DiffVaxImmunization's loss1-weight selection logic.

Covers the OR-logic that decides whether loss1 is weighted toward the mask
region: true whenever a mask was actually used to produce img_out this batch
— either because the surrogate is mask-ONLY (is_inpainting, e.g. SD 1.5) or
because this particular call opted into the masked/RePaint path (e.g. SD3.5).
Exercised as a pure function rather than through the full training loop (DDP,
dataloaders, reporter, real attack models) since it's a 3-line conditional.
"""

import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffvax.immunization.diffvax_immunization import _select_loss1_weight  # noqa: E402


def test_loss1_weight_or_logic():
    mask = torch.tensor([[1.0]])
    ones = torch.ones_like(mask)

    # Full-image model, masked path NOT used this batch -> uniform weight.
    fake_model = types.SimpleNamespace(loss_uses_mask_weighting=False)
    assert _select_loss1_weight(fake_model, False, mask, ones) is ones

    # Full-image model, masked path USED this batch (e.g. SD3.5 RePaint) ->
    # mask weighting, even though the surrogate class itself is not mask-only.
    assert _select_loss1_weight(fake_model, True, mask, ones) is mask

    # Mask-only model (e.g. SD 1.5 inpainting) -> always mask weighting,
    # regardless of the per-batch flag.
    fake_model.loss_uses_mask_weighting = True
    assert _select_loss1_weight(fake_model, False, mask, ones) is mask
    assert _select_loss1_weight(fake_model, True, mask, ones) is mask
