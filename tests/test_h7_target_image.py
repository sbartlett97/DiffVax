"""Unit tests for the H7 fixed-target-image path in DiffVaxImmunization.

Covers _load_target_image_tensor in isolation (bypassing the heavy
DiffVaxImmunization.__init__ — no attack_manager/optimizer needed — same
pattern as test_attack_gradient_flow.py's make_sd3_attack()).

Background: loss1's original H7 target was a fixed random ±1 noise pattern.
A real training run showed loss1 pinned at its exact theoretical chance-level
value for 58+ epochs: comparing a smooth generated image against independent
per-pixel random noise via mean L1 distance saturates near 1.0 for virtually
any image content (law of large numbers), so that objective couldn't
distinguish a disrupted output from an undisrupted one. Replacing the noise
pattern with a fixed real image (noise_target.image_path) gives loss1 genuine
spatial structure to push the generated output toward or away from.
"""

import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffvax.immunization.diffvax_immunization import DiffVaxImmunization  # noqa: E402


def _make_immun_with_target_image(tmp_path, size=(37, 29)):
    """Construct a DiffVaxImmunization with only _target_image_source set,
    bypassing __init__ (which needs a real attack_manager/optimizer)."""
    img_path = tmp_path / "fake_target.png"
    # Deliberately non-square, non-power-of-2 source size to prove resizing
    # (not just cropping/assuming square) actually happens.
    Image.new("RGB", size, color=(10, 200, 30)).save(img_path)

    immun = DiffVaxImmunization.__new__(DiffVaxImmunization)
    immun._target_image_source = Image.open(str(img_path)).convert("RGB")
    return immun


def test_load_target_image_tensor_shape_and_range(tmp_path):
    immun = _make_immun_with_target_image(tmp_path)
    shape = (2, 3, 64, 64)  # (B, C, H, W)
    t = immun._load_target_image_tensor(shape, dtype=torch.float32, device=torch.device("cpu"))

    assert t.shape == shape
    assert t.dtype == torch.float32
    assert t.device.type == "cpu"
    assert torch.all(t >= -1.0) and torch.all(t <= 1.0), (
        "Target image tensor must be normalized to [-1, 1] like utils.load_image"
    )


def test_load_target_image_tensor_batch_elements_identical(tmp_path):
    """Every batch element should see the SAME fixed target image."""
    immun = _make_immun_with_target_image(tmp_path)
    t = immun._load_target_image_tensor(
        (3, 3, 32, 32), dtype=torch.float32, device=torch.device("cpu")
    )
    assert torch.allclose(t[0], t[1]) and torch.allclose(t[1], t[2])


def test_load_target_image_tensor_resizes_to_requested_shape(tmp_path):
    """Source image is 37x29 (non-square); requested shape must win."""
    immun = _make_immun_with_target_image(tmp_path, size=(37, 29))
    for h, w in [(16, 16), (48, 32), (100, 100)]:
        t = immun._load_target_image_tensor(
            (1, 3, h, w), dtype=torch.float32, device=torch.device("cpu")
        )
        assert t.shape == (1, 3, h, w)


def test_load_target_image_tensor_approximates_known_color(tmp_path):
    """A solid-color source image, requested at its own size, should decode
    back to approximately the same normalized color (sanity check on the
    /127.5 - 1.0 normalization direction and RGB channel order)."""
    immun = _make_immun_with_target_image(tmp_path, size=(20, 20))
    t = immun._load_target_image_tensor(
        (1, 3, 20, 20), dtype=torch.float32, device=torch.device("cpu")
    )
    expected = torch.tensor([10, 200, 30], dtype=torch.float32) / 127.5 - 1.0
    actual = t[0, :, 10, 10]
    assert torch.allclose(actual, expected, atol=1e-3)
