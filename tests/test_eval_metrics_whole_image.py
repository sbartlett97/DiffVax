"""Regression test: eval_multimodel.py's compute_image_metrics() must measure
the WHOLE edited image, not a masked sub-region.

Background: a previous "extract_mask_region" step called
recover_image(image, image, mask, ...) with the same image as both
arguments, which always returns that image unchanged regardless of the mask
(verified empirically — max pixel diff 0). So metrics were already
computed over the whole image in practice, but the code pretended otherwise.
This test proves the current (explicit, no-op-free) implementation actually
does what it claims: differences OUTSIDE where a mask would have been are
still detected, not silently ignored.

Uses real local metrics (PSNR/SSIM/FSIM — skimage/opencv/phasepack based, no
network) with a lightweight stub for CLIP (which needs a network model
download on first use) so this runs fully offline.
"""

import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from diffvax.metrics import MetricType, create_metric  # noqa: E402


@pytest.fixture
def real_metrics_with_stub_clip():
    import eval_multimodel as em

    class _StubClip:
        def __call__(self, images, prompts):
            return [0.0 for _ in images]

    return {
        "psnr": create_metric(MetricType.PSNR),
        "ssim": create_metric(MetricType.SSIM),
        "fsim": create_metric(MetricType.FSIM),
        "clip": _StubClip(),
    }, em


def _solid(color, size=(64, 64)):
    return Image.new("RGB", size, color=color)


def _half_and_half(left_color, right_color, size=(64, 64)):
    """Image that's left_color on the left half, right_color on the right."""
    img = Image.new("RGB", size, color=left_color)
    right = Image.new("RGB", (size[0] // 2, size[1]), color=right_color)
    img.paste(right, (size[0] // 2, 0))
    return img


def test_edit_metrics_detect_difference_outside_a_hypothetical_mask_region(
    real_metrics_with_stub_clip,
):
    """edited_orig and edited_imm differ ONLY in the right half of the image
    (imagine a mask that only covered the left half, e.g. an inpainting
    hole) — whole-image metrics must still detect this difference, proving
    the comparison isn't silently restricted to any sub-region."""
    metrics, em = real_metrics_with_stub_clip

    original = _solid((100, 100, 100))
    immunized = _solid((101, 101, 101))  # imperceptible perturbation

    # Identical on the left "hole", but the right "background" half differs
    # a lot — a masked-region-only comparison over just the left half would
    # report these as identical; a whole-image comparison must not.
    edited_orig = _half_and_half((50, 50, 50), (200, 200, 200))
    edited_imm = _half_and_half((50, 50, 50), (10, 10, 10))

    result = em.compute_image_metrics(
        metrics, original, immunized, edited_orig, edited_imm, prompt="a photo"
    )

    assert result["edit_ssim"] < 0.99, (
        "Whole-image SSIM should detect the large right-half difference — "
        "a masked-region-only comparison covering just the identical left "
        "half would have reported near-perfect similarity instead."
    )
    assert result["edit_psnr"] < 40, (
        "Whole-image PSNR should reflect the right-half difference, not "
        "read as near-identical."
    )


def test_edit_metrics_report_near_identical_when_whole_image_matches(
    real_metrics_with_stub_clip,
):
    """Sanity check in the other direction: when edited_orig and edited_imm
    really are (nearly) identical everywhere, metrics should say so."""
    metrics, em = real_metrics_with_stub_clip

    original = _solid((100, 100, 100))
    immunized = _solid((101, 101, 101))
    edited_orig = _solid((150, 150, 150))
    edited_imm = _solid((151, 151, 151))

    result = em.compute_image_metrics(
        metrics, original, immunized, edited_orig, edited_imm, prompt="a photo"
    )

    assert result["edit_ssim"] > 0.95
