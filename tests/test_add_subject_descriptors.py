"""Unit tests for the deterministic parts of scripts/add_subject_descriptors.py
(string substitution, mask-bbox cropping) — no CLIP model / network needed.
"""

import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from add_subject_descriptors import substitute_subject, crop_to_mask_bbox  # noqa: E402


@pytest.mark.parametrize(
    "prompt,candidate,expected",
    [
        ("A person in a game arcade", "a young girl", "A young girl in a game arcade"),
        ("a person in a ski shop", "a young girl", "a young girl in a ski shop"),
        ("A person in a market", "an elderly man", "An elderly man in a market"),
        ("a person waving", "an elderly woman", "an elderly woman waving"),
    ],
)
def test_substitute_subject_matches_case_and_uses_candidates_own_article(
    prompt, candidate, expected
):
    assert substitute_subject(prompt, candidate) == expected


def test_substitute_subject_only_replaces_first_occurrence():
    prompt = "A person is talking to a person nearby"
    result = substitute_subject(prompt, "a young boy")
    assert result == "A young boy is talking to a person nearby"


def test_substitute_subject_no_match_returns_unchanged():
    prompt = "Turn this into a pencil sketch"
    assert substitute_subject(prompt, "a young girl") == prompt


def test_substitute_subject_does_not_match_substring_words():
    """'a personality trait' must not match 'a person' as a prefix."""
    prompt = "a personality trait is visible"
    assert substitute_subject(prompt, "a young girl") == prompt


def test_crop_to_mask_bbox_isolates_masked_region():
    image = Image.new("RGB", (100, 100), color=(0, 0, 0))
    mask = Image.new("L", (100, 100), color=0)
    mask.paste(255, (40, 40, 60, 60))  # small white square in the middle

    cropped = crop_to_mask_bbox(image, mask, pad_frac=0.0)
    # Bounding box of a 20x20 region at (40,40)-(60,60): width/height ~20
    assert cropped.size[0] <= 25 and cropped.size[1] <= 25
    assert cropped.size[0] > 0 and cropped.size[1] > 0


def test_crop_to_mask_bbox_falls_back_to_full_image_when_mask_empty():
    image = Image.new("RGB", (50, 50), color=(1, 2, 3))
    empty_mask = Image.new("L", (50, 50), color=0)

    result = crop_to_mask_bbox(image, empty_mask)
    assert result.size == image.size
