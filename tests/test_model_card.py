"""Unit tests for DiffVaxImmunization._model_card_kwargs() and the Hub model
card template it feeds (model.py::_MODEL_CARD_TEMPLATE).

Covers _model_card_kwargs() in isolation (bypassing DiffVaxImmunization's
heavy __init__ — same pattern as test_h7_target_image.py) against a range of
config shapes, and a template-rendering smoke test via save_pretrained() to
a local temp dir (no network/token needed — push_to_hub() is untestable
without real Hub access, but the card is built identically either way).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffvax.model import NestedUNet  # noqa: E402
from diffvax.immunization.diffvax_immunization import DiffVaxImmunization  # noqa: E402


def _make_immun(config):
    immun = DiffVaxImmunization.__new__(DiffVaxImmunization)
    immun._config = config
    return immun


def test_model_card_kwargs_v1_baseline_config():
    """A v1-equivalent config (every optional phase off) should report
    'none configured' / 'none (v1 baseline...)' rather than crashing on
    missing optional sections."""
    config = {
        "project_name": "diffvax-v1",
        "attack_model_link": "runwayml/stable-diffusion-inpainting",
        "sd_probability": 1.0,
        "resolution": 512,
        "alpha": 4,
        "beta": 0.5,
        "learning_rate": 1e-5,
        "num_inference_steps": 4,
    }
    kwargs = _make_immun(config)._model_card_kwargs("final", 100, 0.5)

    assert "SD 1.5 inpainting" in kwargs["surrogates"]
    assert kwargs["resolution_info"] == "512px (static)"
    assert kwargs["loss_terms"] == "none (v1 baseline: loss1 + loss2 only)"
    assert kwargs["checkpoint_type"] == "final"
    assert kwargs["epoch"] == 100
    assert kwargs["loss_value"] == "0.50000"


def test_model_card_kwargs_multi_surrogate_with_curriculum():
    config = {
        "project_name": "diffvax-full",
        "attack_model_link": "runwayml/stable-diffusion-inpainting",
        "sd_probability": 0.5,
        "sd3_model_link": "stabilityai/stable-diffusion-3.5-medium",
        "sd3_probability": 0.3,
        "flux_model_link": "black-forest-labs/FLUX.2-klein-9B",
        "flux_probability": 0.2,
        "resolution": 512,
        "alpha": 4,
        "beta": 0.5,
        "learning_rate": 1e-5,
        "num_inference_steps": 4,
        "curriculum": {
            "enabled": True,
            "stages": [
                {"resolution": 512, "until_epoch": 300},
                {"resolution": 1024, "until_epoch": 600},
            ],
        },
        "clip_loss": {"enabled": True},
        "sd3_attack": {"masked_attack_probability": 0.5},
    }
    kwargs = _make_immun(config)._model_card_kwargs("best", 42, 1.23456)

    assert "SD 1.5 inpainting" in kwargs["surrogates"]
    assert "SD3/3.5" in kwargs["surrogates"]
    assert "FLUX" in kwargs["surrogates"]
    assert kwargs["resolution_info"] == "512px (until epoch 300) → 1024px (until epoch 600)"
    assert "CLIP disruption" in kwargs["loss_terms"]
    assert "masked/inpainting-style RePaint attack (p=0.5)" in kwargs["loss_terms"]


def test_model_card_kwargs_zero_probability_surrogate_excluded():
    """A surrogate with probability 0 (even if its model_link is set) must
    not be listed as trained-against."""
    config = {
        "attack_model_link": "runwayml/stable-diffusion-inpainting",
        "sd_probability": 0.0,
        "sd3_model_link": "stabilityai/stable-diffusion-3.5-medium",
        "sd3_probability": 1.0,
        "flux_model_link": None,
        "flux_probability": 0.0,
        "resolution": 512,
        "alpha": 4,
        "beta": 0.5,
        "learning_rate": 1e-5,
        "num_inference_steps": 4,
    }
    kwargs = _make_immun(config)._model_card_kwargs("periodic", 10, 2.0)
    assert "SD 1.5" not in kwargs["surrogates"]
    assert "FLUX" not in kwargs["surrogates"]
    assert "SD3/3.5" in kwargs["surrogates"]


def test_model_card_renders_via_save_pretrained(tmp_path):
    """End-to-end smoke test: real NestedUNet + real _model_card_kwargs()
    output renders into a README.md with no missing/unrendered {{ }}
    placeholders. No network access — save_pretrained() writes locally."""
    config = {
        "project_name": "diffvax-test",
        "attack_model_link": "runwayml/stable-diffusion-inpainting",
        "sd_probability": 1.0,
        "resolution": 512,
        "alpha": 4,
        "beta": 0.5,
        "learning_rate": 1e-5,
        "num_inference_steps": 4,
    }
    kwargs = _make_immun(config)._model_card_kwargs("final", 5, 0.9)

    net = NestedUNet(num_classes=3, nb_filter=[4, 8, 16, 32, 64])
    out_dir = tmp_path / "card_out"
    net.save_pretrained(str(out_dir), model_card_kwargs=kwargs)

    card = (out_dir / "README.md").read_text()
    assert "{{" not in card and "}}" not in card, "unrendered template placeholder in model card"
    assert "SD 1.5 inpainting" in card
    assert "final" in card
    assert "0.90000" in card
