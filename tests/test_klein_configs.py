"""The shipped Klein edit-mode configs must parse, pass through to the
trainer, and build the right surrogate spec without loading any weights."""
import os
import sys

import pytest
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from train import build_immunization_config, _build_surrogate_specs  # noqa: E402

CONFIGS = ["klein_edit_512.yml", "klein_edit_1088.yml"]


@pytest.mark.parametrize("name", CONFIGS)
def test_config_parses_and_is_consistent(name):
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", name)))
    assert cfg["epsilon"] == 16
    assert cfg["flux_model_link"] == "black-forest-labs/FLUX.2-klein-4B"
    assert cfg["flux_attack"]["mode"] == "edit"
    specs = _build_surrogate_specs(cfg)
    assert [s[0] for s in specs] == ["flux"]
    assert abs(sum(s[1] for s in specs) - 1.0) < 1e-9
    # dead terms stay off by default
    for k in ("clip_loss", "attention_loss", "flat_minima", "noise_target", "adaptive_ensemble"):
        assert not cfg[k]["enabled"], k
    passed = build_immunization_config(cfg)
    assert passed["epsilon"] == 16
    assert passed["hidden_state_loss"]["enabled"]
    assert passed["latent_loss"]["mode"] == "l2"


def test_1088_curriculum_is_monotonic_and_reaches_1088():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "klein_edit_1088.yml")))
    st = cfg["curriculum"]["stages"]
    assert [s["resolution"] for s in st] == sorted(s["resolution"] for s in st)
    ends = [s["until_epoch"] for s in st]
    assert ends == sorted(ends) and len(set(ends)) == len(ends)
    assert st[-1]["resolution"] == 1088
    assert ends[-1] == cfg["iter_num"]          # the last stage actually runs
    assert all(r % 16 == 0 for r in (s["resolution"] for s in st))


def test_v2_config_enables_the_new_terms():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "klein_edit_v2_512.yml")))
    p = build_immunization_config(cfg)
    assert p["loss1_weight"] == 0.0 and p["lpips_loss"]["enabled"]
    assert cfg["clip_loss"]["enabled"] and cfg["clip_loss"]["feature_weight"] == 0.0
    assert cfg["clip_loss"]["model"] != "ViT-B-32"       # evaluation uses B-32; do not train on it
