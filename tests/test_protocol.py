"""Statistics and controls of the evaluation protocol (no models needed)."""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffvax import protocol as P  # noqa: E402


def _img(seed=0, n=64):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(30, 220, (n, n, 3), dtype=np.uint8))


def test_social_transform_downsizes_and_jpegs():
    big = _img(0, 1088)
    out = P.social_media_transform(big, max_side=1080, quality=80)
    assert out.size == (1080, 1080)
    small = P.social_media_transform(_img(1, 64), 1080, 80)
    assert small.size == (64, 64)
    assert P.psnr_u8(_img(1, 64), small) < 60   # lossy


def test_noise_controls_match_budget_and_psnr():
    rng = np.random.default_rng(0)
    orig = _img(0, 128)
    arr = np.asarray(orig, dtype=np.int32)
    prot = Image.fromarray(np.clip(arr + np.random.default_rng(1).integers(-10, 11, arr.shape), 0, 255).astype(np.uint8))
    n = P.gaussian_noise_matched_psnr(orig, prot, rng)
    assert abs(P.psnr_u8(orig, n) - P.psnr_u8(orig, prot)) < 0.3
    s = P.random_sign_noise(orig, 16, rng)
    assert P.linf_u8(orig, s) == 16


def test_bootstrap_ci_covers_mean_and_detects_zero():
    rng = np.random.default_rng(0)
    g = {f"i{i}": list(rng.normal(0.2, 0.05, 3)) for i in range(40)}
    p, lo, hi = P.bootstrap_ci(g, n_boot=500)
    assert lo < p < hi and lo > 0
    z = {f"i{i}": list(rng.normal(0.0, 0.05, 3)) for i in range(40)}
    _, lo, hi = P.bootstrap_ci(z, n_boot=500)
    assert lo < 0 < hi


def _records(shift, n=30):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        base = rng.uniform(0.2, 0.4)
        out.append(P.Record("ed", f"img{i}", "p", 0,
                            {"protected": base + shift + rng.normal(0, .02),
                             "noise_linf": base, "noise_psnr": base},
                            {"clean": 30.0, "protected": 30.0 - shift * 10}))
    return out


def test_verdicts():
    assert P.verdict(P.summarize(_records(0.15), n_boot=300)["ed"]) == "PROTECTED"
    assert P.verdict(P.summarize(_records(0.0), n_boot=300)["ed"]) == "NO EFFECT"
    assert P.verdict(P.summarize(_records(-0.15), n_boot=300)["ed"]) == "WORSE THAN CONTROL"


def test_run_editor_uses_same_seed_and_clean_baseline():
    calls = []

    def edit(img, mask, prompt, seed):
        calls.append(seed)
        return img

    cond = {"clean": _img(0), "protected": _img(1), "noise_psnr": _img(2), "noise_linf": _img(3)}
    items = [{"name": "a", "mask": None, "prompts": ["p1", "p2"], "conditions": cond}]
    div = lambda a, b: float(np.abs(np.asarray(a, float) - np.asarray(b, float)).mean() / 255)
    recs = P.run_editor("ed", edit, items, seeds=[0, 1], divergence=div)
    assert len(recs) == 4                          # 2 prompts x 2 seeds
    assert set(recs[0].d) == {"protected", "noise_psnr", "noise_linf"}
    assert set(calls) == {0, 1}
    md = P.to_markdown(P.summarize(recs, n_boot=100), {"psnr": 40.0}, 0.05)
    assert "| ed |" in md
