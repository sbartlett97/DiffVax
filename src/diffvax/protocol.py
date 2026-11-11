"""Evaluation protocol for image-immunization claims (pure, model-agnostic).

Why this exists: the earlier eval reports (n=1-3, no controls) could not
distinguish protection from "any perturbation of this size changes a diffusion
edit". This module defines the measurement that can:

  * every image is passed through a social-media transform (resize + JPEG)
    BEFORE editing, for the clean, protected and control conditions alike;
  * controls: Gaussian noise at MATCHED PSNR, and random-sign noise at the SAME
    L-inf budget -- the protected image must beat these, not just the clean
    edit;
  * metric: edit divergence d(c) = LPIPS(edit(condition), edit(clean)), computed
    with identical seeds. Protection = d(protected) - d(noise control) > 0,
    paired per (image, prompt, seed);
  * uncertainty: bootstrap confidence intervals resampling IMAGES (the
    independent unit; prompts/seeds within an image are correlated).

Models and metrics are injected, so the statistics are unit-testable on CPU.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Image-space helpers
# ---------------------------------------------------------------------------

def social_media_transform(
    img: Image.Image, max_side: int = 1080, quality: int = 80
) -> Image.Image:
    """Typical upload pipeline: downscale so the long side <= max_side
    (LANCZOS, aspect preserved), then JPEG at ``quality``."""
    img = img.convert("RGB")
    w, h = img.size
    long_side = max(w, h)
    if long_side > max_side:
        s = max_side / long_side
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def psnr_u8(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB"), dtype=np.float64)
    y = np.asarray(b.convert("RGB"), dtype=np.float64)
    mse = float(np.mean((x - y) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def linf_u8(a: Image.Image, b: Image.Image) -> int:
    x = np.asarray(a.convert("RGB"), dtype=np.int32)
    y = np.asarray(b.convert("RGB"), dtype=np.int32)
    return int(np.abs(x - y).max())


def gaussian_noise_matched_psnr(
    orig: Image.Image, protected: Image.Image, rng: np.random.Generator
) -> Image.Image:
    """Gaussian noise whose MSE vs ``orig`` matches the protected image's.

    sigma = sqrt(MSE) in uint8 units. Clipping to [0,255] lowers the realised
    MSE slightly for bright/dark pixels, so sigma is calibrated by a short
    fixed-point iteration on the realised PSNR.
    """
    x = np.asarray(orig.convert("RGB"), dtype=np.float64)
    target = float(np.mean((np.asarray(protected.convert("RGB"), dtype=np.float64) - x) ** 2))
    if target == 0:
        return orig.convert("RGB")
    sigma = math.sqrt(target)
    z = rng.standard_normal(x.shape)
    for _ in range(8):
        y = np.clip(np.round(x + sigma * z), 0, 255)
        mse = float(np.mean((y - x) ** 2))
        if mse == 0 or abs(mse - target) / target < 0.01:
            break
        sigma *= math.sqrt(target / mse)
    return Image.fromarray(np.clip(np.round(x + sigma * z), 0, 255).astype(np.uint8))


def random_sign_noise(
    orig: Image.Image, eps: int, rng: np.random.Generator
) -> Image.Image:
    """+/-eps random-sign noise: same L-inf budget as the protected image and
    (for the saturating tanh bound) a comparable or larger energy."""
    x = np.asarray(orig.convert("RGB"), dtype=np.int32)
    s = rng.choice([-1, 1], size=x.shape)
    return Image.fromarray(np.clip(x + eps * s, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(
    groups: Dict[str, List[float]],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
):
    """Cluster bootstrap: resample GROUPS (images) with replacement, pooling
    each group's observations. Returns (point, lo, hi)."""
    keys = list(groups)
    if not keys:
        return float("nan"), float("nan"), float("nan")
    arrs = [np.asarray(groups[k], dtype=np.float64) for k in keys]
    point = float(stat(np.concatenate(arrs)))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(arrs), len(arrs))
        boots[i] = stat(np.concatenate([arrs[j] for j in idx]))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


@dataclass
class Record:
    editor: str
    image: str
    prompt: str
    seed: int
    d: Dict[str, float]            # condition -> edit divergence vs clean edit
    clip: Dict[str, float] = field(default_factory=dict)  # condition -> CLIP(edit,prompt)


def summarize(
    records: Sequence[Record],
    protected: str = "protected",
    controls: Sequence[str] = ("noise_psnr", "noise_linf"),
    margin: float = 0.05,
    n_boot: int = 2000,
) -> Dict[str, dict]:
    """Per-editor protection statistics.

    excess_<c> = d(protected) - d(c), paired per record.
    success_<c> = fraction of records with excess_<c> > margin.
    Both with image-level cluster-bootstrap 95% CIs.
    """
    out: Dict[str, dict] = {}
    for editor in sorted({r.editor for r in records}):
        rs = [r for r in records if r.editor == editor]
        s: dict = {"n_records": len(rs), "n_images": len({r.image for r in rs})}
        for cond in {c for r in rs for c in r.d}:
            g: Dict[str, List[float]] = {}
            for r in rs:
                if cond in r.d:
                    g.setdefault(r.image, []).append(r.d[cond])
            s[f"d_{cond}"] = bootstrap_ci(g, n_boot=n_boot)
        for c in controls:
            ex: Dict[str, List[float]] = {}
            for r in rs:
                if protected in r.d and c in r.d:
                    ex.setdefault(r.image, []).append(r.d[protected] - r.d[c])
            if not ex:
                continue
            s[f"excess_{c}"] = bootstrap_ci(ex, n_boot=n_boot)
            succ = {k: [float(v > margin) for v in vs] for k, vs in ex.items()}
            s[f"success_{c}"] = bootstrap_ci(succ, n_boot=n_boot)
        cd: Dict[str, List[float]] = {}
        for r in rs:
            if protected in r.clip and "clean" in r.clip:
                cd.setdefault(r.image, []).append(r.clip["clean"] - r.clip[protected])
        if cd:
            s["clip_delta"] = bootstrap_ci(cd, n_boot=n_boot)
        out[editor] = s
    return out


def verdict(summary_for_editor: dict, control: str = "noise_psnr") -> str:
    """'PROTECTED' only if the CI lower bound of the excess over the control is
    > 0; 'NO EFFECT' if the CI straddles 0; 'WORSE THAN CONTROL' if the upper
    bound is < 0.

    The PRIMARY control is the matched-PSNR Gaussian noise (like-for-like
    energy). Saturated +/-eps sign noise is much louder (16 levels everywhere
    is ~24 dB) so it is reported as a stricter SECONDARY verdict, not the
    headline."""
    key = f"excess_{control}"
    if key not in summary_for_editor:
        return "UNMEASURED"
    _, lo, hi = summary_for_editor[key]
    if lo > 0:
        return "PROTECTED"
    if hi < 0:
        return "WORSE THAN CONTROL"
    return "NO EFFECT"


# ---------------------------------------------------------------------------
# Orchestration (models injected)
# ---------------------------------------------------------------------------

def build_conditions(
    original: Image.Image,
    protected: Image.Image,
    eps_pixels: Optional[int],
    rng: np.random.Generator,
    transform: Callable[[Image.Image], Image.Image],
    extra: Optional[Dict[str, Callable[[Image.Image], Image.Image]]] = None,
) -> Dict[str, Image.Image]:
    """All conditions that get edited, each AFTER the social-media transform.

    ``eps_pixels=None`` uses the MEASURED L-inf of this protected image for the
    random-sign control (legacy checkpoints have no declared budget)."""
    if eps_pixels is None:
        eps_pixels = max(1, linf_u8(original, protected))
    conds = {
        "clean": original,
        "protected": protected,
        "noise_psnr": gaussian_noise_matched_psnr(original, protected, rng),
        "noise_linf": random_sign_noise(original, eps_pixels, rng),
    }
    for name, fn in (extra or {}).items():
        conds[name] = fn(original)
    return {k: transform(v) for k, v in conds.items()}


def run_editor(
    editor_name: str,
    edit_fn: Callable[[Image.Image, Optional[Image.Image], str, int], Image.Image],
    items: Iterable[dict],
    seeds: Sequence[int],
    divergence: Callable[[Image.Image, Image.Image], float],
    clip_score: Optional[Callable[[Image.Image, str], float]] = None,
) -> List[Record]:
    """``items``: dicts with keys name, mask, prompts, conditions (from
    build_conditions). Edits clean once per (prompt, seed) and each condition
    with the SAME seed."""
    records: List[Record] = []
    for it in items:
        for prompt in it["prompts"]:
            for seed in seeds:
                edits = {
                    c: edit_fn(img, it.get("mask"), prompt, seed)
                    for c, img in it["conditions"].items()
                }
                clean = edits["clean"]
                d = {c: divergence(e, clean) for c, e in edits.items() if c != "clean"}
                clip = {c: clip_score(e, prompt) for c, e in edits.items()} if clip_score else {}
                records.append(Record(editor_name, it["name"], prompt, seed, d, clip))
    return records


def to_markdown(summary: Dict[str, dict], perturbation: dict, margin: float) -> str:
    f = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
    lines = [
        "| Editor | n imgs | d(protected) | d(noise PSNR-matched) | excess vs PSNR-matched "
        f"| excess vs same-L-inf | success(>{margin}) | CLIP delta | verdict (PSNR) | verdict (L-inf) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ed, s in summary.items():
        lines.append(
            f"| {ed} | {s['n_images']} | {f(s['d_protected'])} | {f(s.get('d_noise_psnr', (float('nan'),)*3))} "
            f"| {f(s.get('excess_noise_psnr', (float('nan'),)*3))} "
            f"| {f(s.get('excess_noise_linf', (float('nan'),)*3))} "
            f"| {f(s.get('success_noise_psnr', (float('nan'),)*3))} "
            f"| {f(s.get('clip_delta', (float('nan'),)*3))} | **{verdict(s)}** | {verdict(s, 'noise_linf')} |"
        )
    lines += [
        "",
        "Perturbation (before transform): "
        + ", ".join(f"{k}={v:.3f}" for k, v in perturbation.items()),
    ]
    return "\n".join(lines)
