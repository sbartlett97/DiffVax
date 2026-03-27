#!/usr/bin/env python3
"""JPEG robustness measurement for DiffVax perturbations.

Social media platforms apply JPEG compression (quality 70-85) to uploaded images.
This script measures how well DiffVax perturbations survive that compression.

Metrics:
  1. Perturbation survival rate (L2 ratio before/after JPEG)
  2. SSIM / PSNR of adversarial image vs original, clean and post-JPEG
  3. Spectral low-frequency energy ratio (lower = perturbation in high-freq bands)
  4. Optional two-checkpoint comparison

Usage:
    python scripts/jpeg_robustness.py \
        --checkpoint checkpoints/diffvax_v3_best.pth \
        --images data/test_images/ \
        --output research/data/jpeg_robustness.json

    # Comparison mode:
    python scripts/jpeg_robustness.py \
        --checkpoint checkpoints/v2_baseline.pth \
        --checkpoint2 checkpoints/diffvax_v3_best.pth \
        --images data/test_images/
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root / "src"))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

JPEG_QUALITIES = [70, 75, 80, 85, 90, 95]

PLATFORM_QUALITY = {
    "Instagram": 85,
    "Twitter/X": 80,
    "Facebook":  85,
    "TikTok":    70,
}

LOW_FREQ_RADIUS = 0.1  # matches SpectralLoss default


# -----------------------------------------------------------------------
# Image utilities
# -----------------------------------------------------------------------

def load_as_tensor(path: Path, size: int | None = None) -> torch.Tensor:
    """Return (1, 3, H, W) float32 tensor in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0) * 2.0 - 1.0


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    return transforms.ToTensor()(img).unsqueeze(0) * 2.0 - 1.0


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = ((t.squeeze(0).cpu().float() + 1.0) / 2.0).clamp(0.0, 1.0)
    return transforms.ToPILImage()(arr)


def jpeg_roundtrip(img_pil: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a.float() - b.float()) ** 2).mean().item()
    return float("inf") if mse == 0.0 else 10.0 * np.log10(4.0 / mse)


def ssim_score(a: torch.Tensor, b: torch.Tensor) -> float:
    try:
        from pytorch_msssim import ssim
        return float(ssim((a.float() + 1) / 2, (b.float() + 1) / 2, data_range=1.0))
    except ImportError:
        # Lightweight fallback: normalised cross-correlation
        af = a.float().flatten()
        bf = b.float().flatten()
        af_c = af - af.mean()
        bf_c = bf - bf.mean()
        return float((af_c @ bf_c) / (af_c.norm() * bf_c.norm()).clamp(min=1e-8))


def low_freq_ratio(delta: torch.Tensor, radius: float = LOW_FREQ_RADIUS) -> float:
    """Fraction of perturbation energy in the low-frequency band."""
    H, W = delta.shape[-2], delta.shape[-1]
    fft = torch.fft.rfft2(delta.float(), norm="ortho")
    mag = fft.abs()

    fw = W // 2 + 1
    fy = torch.arange(H, dtype=torch.float32) / H
    fx = torch.arange(fw, dtype=torch.float32) / fw
    fy_c = torch.min(fy, 1.0 - fy)
    fy_g, fx_g = torch.meshgrid(fy_c, fx, indexing="ij")
    mask = ((fy_g ** 2 + fx_g ** 2).sqrt() < radius).float()
    mask = mask.unsqueeze(0).unsqueeze(0)

    return float((mag * mask).sum() / mag.sum().clamp(min=1e-8))


def survival_ratio(before: torch.Tensor, after: torch.Tensor) -> float:
    return float(after.norm(p=2) / before.norm(p=2).clamp(min=1e-8))


# -----------------------------------------------------------------------
# Core measurement
# -----------------------------------------------------------------------

def measure_checkpoint(
    ckpt_path: Path,
    image_paths: list[Path],
    nb_filter: list[int] | None,
    resolution: int | None,
    label: str = "ckpt",
) -> dict:
    from diffvax.model import NestedUNet

    print(f"\n[{label}] Loading {ckpt_path.name} ...")
    model = NestedUNet(num_classes=3, nb_filter=nb_filter)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    rows: list[dict] = []

    with torch.no_grad():
        for img_path in image_paths:
            orig = load_as_tensor(img_path, size=resolution)
            raw_pert = model(orig.float())
            eps = 32.0 / 255.0 * 2.0
            delta = raw_pert.clamp(-eps, eps)
            adv = (orig + delta).clamp(-1.0, 1.0)
            delta_actual = adv - orig

            row: dict = {
                "name":           img_path.name,
                "psnr_clean":     psnr(orig, adv),
                "ssim_clean":     ssim_score(orig, adv),
                "low_freq_clean": low_freq_ratio(delta_actual),
                "jpeg":           {},
            }

            orig_pil = tensor_to_pil(orig)
            adv_pil  = tensor_to_pil(adv)

            for q in JPEG_QUALITIES:
                adv_q  = pil_to_tensor(jpeg_roundtrip(adv_pil, q))
                orig_q = pil_to_tensor(jpeg_roundtrip(orig_pil, q))
                dq     = adv_q - orig_q

                row["jpeg"][str(q)] = {
                    "psnr":     psnr(orig, adv_q),
                    "ssim":     ssim_score(orig, adv_q),
                    "survival": survival_ratio(delta_actual, dq),
                    "low_freq": low_freq_ratio(dq),
                }

            rows.append(row)
            print(f"  {img_path.name}: PSNR={row['psnr_clean']:.1f}dB  "
                  f"SSIM={row['ssim_clean']:.4f}  LF={row['low_freq_clean']:.3f}")

    def mean_of(key_fn) -> float:
        vals = [key_fn(r) for r in rows]
        return float(np.mean(vals))

    agg: dict = {
        "label":      label,
        "checkpoint": str(ckpt_path),
        "n":          len(rows),
        "psnr_clean": mean_of(lambda r: r["psnr_clean"]),
        "ssim_clean": mean_of(lambda r: r["ssim_clean"]),
        "low_freq":   mean_of(lambda r: r["low_freq_clean"]),
        "jpeg":       {},
    }
    for q in JPEG_QUALITIES:
        sq = str(q)
        psnr_vals     = [r["jpeg"][sq]["psnr"]     for r in rows]
        ssim_vals     = [r["jpeg"][sq]["ssim"]     for r in rows]
        survival_vals = [r["jpeg"][sq]["survival"] for r in rows]
        lf_vals       = [r["jpeg"][sq]["low_freq"] for r in rows]
        agg["jpeg"][sq] = {
            "psnr":     float(np.mean(psnr_vals)),
            "ssim":     float(np.mean(ssim_vals)),
            "survival": float(np.mean(survival_vals)),
            "low_freq": float(np.mean(lf_vals)),
        }

    return {"aggregate": agg, "per_image": rows}


# -----------------------------------------------------------------------
# Pretty-print helpers
# -----------------------------------------------------------------------

def print_summary(agg: dict, title: str) -> None:
    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Checkpoint : {Path(agg['checkpoint']).name}")
    print(f"  N images   : {agg['n']}")
    print(f"  PSNR clean : {agg['psnr_clean']:.2f} dB")
    print(f"  SSIM clean : {agg['ssim_clean']:.4f}")
    print(f"  LowFreq    : {agg['low_freq']:.4f}  (lower => perturbation in high-freq)")
    print()
    print(f"  {'Q':>4}  {'PSNR(dB)':>9}  {'SSIM':>8}  {'Survival':>9}  {'LowFreq':>8}")
    print(f"  {'-'*4}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*8}")
    for q in JPEG_QUALITIES:
        m = agg["jpeg"][str(q)]
        print(f"  {q:>4}  {m['psnr']:>9.2f}  {m['ssim']:>8.4f}  "
              f"{m['survival']:>9.3f}  {m['low_freq']:>8.4f}")
    print()
    print("  Platform targets:")
    for name, q in PLATFORM_QUALITY.items():
        m = agg["jpeg"][str(q)]
        print(f"    {name:<12}  Q={q}  survival={m['survival']:.3f}  SSIM={m['ssim']:.4f}")


def print_delta(a1: dict, a2: dict) -> None:
    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  delta = {Path(a2['checkpoint']).name} minus {Path(a1['checkpoint']).name}")
    print(sep)
    print(f"  PSNR clean : {a2['psnr_clean'] - a1['psnr_clean']:+.2f} dB")
    print(f"  SSIM clean : {a2['ssim_clean'] - a1['ssim_clean']:+.4f}")
    print(f"  LowFreq    : {a2['low_freq'] - a1['low_freq']:+.4f}")
    print()
    print(f"  {'Q':>4}  {'dPSNR':>8}  {'dSSIM':>8}  {'dSurvival':>10}  {'dLowFreq':>9}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*9}")
    for q in JPEG_QUALITIES:
        sq = str(q)
        m1 = a1["jpeg"][sq]
        m2 = a2["jpeg"][sq]
        print(f"  {q:>4}  {m2['psnr']-m1['psnr']:>+8.2f}  "
              f"{m2['ssim']-m1['ssim']:>+8.4f}  "
              f"{m2['survival']-m1['survival']:>+10.3f}  "
              f"{m2['low_freq']-m1['low_freq']:>+9.4f}")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint2", type=Path, default=None,
                        help="Second checkpoint for comparison")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("research/data/jpeg_robustness.json"))
    parser.add_argument("--nb-filter", type=int, nargs=5,
                        metavar=("F0", "F1", "F2", "F3", "F4"),
                        help="NestedUNet filter counts (default: 32 64 128 256 512)")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    image_paths = (sorted(args.images.glob("*.png")) +
                   sorted(args.images.glob("*.jpg")))
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    if not image_paths:
        print(f"No images found in {args.images}", file=sys.stderr)
        sys.exit(1)

    nb_filter = list(args.nb_filter) if args.nb_filter else None
    results: dict = {}

    r1 = measure_checkpoint(
        args.checkpoint, image_paths, nb_filter, args.resolution, label="ckpt1"
    )
    results["ckpt1"] = r1
    print_summary(r1["aggregate"], f"Checkpoint 1: {args.checkpoint.name}")

    if args.checkpoint2:
        r2 = measure_checkpoint(
            args.checkpoint2, image_paths, nb_filter, args.resolution, label="ckpt2"
        )
        results["ckpt2"] = r2
        print_summary(r2["aggregate"], f"Checkpoint 2: {args.checkpoint2.name}")
        print_delta(r1["aggregate"], r2["aggregate"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
