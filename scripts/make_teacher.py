#!/usr/bin/env python3
"""Per-image teacher perturbations: VAE-encoder attack under an LPIPS limit.

For every image (validation first, then train) at --resolution, run PGD on
    ||E(T(x+d)) - E(gray)||^2 / ||E(x) - E(gray)||^2 + w * relu(LPIPS(x, x+d) - target)
with JPEG/resize EoT T and ||d||_inf <= epsilon. Saves the protected image as a
PNG (so scripts/eval_protocol.py --protected-dir can evaluate it with the real
editor) and the perturbation as a .pt (for distilling a one-shot generator).

    python scripts/make_teacher.py --out /workspace/outputs/teacher_512
"""
import argparse
import glob
import json
import os
import sys
import time

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffvax.eot import DifferentiableEoT
from diffvax.pgd import pgd


def post(url, text):
    if not url:
        return
    try:
        import requests
        requests.post(url, json={"content": text[:1900]}, timeout=15)
    except Exception as exc:
        print("discord post failed:", type(exc).__name__, file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(_root, "data"))
    ap.add_argument("--vae", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--epsilon", type=float, default=16)
    ap.add_argument("--lpips-target", type=float, default=0.05)
    ap.add_argument("--lpips-weight", type=float, default=20.0)
    ap.add_argument("--ssim-target", type=float, default=None,
                    help="hinge: penalise SSIM(x, x+d) BELOW this (e.g. 0.95). LPIPS<=0.05 alone "
                         "still left a visible 8px mesh (SSIM 0.62)")
    ap.add_argument("--ssim-weight", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--step-frac", type=float, default=0.05)
    ap.add_argument("--splits", nargs="+", default=["validation", "train"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    url = os.environ.get("DISCORD_WEBHOOK")

    import lpips
    from diffusers import AutoencoderKLFlux2
    dev = "cuda"
    vae = AutoencoderKLFlux2.from_pretrained(args.vae, subfolder="vae", torch_dtype=torch.bfloat16)
    vae = vae.to(dev).eval().requires_grad_(False)
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval().requires_grad_(False)
    eot = DifferentiableEoT({"eot": {"enabled": True, "jpeg_quality_range": [70, 90],
                                     "resize_range": [0.75, 1.0], "p_jpeg": 1.0, "p_resize": 0.5,
                                     "p_blur": 0.0, "p_noise": 0.0}})
    enc = lambda x: vae.encode(x.to(torch.bfloat16)).latent_dist.mode().float()
    gray = {}

    post(url, f"**teacher** started: eps {args.epsilon}/255, LPIPS target {args.lpips_target}, "
              f"{args.steps} steps, splits {args.splits}")
    for split in args.splits:
        files = sorted(glob.glob(os.path.join(args.data_dir, split, "images", "*.png")))
        img_dir = os.path.join(args.out, split, "images"); os.makedirs(img_dir, exist_ok=True)
        d_dir = os.path.join(args.out, split, "deltas"); os.makedirs(d_dir, exist_ok=True)
        log = open(os.path.join(args.out, split, "stats.jsonl"), "a")
        t0 = time.time()
        for k, f in enumerate(files):
            name = os.path.splitext(os.path.basename(f))[0]
            if os.path.exists(os.path.join(d_dir, name + ".pt")):
                continue
            img = Image.open(f).convert("RGB").resize((args.resolution, args.resolution), Image.LANCZOS)
            x = torch.from_numpy(np.asarray(img, np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(dev)
            if x.shape not in gray:
                with torch.no_grad():
                    gray[x.shape] = enc(torch.zeros_like(x))
            zg = gray[x.shape]
            with torch.no_grad():
                base = F.mse_loss(enc(x), zg).clamp(min=1e-8)
            state = {}

            def eot_wrap(xa):
                state["xa"] = xa
                return eot(xa)

            from kornia.metrics import ssim as kssim

            def ssim01(a, b):
                return kssim((a + 1) / 2, (b + 1) / 2, 11).mean()

            def obj(xa, _p):
                loss = F.mse_loss(enc(xa), zg) / base
                if args.lpips_weight > 0:
                    loss = loss + args.lpips_weight * F.relu(lp(x, state["xa"]).mean() - args.lpips_target)
                if args.ssim_target is not None:
                    loss = loss + args.ssim_weight * F.relu(args.ssim_target - ssim01(x, state["xa"]))
                return loss

            adv = pgd(x, obj, None, args.epsilon, steps=args.steps, step_frac=args.step_frac, eot=eot_wrap)
            with torch.no_grad():
                e = float(np.mean([(F.mse_loss(enc(eot(adv)), zg) / base).item() for _ in range(3)]))
                l = float(lp(x, adv).item())
                from kornia.metrics import ssim as kssim
                s_ = float(kssim((x + 1) / 2, (adv + 1) / 2, 11).mean().item())
            arr = ((adv[0].clamp(-1, 1) / 2 + 0.5) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(arr).save(os.path.join(img_dir, name + ".png"))
            torch.save((adv - x).half().cpu(), os.path.join(d_dir, name + ".pt"))
            log.write(json.dumps({"name": name, "enc": e, "lpips": l, "ssim": s_}) + "\n"); log.flush()
            if (k + 1) % 20 == 0 or k + 1 == len(files):
                msg = (f"**teacher** {split} {k + 1}/{len(files)} | last enc {e:.3f} LPIPS {l:.3f} SSIM {s_:.3f} | "
                       f"{(time.time() - t0) / (k + 1):.1f} s/img")
                print(msg, flush=True); post(url, msg)
        post(url, f"**teacher** {split} done")
    post(url, "@here **teacher** all splits done")


if __name__ == "__main__":
    main()
