#!/usr/bin/env python3
"""Real-weight parity: FluxKleinEditAttack.attack() vs the REAL Flux2KleinPipeline.

The CPU test uses tiny random weights; this exercises what that cannot -- bf16
casts, the real Qwen3 text path, the real distilled flag and scheduler config.
Run it ONCE on the GPU before training (runbook stage 1) and read the numbers:

    python scripts/parity_check_klein.py --sizes 512 1088

512px: no resize happens, so the difference is pure numerics (expect mean abs
diff ~1e-3 in bf16). 1088px: the pipeline shrinks the reference to ~1024px
(LANCZOS) and returns 1024px; ours uses bicubic-antialias and returns 1088px
(resized back), so only an approximate match is expected -- the point is that it
is the SAME edit, not a different image.
"""
import argparse
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffvax.flux_edit_attack import FluxKleinEditAttack
from diffvax.utils import resolve_device, resolve_dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--image", default=None, help="default: first data/validation image")
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1088])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--prompt", default="a person in a cafe, golden hour")
    ap.add_argument("--tol512", type=float, default=0.01, help="max mean abs diff at no-resize sizes")
    args = ap.parse_args()

    from diffusers import Flux2KleinPipeline
    device, dtype = resolve_device(), None
    dtype = resolve_dtype(device)
    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
    print(f"is_distilled={pipe.config.is_distilled}  dtype={dtype}  device={device}")

    atk = FluxKleinEditAttack.__new__(FluxKleinEditAttack)
    atk.pipe, atk.model_link = pipe, args.model
    atk._init_common(1.0, False, 1.0, 1024 * 1024, False)
    pipe.vae.requires_grad_(False); pipe.transformer.requires_grad_(False)

    path = args.image or os.path.join(
        _root, "data", "validation", "images",
        sorted(os.listdir(os.path.join(_root, "data", "validation", "images")))[0])
    base = Image.open(path).convert("RGB")
    ok = True
    for size in args.sizes:
        pil = base.resize((size, size), Image.LANCZOS)
        t = (torch.from_numpy(np.asarray(pil)).permute(2, 0, 1)[None].float() / 127.5 - 1.0).to(device)
        ref_h, ref_w = atk._reference_size(size, size)
        gh, gw = ref_h // 16, ref_w // 16
        noise = torch.randn(1, 128, gh, gw, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            ref = pipe(image=pil, prompt=args.prompt, num_inference_steps=args.steps,
                       guidance_scale=1.0, latents=noise.clone().to(device, dtype),
                       output_type="pt", generator=torch.Generator(device="cpu").manual_seed(0)).images
            out = atk.attack(prompt=[args.prompt], image=t, num_inference_steps=args.steps,
                             latents=noise.clone().to(device, dtype))
        mine = (out.float() / 2 + 0.5).clamp(0, 1)
        ref = ref.float().to(mine.device)
        if mine.shape != ref.shape:
            ref = F.interpolate(ref, mine.shape[-2:], mode="bicubic", align_corners=False)
        d = (mine - ref).abs()
        print(f"size {size}: ref-grid {gh}x{gw}  mean|diff|={d.mean():.5f}  max|diff|={d.max():.4f}")
        if size * size <= 1024 * 1024 and d.mean() > args.tol512:
            ok = False
    print("PARITY OK" if ok else "PARITY FAILED -- do not train until explained")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
