#!/usr/bin/env python3
"""Upper-bound test: can ANY imperceptible per-image perturbation break Klein edits?

For each validation image and each config (objective x epsilon x EoT), optimise a
perturbation directly with PGD (see src/diffvax/pgd.py), then edit clean,
protected and matched-PSNR-noise versions with the REAL Flux2KleinPipeline (same
seed), both as-is and after the social-media transform (resize 1080 + JPEG q80).
Reports edit divergence vs the clean edit (LPIPS), CLIP prompt-alignment drop,
and perturbation visibility, and posts the table plus sample grids to Discord.

    python scripts/pgd_upper_bound.py --images 6 --eps 16 8 --objectives encoder e2e --eot off on
"""
import argparse
import itertools
import json
import os
import sys
import time

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "scripts"))

import numpy as np
import torch
from PIL import Image

from diffvax import protocol as P
from diffvax.eot import DifferentiableEoT
from diffvax.flux_edit_attack import FluxKleinEditAttack
from diffvax.pgd import make_objective, pgd
from diffvax.utils import resolve_device, resolve_dtype


def to_t(pil, device):
    a = torch.from_numpy(np.asarray(pil.convert("RGB"), np.float32) / 127.5 - 1.0)
    return a.permute(2, 0, 1)[None].to(device)


def to_pil(t):
    a = ((t[0].float().clamp(-1, 1) / 2 + 0.5) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(a)


def post(url, text, path=None):
    if not url:
        return
    try:
        import requests
        if path and os.path.exists(path):
            with open(path, "rb") as fh:
                requests.post(url, data={"payload_json": json.dumps({"content": text[:1900]})},
                              files={"file": (os.path.basename(path), fh, "image/png")}, timeout=60)
        else:
            requests.post(url, json={"content": text[:1900]}, timeout=15)
    except Exception as exc:  # type only: messages can embed the webhook URL
        print("discord post failed:", type(exc).__name__, file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--data-dir", default=os.path.join(_root, "data"))
    ap.add_argument("--images", type=int, default=6)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--eps", type=float, nargs="+", default=[16, 8])
    ap.add_argument("--objectives", nargs="+", default=["encoder", "e2e"])
    ap.add_argument("--eot", nargs="+", default=["off", "on"])
    ap.add_argument("--steps-encoder", type=int, default=150)
    ap.add_argument("--steps-e2e", type=int, default=60)
    ap.add_argument("--out", default="/workspace/outputs/pgd_upper_bound")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    url = os.environ.get("DISCORD_WEBHOOK")
    device = resolve_device(); dtype = resolve_dtype(device)

    from diffusers import Flux2KleinPipeline
    import lpips
    from diffvax.metrics import MetricType, create_metric
    from skimage.metrics import structural_similarity as ssim

    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False); pipe.transformer.requires_grad_(False)
    atk = FluxKleinEditAttack.__new__(FluxKleinEditAttack)
    atk.pipe, atk.model_link = pipe, args.model
    atk._init_common(0.5, True, 1.0, 1024 * 1024, False)
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    clip = create_metric(MetricType.CLIP, model="ViT-B-32", pretrained_on="laion2b_s34b_b79k")
    eot = DifferentiableEoT({"eot": {"enabled": True, "jpeg_quality_range": [70, 90],
                                     "resize_range": [0.75, 1.0], "p_jpeg": 1.0, "p_resize": 0.5,
                                     "p_blur": 0.0, "p_noise": 0.0}})

    def lpips_d(a, b):
        with torch.no_grad():
            return float(lp(to_t(a, device), to_t(b.resize(a.size, Image.LANCZOS), device)).item())

    def edit(pil, prompt):
        with torch.no_grad():
            return pipe(image=pil, prompt=prompt, num_inference_steps=4, guidance_scale=1.0,
                        generator=torch.Generator("cpu").manual_seed(0)).images[0]

    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "validation", "metadata.jsonl")) if l.strip()]
    rows = rows[: args.images]
    items = []
    for r in rows:
        img = Image.open(os.path.join(args.data_dir, "validation", r["file_name"])).convert("RGB")
        img = img.resize((args.resolution, args.resolution), Image.LANCZOS)
        prompt = r["flux_prompts"][0]
        items.append({"name": os.path.splitext(os.path.basename(r["file_name"]))[0], "img": img,
                      "prompt": prompt, "clean_edit": edit(img, prompt),
                      "clean_edit_soc": edit(P.social_media_transform(img), prompt)})

    rng = np.random.default_rng(0)
    results = {}
    post(url, f"**PGD upper bound** started: {len(items)} images, eps {args.eps}, "
              f"objectives {args.objectives}, EoT {args.eot}")
    for obj_name, eps, eot_flag in itertools.product(args.objectives, args.eps, args.eot):
        tag = f"{obj_name}_eps{int(eps)}_eot{eot_flag}"
        obj = make_objective(atk, obj_name, num_inference_steps=4)
        steps = args.steps_encoder if obj_name == "encoder" else args.steps_e2e
        per = []
        samples = []
        t0 = time.time()
        for it in items:
            x = to_t(it["img"], device)
            adv = pgd(x, obj, [it["prompt"]], eps, steps=steps, eot=eot if eot_flag == "on" else None)
            prot = to_pil(adv)
            noise = P.gaussian_noise_matched_psnr(it["img"], prot, rng)
            e_prot, e_noise = edit(prot, it["prompt"]), edit(noise, it["prompt"])
            e_prot_s = edit(P.social_media_transform(prot), it["prompt"])
            e_noise_s = edit(P.social_media_transform(noise), it["prompt"])
            c0 = float(clip([it["clean_edit"]], [it["prompt"]])[0])
            per.append({
                "image": it["name"],
                "psnr": P.psnr_u8(it["img"], prot), "linf": P.linf_u8(it["img"], prot),
                "ssim": float(ssim(np.asarray(it["img"]), np.asarray(prot), channel_axis=2, data_range=255)),
                "lpips_pert": lpips_d(prot, it["img"]),
                "d_prot": lpips_d(e_prot, it["clean_edit"]), "d_noise": lpips_d(e_noise, it["clean_edit"]),
                "d_prot_soc": lpips_d(e_prot_s, it["clean_edit_soc"]),
                "d_noise_soc": lpips_d(e_noise_s, it["clean_edit_soc"]),
                "clip_drop": c0 - float(clip([e_prot], [it["prompt"]])[0]),
                "clip_drop_soc": float(clip([it["clean_edit_soc"]], [it["prompt"]])[0])
                                  - float(clip([e_prot_s], [it["prompt"]])[0]),
            })
            if len(samples) < 3:
                samples.append({"name": it["name"], "prompt": it["prompt"],
                                "conditions": {"clean": it["img"], "protected": prot, "noise_psnr": noise},
                                "edits": {"clean": it["clean_edit"], "protected": e_prot, "noise_psnr": e_noise}})
        def ci(key):
            return P.bootstrap_ci({p["image"]: [p[key]] for p in per}, n_boot=1000)
        ex = P.bootstrap_ci({p["image"]: [p["d_prot"] - p["d_noise"]] for p in per}, n_boot=1000)
        ex_s = P.bootstrap_ci({p["image"]: [p["d_prot_soc"] - p["d_noise_soc"]] for p in per}, n_boot=1000)
        summ = {"per_image": per, "excess": ex, "excess_social": ex_s,
                **{k: ci(k) for k in ("d_prot", "d_noise", "d_prot_soc", "clip_drop", "clip_drop_soc",
                                      "psnr", "ssim", "lpips_pert")},
                "seconds": time.time() - t0}
        results[tag] = summ
        grid = os.path.join(args.out, f"grid_{tag}.png")
        P.make_sample_grid(samples).save(grid)
        f = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]"
        line = (f"**PGD {tag}** | excess vs noise {f(ex)} | after JPEG {f(ex_s)} | "
                f"CLIP drop {summ['clip_drop'][0]:+.2f} (JPEG {summ['clip_drop_soc'][0]:+.2f}) | "
                f"PSNR {summ['psnr'][0]:.1f} SSIM {summ['ssim'][0]:.3f} LPIPS {summ['lpips_pert'][0]:.3f} | "
                f"{summ['seconds']/60:.1f} min")
        print(line, flush=True)
        post(url, line, grid)
        json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1, default=float)
    post(url, "@here **PGD upper bound finished**")


if __name__ == "__main__":
    main()
