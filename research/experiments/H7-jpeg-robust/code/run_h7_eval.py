#!/usr/bin/env python3
"""H7 JPEG-robust immunization evaluation.

Tests H7 (STE JPEG-augmented training) vs sd15_only baseline at multiple
JPEG quality levels. Checks whether H7 further amplifies the JPEG paradox:
  sd15_only: FLUX EDR increases 0.200 -> 0.300 at q=75 (+50%)
  H7 target:  FLUX EDR >= 0.400 at q=75 (explicit DCT-DiT targeting)

Usage:
    python run_h7_eval.py \
        --h7-checkpoint ../../../../outputs/h7/models/final.pth \
        --sd15-checkpoint ../../../../checkpoints/diffvax_trained.pth \
        --data-dir ../../../../data --output-dir results/ --n-images 50
"""
import argparse, csv, io, sys
from collections import defaultdict
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

from diffvax.model import NestedUNet
from diffvax.attack import Attack
from diffvax.attack_flux import FluxAttack
from diffvax.utils import prepare_mask_and_masked_image, get_train_val_image_prompt_list
from eval_metrics import psnr as _psnr, ssim as _ssim

RESOLUTION = 512
FLUX_STEPS = 4
SD15_STEPS = 20
JPEG_CONDITIONS = [("clean", None), ("q85", 85), ("q75", 75), ("q70", 70)]


def t2pil(t):
    return TF.to_pil_image((t.float().squeeze(0).cpu().clamp(-1, 1) + 1) / 2)

def pil2t(p):
    return TF.to_tensor(p).unsqueeze(0) * 2 - 1

def compress_jpeg(image_t, quality):
    buf = io.BytesIO()
    t2pil(image_t).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return pil2t(Image.open(buf).convert("RGB"))

def load_model(path):
    m = NestedUNet(num_classes=3).cuda()
    m.load_state_dict(torch.load(path, weights_only=True))
    m.train()  # BN train mode: running_var near-zero from batch_size=1 training
    return m

def do_edit(attack_mdl, img_t, mask_t, prompt, steps, seed=None):
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        out = attack_mdl.attack(
            prompt=[prompt], masked_image=img_t.half().cuda(),
            mask=mask_t.half().cuda(), height=RESOLUTION, width=RESOLUTION,
            num_inference_steps=steps, batch_size=1, generator=gen,
        )
    return out.float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h7-checkpoint", default=None)
    ap.add_argument("--sd15-checkpoint", required=True)
    ap.add_argument("--data-dir", default="../../../../data")
    ap.add_argument("--output-dir", default="results/")
    ap.add_argument("--n-images", type=int, default=50)
    ap.add_argument("--flux-model", default="black-forest-labs/FLUX.1-schnell")
    ap.add_argument("--sd15-model", default="runwayml/stable-diffusion-inpainting")
    ap.add_argument("--skip-sd15-editor", action="store_true",
                    help="Only test against FLUX (faster, ~2x)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = {"sd15_only": args.sd15_checkpoint}
    if args.h7_checkpoint:
        ckpts["h7_jpeg"] = args.h7_checkpoint

    print("Loading attack models...")
    attack_models = {}
    attack_models["flux_schnell"] = FluxAttack(args.flux_model, guidance_scale=0.0)
    if not args.skip_sd15_editor:
        attack_models["sd15"] = Attack(args.sd15_model)

    _, val_list = get_train_val_image_prompt_list(args.data_dir)
    val_list = val_list[:args.n_images]

    rows = []
    for ckpt_name, ckpt_path in ckpts.items():
        print(f"\n=== Checkpoint: {ckpt_name} ===")
        imm_model = load_model(ckpt_path)

        for item in tqdm(val_list, desc=ckpt_name):
            img_name = item["image"]
            prompts = item["prompts"][:3]

            pil_img = Image.open(
                Path(args.data_dir) / "validation" / "images" / img_name
            ).convert("RGB").resize((RESOLUTION, RESOLUTION))
            pil_mask = Image.open(
                Path(args.data_dir) / "validation" / "masks" / ("mask_" + Path(img_name).stem + ".png")
            ).convert("L").resize((RESOLUTION, RESOLUTION))

            mask_t, _, img_t = prepare_mask_and_masked_image(pil_img, pil_mask)
            img_t = img_t.half().cuda()
            mask_t = mask_t.half().cuda()

            # Generate immunization
            img_f = img_t.float()
            with torch.no_grad():
                perturb = imm_model(img_f)
            perturb = perturb * (1 - mask_t.float())
            imm_t = torch.clamp(img_f + perturb, -1, 1)

            psnr_v = _psnr(imm_t.cpu(), img_f.cpu())
            ssim_v = _ssim(imm_t.cpu(), img_f.cpu())
            base_seed = abs(hash(img_name)) % (2**31)

            for mdl_name, atk in attack_models.items():
                steps = FLUX_STEPS if mdl_name == "flux_schnell" else SD15_STEPS

                # Clean edits (no JPEG on test image)
                clean_edits = {}
                for i_p, prompt in enumerate(prompts):
                    clean_edits[prompt] = do_edit(atk, img_t, mask_t, prompt, steps, base_seed + i_p)
                torch.cuda.empty_cache()

                for label, quality in JPEG_CONDITIONS:
                    if quality is None:
                        test_img_t = imm_t.half()
                    else:
                        test_img_t = compress_jpeg(imm_t.cpu(), quality).half().cuda()

                    for i_p, prompt in enumerate(prompts):
                        edited = do_edit(atk, test_img_t, mask_t, prompt, steps, base_seed + i_p)
                        torch.cuda.empty_cache()

                        sc = _ssim(clean_edits[prompt].cuda().float(), img_f.cuda())
                        si = _ssim(edited.cuda().float(), img_f.cuda())

                        rows.append({
                            "checkpoint": ckpt_name,
                            "edit_model": mdl_name,
                            "jpeg": label,
                            "image": img_name,
                            "prompt": prompt,
                            "psnr_imm": round(psnr_v, 3),
                            "ssim_imm": round(ssim_v, 4),
                            "ssim_clean_edit": round(sc, 4),
                            "ssim_imm_edit": round(si, 4),
                            "disrupted": int(si < sc - 0.05),
                        })

    csv_path = out_dir / "h7_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    # Summary table
    print("\n=== H7 Summary ===")
    print(f"{'Checkpoint':12s} | {'Model':12s} | {'clean':6s} | {'q85':6s} | {'q75':6s} | {'q70':6s} | JPEG_PARADOX")
    print("-" * 75)
    bk = defaultdict(list)
    for r in rows:
        bk[(r["checkpoint"], r["edit_model"], r["jpeg"])].append(r)

    for ck in sorted(set(r["checkpoint"] for r in rows)):
        for mdl in list(attack_models.keys()):
            parts = [f"{ck:12s}", f"{mdl:12s}"]
            edrs = {}
            for lbl, _ in JPEG_CONDITIONS:
                rs = bk.get((ck, mdl, lbl), [])
                edrs[lbl] = sum(r["disrupted"] for r in rs) / len(rs) if rs else float("nan")
                parts.append(f"{edrs[lbl]:.3f}" if rs else "  N/A")
            paradox = ""
            if mdl == "flux_schnell" and edrs.get("q75", 0) > edrs.get("clean", 0):
                paradox = " PARADOX (+{:.0f}%)".format(
                    (edrs["q75"] / edrs["clean"] - 1) * 100 if edrs["clean"] > 0 else 0)
            parts.append(paradox)
            print(" | ".join(parts))

    print(f"\nResults: {csv_path}")
