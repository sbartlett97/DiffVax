#!/usr/bin/env python3
"""Evaluate DiffVax immunization against FLUX-based purification attacks (H6).

Tests whether immunizations trained against FLUX resist the EditorClean
purification attack from arXiv:2603.13028 ("Purify Once, Edit Freely").

The EditorClean attack reconstructs the image using FLUX-Fill with an empty
mask to "purify" the immunization before editing. This script compares:
  - DiffVax-SD15: purification succeeds (expected)
  - DiffVax-FLUX: purification fails (H6 prediction)

Usage:
    python eval_purification_robustness.py \
        --checkpoint-sd15 path/to/sd15.pth \
        --checkpoint-flux path/to/multimodel.pth \
        --data-dir ../../../../data \
        --output-dir results/ \
        --n-images 30
"""

import argparse
import csv
import sys
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
from diffvax.attack_flux import FluxAttack
from diffvax.utils import prepare_mask_and_masked_image, get_train_val_image_prompt_list
from eval_metrics import psnr as _psnr, ssim as _ssim


RESOLUTION = 512
PURIFICATION_STEPS = 20  # EditorClean uses full denoise
EDIT_STEPS = 20
PROMPTS_PER_IMAGE = 3


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = (t.float().squeeze(0).cpu().clamp(-1, 1) + 1) / 2
    return TF.to_pil_image(t)


def pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    return TF.to_tensor(pil).unsqueeze(0) * 2 - 1


def apply_immunization(model: torch.nn.Module, image_t: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
    img_f = image_t.float().cuda()
    with torch.no_grad():
        perturb = model(img_f)
    perturb = perturb * (1 - mask_t.float().cuda())
    return torch.clamp(img_f + perturb, -1, 1)


def purify_with_flux(flux_model: FluxAttack, image_t: torch.Tensor) -> torch.Tensor:
    """Run FLUX reconstruction over the entire image (empty mask = full reconstruction).

    This is the EditorClean purification strategy from arXiv:2603.13028.
    An all-zero mask means the model reconstructs the whole image from scratch,
    removing the adversarial perturbation in the process.
    """
    # Empty mask: reconstruct entire image (no masked region to preserve)
    empty_mask = torch.zeros(1, 1, RESOLUTION, RESOLUTION, device="cuda")
    with torch.no_grad():
        purified = flux_model.attack(
            prompt=[""],
            masked_image=image_t.half().cuda(),
            mask=empty_mask,
            height=RESOLUTION,
            width=RESOLUTION,
            num_inference_steps=PURIFICATION_STEPS,
            strength=0.3,  # Low strength: minimal change, remove only perturbation
            batch_size=1,
        )
    return purified.float()


def compute_ssim(a, b):
    return _ssim(a.cuda().float(), b.cuda().float())


def compute_psnr(a, b):
    return _psnr(a.cuda().float(), b.cuda().float())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-sd15", required=True,
                        help="DiffVax-SD15-only checkpoint path")
    parser.add_argument("--checkpoint-flux", required=True,
                        help="DiffVax-FLUX multi-model checkpoint path")
    parser.add_argument("--data-dir", default="../../../../data")
    parser.add_argument("--output-dir", default="results/")
    parser.add_argument("--n-images", type=int, default=30)
    parser.add_argument("--flux-model", default="black-forest-labs/FLUX.1-schnell")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load FLUX for both purification and editing
    print("Loading FLUX model...")
    flux = FluxAttack(args.flux_model, guidance_scale=0.0)

    checkpoints = {
        "sd15_only": Path(args.checkpoint_sd15),
        "flux_trained": Path(args.checkpoint_flux),
    }

    _, val_list = get_train_val_image_prompt_list(args.data_dir)
    val_list = val_list[:args.n_images]

    results = []

    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\nLoading checkpoint: {ckpt_name}")
        model = NestedUNet(num_classes=3).cuda()
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.training = False

        pbar = tqdm(val_list, desc=f"H6 eval — {ckpt_name}")
        for item in pbar:
            image_name = item["image"]
            prompts = item["prompts"][:PROMPTS_PER_IMAGE]

            data_path = Path(args.data_dir)
            pil_image = Image.open(
                data_path / "validation" / "images" / image_name
            ).convert("RGB").resize((RESOLUTION, RESOLUTION))
            mask_name = "mask_" + Path(image_name).stem + ".png"
            pil_mask = Image.open(
                data_path / "validation" / "masks" / mask_name
            ).convert("L").resize((RESOLUTION, RESOLUTION))

            mask_t, _, image_t = prepare_mask_and_masked_image(pil_image, pil_mask)
            image_t = image_t.half().cuda()
            mask_t = mask_t.half().cuda()

            # Apply immunization
            immunized_t = apply_immunization(model, image_t, mask_t)

            # Purify using FLUX (EditorClean from arXiv:2603.13028)
            purified_t = purify_with_flux(flux, immunized_t)

            # Metrics: did purification succeed?
            psnr_imm_vs_orig = compute_psnr(immunized_t, image_t.float())
            psnr_purified_vs_orig = compute_psnr(purified_t, image_t.float())
            ssim_purified_vs_orig = compute_ssim(purified_t, image_t.float())

            for prompt in prompts:
                # Edit 1: edit clean image (baseline)
                edited_clean = flux.attack(
                    prompt=[prompt], masked_image=image_t, mask=mask_t,
                    height=RESOLUTION, width=RESOLUTION,
                    num_inference_steps=EDIT_STEPS, batch_size=1,
                ).float()

                # Edit 2: edit immunized image directly (no purification)
                edited_immunized = flux.attack(
                    prompt=[prompt], masked_image=immunized_t.half(), mask=mask_t,
                    height=RESOLUTION, width=RESOLUTION,
                    num_inference_steps=EDIT_STEPS, batch_size=1,
                ).float()

                # Edit 3: edit purified image (purification attack)
                edited_purified = flux.attack(
                    prompt=[prompt], masked_image=purified_t.half(), mask=mask_t,
                    height=RESOLUTION, width=RESOLUTION,
                    num_inference_steps=EDIT_STEPS, batch_size=1,
                ).float()

                ssim_clean_edit = compute_ssim(edited_clean, image_t.float())
                ssim_direct_edit = compute_ssim(edited_immunized, image_t.float())
                ssim_purified_edit = compute_ssim(edited_purified, image_t.float())

                # Disrupted = immunized edit looks more like original (less edited)
                direct_disrupted = int(ssim_direct_edit > ssim_clean_edit - 0.05)
                purified_disrupted = int(ssim_purified_edit > ssim_clean_edit - 0.05)

                results.append({
                    "checkpoint": ckpt_name,
                    "image": image_name,
                    "prompt": prompt,
                    "psnr_immunized": round(psnr_imm_vs_orig, 3),
                    "psnr_purified": round(psnr_purified_vs_orig, 3),
                    "ssim_purified_vs_orig": round(ssim_purified_vs_orig, 4),
                    "ssim_clean_edit": round(ssim_clean_edit, 4),
                    "ssim_direct_edit": round(ssim_direct_edit, 4),
                    "ssim_purified_edit": round(ssim_purified_edit, 4),
                    "direct_disrupted": direct_disrupted,
                    "purified_disrupted": purified_disrupted,
                })

    # Write CSV
    csv_path = out_dir / "purification_robustness.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Summary
    print("\n=== H6 Summary ===")
    print(f"{'Checkpoint':15s} | Direct EDR | After-Purify EDR | PSNR Purified")
    print("-" * 60)
    by_ckpt = defaultdict(list)
    for r in results:
        by_ckpt[r["checkpoint"]].append(r)

    for ckpt, rows in sorted(by_ckpt.items()):
        direct_edr = sum(r["direct_disrupted"] for r in rows) / len(rows)
        purified_edr = sum(r["purified_disrupted"] for r in rows) / len(rows)
        psnr_p = sum(r["psnr_purified"] for r in rows) / len(rows)
        print(f"{ckpt:15s} | {direct_edr:.3f}      | {purified_edr:.3f}            | {psnr_p:.1f}")

    print(f"\nH6 prediction: flux_trained purified_edr should be HIGHER than sd15_only purified_edr")
    print(f"Full results: {csv_path}")


if __name__ == "__main__":
    main()
