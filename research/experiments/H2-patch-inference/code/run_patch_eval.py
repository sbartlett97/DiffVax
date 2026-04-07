#!/usr/bin/env python3
"""Evaluate patch-based 1088x1088 immunization (Experiment H2).

Loads the 512-trained DiffVax checkpoint and applies patch_immunize at
1088x1088 with different stride settings. Evaluates:
  - Edit Disruption Rate (EDR) on SD 1.5 inpainting
  - PSNR/SSIM of immunized vs original image (imperceptibility)

Usage:
    python run_patch_eval.py \
        --checkpoint ../../../../checkpoints/diffvax_trained.pth \
        --data-dir ../../../../data \
        --output-dir results/ \
        --n-images 50
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

from diffvax.model import NestedUNet
from diffvax.attack import Attack
from diffvax.patch_immunize import patch_immunize
from diffvax.utils import prepare_mask_and_masked_image, get_train_val_image_prompt_list
from eval_metrics import psnr as compute_psnr_fn, ssim as compute_ssim_fn


TARGET_RESOLUTION = 1088
PATCH_SIZE = 512
STRIDE_CONDITIONS = {
    "no_overlap": 512,
    "25pct_overlap": 384,
    "50pct_overlap": 256,
    "baseline_512": None,  # direct 512x512 inference
}
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert (1,3,H,W) tensor in [-1,1] to PIL RGB."""
    t = (t.float().squeeze(0).cpu().clamp(-1, 1) + 1) / 2
    return TF.to_pil_image(t)


def compute_ssim(img_a: torch.Tensor, img_b: torch.Tensor) -> float:
    return compute_ssim_fn(img_a, img_b)


def compute_psnr(img_a: torch.Tensor, img_b: torch.Tensor) -> float:
    return compute_psnr_fn(img_a, img_b)


def run_edit(attack_model, pil_image, pil_mask, prompt, resolution):
    """Run SD 1.5 inpainting and return edited image tensor."""
    pil_image = pil_image.resize((resolution, resolution))
    pil_mask = pil_mask.resize((resolution, resolution))
    mask_t, masked_image_t, image_t = prepare_mask_and_masked_image(pil_image, pil_mask)

    edited = attack_model.attack(
        prompt=[prompt],
        masked_image=masked_image_t.half().cuda(),
        mask=mask_t.half().cuda(),
        height=resolution,
        width=resolution,
        num_inference_steps=NUM_INFERENCE_STEPS,
        batch_size=1,
    )
    return edited.float().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../../../../checkpoints/diffvax_trained.pth")
    parser.add_argument("--data-dir", default="../../../../data")
    parser.add_argument("--output-dir", default="results/")
    parser.add_argument("--n-images", type=int, default=50)
    parser.add_argument("--attack-model", default="runwayml/stable-diffusion-inpainting")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load immunization model
    model = NestedUNet(num_classes=3).cuda()
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.training = False

    # Load attack model (SD 1.5)
    attack_model = Attack(args.attack_model)

    # Load validation data
    _, val_list = get_train_val_image_prompt_list(args.data_dir)
    val_list = val_list[:args.n_images]

    results = []

    pbar = tqdm(val_list, desc="Evaluating H2")
    for item in pbar:
        image_name = item["image"]
        prompts = item["prompts"][:2]  # use first 2 prompts per image

        data_path = Path(args.data_dir)
        pil_image = Image.open(data_path / "validation" / "images" / image_name).convert("RGB")
        mask_name = "mask_" + Path(image_name).stem + ".png"
        pil_mask = Image.open(data_path / "validation" / "masks" / mask_name).convert("L")

        # Upscale to target resolution
        pil_image_1088 = pil_image.resize((TARGET_RESOLUTION, TARGET_RESOLUTION), Image.LANCZOS)
        pil_mask_1088 = pil_mask.resize((TARGET_RESOLUTION, TARGET_RESOLUTION), Image.NEAREST)

        # Prepare tensors at 1088
        mask_t, masked_t, image_t = prepare_mask_and_masked_image(pil_image_1088, pil_mask_1088)
        image_t_half = image_t.half().cuda()
        mask_t_half = mask_t.half().cuda()

        for condition_name, stride in STRIDE_CONDITIONS.items():
            if stride is None:
                # Baseline: resize to 512, immunize at 512, resize back
                pil_512 = pil_image.resize((512, 512))
                pil_mask_512 = pil_mask.resize((512, 512))
                mask_512, _, img_512 = prepare_mask_and_masked_image(pil_512, pil_mask_512)
                immunized_512 = model(img_512.float().cuda())
                immunized_512 = torch.clamp(img_512.cuda() + immunized_512 * (1 - mask_512.cuda()), -1, 1)
                # Upscale immunized back to 1088 for fair comparison
                immunized_pil = tensor_to_pil(immunized_512).resize((TARGET_RESOLUTION, TARGET_RESOLUTION), Image.LANCZOS)
                immunized_t = TF.to_tensor(immunized_pil).unsqueeze(0) * 2 - 1  # (1,3,H,W) in [-1,1]
            else:
                immunized_t = patch_immunize(
                    model, image_t_half.float(), mask_t_half.float(),
                    patch_size=PATCH_SIZE, stride=stride,
                )

            immunized_pil = tensor_to_pil(immunized_t.unsqueeze(0) if immunized_t.dim() == 3 else immunized_t)
            orig_pil = pil_image_1088

            # Imperceptibility metrics
            orig_t = image_t_half.float()
            imm_t = immunized_t.cuda().float() if immunized_t.dim() == 4 else immunized_t.unsqueeze(0).cuda().float()
            psnr_val = compute_psnr(imm_t, orig_t)
            ssim_val = compute_ssim(imm_t, orig_t)

            for prompt in prompts:
                # Edit clean image
                edited_clean = run_edit(
                    attack_model, orig_pil, pil_mask_1088, prompt, TARGET_RESOLUTION
                )
                # Edit immunized image
                edited_imm = run_edit(
                    attack_model, immunized_pil, pil_mask_1088, prompt, TARGET_RESOLUTION
                )

                # EDR: immunized edit should have lower SSIM vs clean edit (more disrupted)
                # Both edited images and orig_t are at TARGET_RESOLUTION x TARGET_RESOLUTION
                ssim_clean_edit = compute_ssim(edited_clean.cuda(), orig_t)
                ssim_imm_edit = compute_ssim(edited_imm.cuda(), orig_t)
                disrupted = ssim_imm_edit < ssim_clean_edit - 0.05

                results.append({
                    "image": image_name,
                    "prompt": prompt,
                    "condition": condition_name,
                    "psnr_immunized": round(psnr_val, 3),
                    "ssim_immunized": round(ssim_val, 4),
                    "ssim_clean_edit": round(ssim_clean_edit, 4),
                    "ssim_imm_edit": round(ssim_imm_edit, 4),
                    "disrupted": int(disrupted),
                })

    # Write CSV
    csv_path = out_dir / "patch_edr_metrics.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Summary by condition
    print("\n=== H2 Summary ===")
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)

    for cond, rows in sorted(by_cond.items()):
        edr = sum(r["disrupted"] for r in rows) / len(rows)
        avg_psnr = sum(r["psnr_immunized"] for r in rows) / len(rows)
        avg_ssim = sum(r["ssim_immunized"] for r in rows) / len(rows)
        print(f"  {cond:20s}: EDR={edr:.3f}  PSNR={avg_psnr:.1f}  SSIM={avg_ssim:.4f}")

    print(f"\nFull results saved to {csv_path}")


if __name__ == "__main__":
    main()
