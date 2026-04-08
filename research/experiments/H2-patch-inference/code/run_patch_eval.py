#!/usr/bin/env python3
"""Evaluate patch-based 1088x1088 immunization (Experiment H2).

Loads the 512-trained DiffVax checkpoint and applies patch_immunize at
1088x1088 with different stride settings. Evaluates:
  - Edit Disruption Rate (EDR) on SD 1.5 inpainting
  - PSNR/SSIM of immunized vs original image (imperceptibility)

Edit protocol: immunize at 1088px (with patches), then DOWNSCALE to 512px
before editing. This matches real adversary workflow (SD 1.5 is 512px-native)
and avoids OOM from 1088px self-attention maps (which are 18x larger than 512px).

Usage:
    python run_patch_eval.py \
        --checkpoint ../../../../checkpoints/diffvax_trained.pth \
        --data-dir ../../../../data \
        --output-dir results/ \
        --n-images 50
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
from diffvax.attack import Attack
from diffvax.patch_immunize import patch_immunize
from diffvax.utils import prepare_mask_and_masked_image, get_train_val_image_prompt_list
from eval_metrics import psnr as _psnr, ssim as _ssim


TARGET_RESOLUTION = 1088   # resolution at which immunization is applied
EDIT_RESOLUTION = 512      # resolution at which editing occurs (SD 1.5 is 512-native)
PATCH_SIZE = 512
STRIDE_CONDITIONS = {
    "no_overlap":    512,
    "25pct_overlap": 384,
    "50pct_overlap": 256,
    "baseline_512":  None,  # immunize at 512, then upscale — no patch tiling
}
NUM_INFERENCE_STEPS = 20


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert (1,3,H,W) or (3,H,W) tensor in [-1,1] to PIL RGB."""
    if t.dim() == 4:
        t = t.squeeze(0)
    return TF.to_pil_image((t.float().cpu().clamp(-1, 1) + 1) / 2)


def pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    """Convert PIL RGB to (1,3,H,W) tensor in [-1,1]."""
    return TF.to_tensor(pil).unsqueeze(0) * 2 - 1


def run_edit(attack_model, pil_image_512, pil_mask_512, prompt):
    """Run SD 1.5 inpainting at 512px and return edited tensor (cpu, float)."""
    mask_t, masked_t, _ = prepare_mask_and_masked_image(pil_image_512, pil_mask_512)
    with torch.no_grad():
        edited = attack_model.attack(
            prompt=[prompt],
            masked_image=masked_t.half().cuda(),
            mask=mask_t.half().cuda(),
            height=EDIT_RESOLUTION,
            width=EDIT_RESOLUTION,
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
    # Use train() mode: DiffVax BN running stats were accumulated with batch_size=1
    # (near-zero running_var). In eval mode, BN collapses activations 78×.
    # model.training=False (old bug) accidentally kept child BN layers in train mode;
    # model.train() makes this explicit and correct.
    model.train()

    # Load SD 1.5 attack model with memory optimisations
    attack_model = Attack(args.attack_model)
    attack_model.model.enable_attention_slicing()      # ~2x memory reduction for self-attention
    # enable_model_cpu_offload conflicts with the manual .cuda() calls in attack.py
    # so we skip it here and rely on attention_slicing + empty_cache instead

    # Load validation data
    _, val_list = get_train_val_image_prompt_list(args.data_dir)
    val_list = val_list[:args.n_images]

    results = []
    data_path = Path(args.data_dir)

    pbar = tqdm(val_list, desc="Evaluating H2")
    for item in pbar:
        image_name = item["image"]
        prompts = item["prompts"][:2]

        pil_image = Image.open(
            data_path / "validation" / "images" / image_name
        ).convert("RGB")
        mask_name = "mask_" + Path(image_name).stem + ".png"
        pil_mask = Image.open(
            data_path / "validation" / "masks" / mask_name
        ).convert("L")

        # 512px versions (for baseline condition and editing)
        pil_image_512 = pil_image.resize((EDIT_RESOLUTION, EDIT_RESOLUTION), Image.LANCZOS)
        pil_mask_512 = pil_mask.resize((EDIT_RESOLUTION, EDIT_RESOLUTION), Image.NEAREST)
        orig_t_512 = pil_to_tensor(pil_image_512)  # (1,3,512,512) in [-1,1]

        # 1088px versions (for patch immunization)
        pil_image_1088 = pil_image.resize((TARGET_RESOLUTION, TARGET_RESOLUTION), Image.LANCZOS)
        pil_mask_1088 = pil_mask.resize((TARGET_RESOLUTION, TARGET_RESOLUTION), Image.NEAREST)
        mask_t_1088, _, image_t_1088 = prepare_mask_and_masked_image(pil_image_1088, pil_mask_1088)
        image_t_1088 = image_t_1088.cuda()
        mask_t_1088 = mask_t_1088.cuda()

        # Precompute: edit the CLEAN image once per image (shared across conditions)
        # Immunization-quality metrics compare against this baseline edit
        clean_edits = {}
        for prompt in prompts:
            with torch.no_grad():
                clean_edits[prompt] = run_edit(attack_model, pil_image_512, pil_mask_512, prompt)
        torch.cuda.empty_cache()

        for condition_name, stride in STRIDE_CONDITIONS.items():
            if stride is None:
                # Baseline: immunize at 512, scale back to 1088 for imperceptibility
                # then downscale to 512 for editing (round-trip to test upscale-downscale)
                mask_512, _, img_512 = prepare_mask_and_masked_image(pil_image_512, pil_mask_512)
                with torch.no_grad():
                    perturb_512 = model(img_512.cuda())
                    perturb_512 = perturb_512 * (1 - mask_512.cuda())
                    immunized_512_t = torch.clamp(img_512.cuda() + perturb_512, -1, 1)
                del perturb_512

                # Imperceptibility measured at 512 for baseline
                psnr_val = _psnr(immunized_512_t.float().cpu(), img_512)
                ssim_val = _ssim(immunized_512_t.float().cpu(), img_512)

                # Editing: use the 512-immunized image directly
                immunized_pil_for_edit = tensor_to_pil(immunized_512_t.cpu())
                del immunized_512_t
            else:
                # Patch immunization at 1088
                with torch.no_grad():
                    immunized_1088_t = patch_immunize(
                        model, image_t_1088.float(), mask_t_1088.float(),
                        patch_size=PATCH_SIZE, stride=stride,
                    )

                # Imperceptibility measured at 1088
                psnr_val = _psnr(immunized_1088_t.float().cpu(), image_t_1088.float().cpu())
                ssim_val = _ssim(immunized_1088_t.float().cpu(), image_t_1088.float().cpu())

                # Downscale to 512 for editing (adversary uses 512px editor)
                immunized_pil_for_edit = tensor_to_pil(immunized_1088_t.cpu()).resize(
                    (EDIT_RESOLUTION, EDIT_RESOLUTION), Image.LANCZOS
                )
                del immunized_1088_t

            torch.cuda.empty_cache()

            for prompt in prompts:
                edited_clean = clean_edits[prompt]
                edited_imm = run_edit(attack_model, immunized_pil_for_edit, pil_mask_512, prompt)
                torch.cuda.empty_cache()

                ssim_clean_edit = _ssim(edited_clean.cuda().float(), orig_t_512.cuda().float())
                ssim_imm_edit = _ssim(edited_imm.cuda().float(), orig_t_512.cuda().float())
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

        del image_t_1088, mask_t_1088
        torch.cuda.empty_cache()

    # Write CSV
    csv_path = out_dir / "patch_edr_metrics.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Summary by condition
    print("\n=== H2 Summary ===")
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)

    for cond, rows in sorted(by_cond.items()):
        edr = sum(r["disrupted"] for r in rows) / len(rows)
        avg_psnr = sum(r["psnr_immunized"] for r in rows) / len(rows)
        avg_ssim = sum(r["ssim_immunized"] for r in rows) / len(rows)
        print(f"  {cond:20s}: EDR={edr:.3f}  PSNR={avg_psnr:.1f}  SSIM={avg_ssim:.4f}")

    print(f"\nEdit protocol: immunize at {TARGET_RESOLUTION}px, edit at {EDIT_RESOLUTION}px")
    print(f"Full results saved to {csv_path}")


if __name__ == "__main__":
    main()
