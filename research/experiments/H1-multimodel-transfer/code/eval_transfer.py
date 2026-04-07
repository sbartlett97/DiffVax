#!/usr/bin/env python3
"""Cross-model transfer evaluation for DiffVax immunization (Experiment H1).

Tests how well immunizations trained on one set of models transfer to
held-out architectures. Produces a CSV with Edit Disruption Rate per
(checkpoint, evaluation_model, image) combination.

Usage:
    python eval_transfer.py \
        --checkpoints sd15_only=path/to/sd15.pth multimodel=path/to/mm.pth \
        --eval-models sd15 flux_schnell sd35 \
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

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

from diffvax.model import NestedUNet
from diffvax.utils import prepare_mask_and_masked_image, get_train_val_image_prompt_list
from eval_metrics import psnr as _psnr, ssim as _ssim


RESOLUTION = 512
NUM_INFERENCE_STEPS = 20
PROMPTS_PER_IMAGE = 3


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = (t.float().squeeze(0).cpu().clamp(-1, 1) + 1) / 2
    return TF.to_pil_image(t)


def load_attack_model(model_type: str):
    """Load and return an attack model by type string."""
    if model_type == "sd15":
        from diffvax.attack import Attack
        return Attack("runwayml/stable-diffusion-inpainting")
    elif model_type == "flux_schnell":
        from diffvax.attack_flux import FluxAttack
        return FluxAttack("black-forest-labs/FLUX.1-schnell", guidance_scale=0.0)
    elif model_type == "flux_klein":
        from diffvax.attack_flux import FluxAttack
        return FluxAttack("black-forest-labs/FLUX.2-klein-4B", guidance_scale=3.5)
    elif model_type == "sd35":
        from diffvax.attack_sd3 import SD3Attack
        return SD3Attack("stabilityai/stable-diffusion-3.5-medium")
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")


def run_edit(attack_model, image_t, mask_t, prompt, model_type):
    """Run editing attack and return output tensor."""
    kwargs = dict(
        prompt=[prompt],
        masked_image=image_t.half().cuda(),
        mask=mask_t.half().cuda(),
        height=RESOLUTION,
        width=RESOLUTION,
        num_inference_steps=NUM_INFERENCE_STEPS,
        batch_size=1,
    )
    return attack_model.attack(**kwargs).float().cpu()


def compute_ssim(a, b):
    return _ssim(a.cuda(), b.cuda())


def apply_immunization(model, image_t, mask_t):
    """Apply NestedUNet immunization."""
    img_f = image_t.float().cuda()
    with torch.no_grad():
        perturb = model(img_f)
    perturb = perturb * (1 - mask_t.float().cuda())
    return torch.clamp(img_f + perturb, -1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+",
                        help="name=path pairs, e.g. sd15_only=path/sd15.pth multimodel=path/mm.pth")
    parser.add_argument("--eval-models", nargs="+", default=["sd15", "flux_schnell", "sd35"],
                        help="Models to evaluate transfer on")
    parser.add_argument("--data-dir", default="../../../../data")
    parser.add_argument("--output-dir", default="results/")
    parser.add_argument("--n-images", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse checkpoints
    checkpoints = {}
    for spec in (args.checkpoints or []):
        name, path = spec.split("=", 1)
        checkpoints[name] = Path(path)

    if not checkpoints:
        parser.error("Provide at least one --checkpoints name=path")

    # Load val data
    _, val_list = get_train_val_image_prompt_list(args.data_dir)
    val_list = val_list[:args.n_images]

    results = []

    # For each eval model
    for eval_model_type in args.eval_models:
        print(f"\nLoading eval model: {eval_model_type}")
        attack_model = load_attack_model(eval_model_type)

        # For each checkpoint
        for ckpt_name, ckpt_path in checkpoints.items():
            print(f"  Loading checkpoint: {ckpt_name}")
            model = NestedUNet(num_classes=3).cuda()
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            model.training = False

            pbar = tqdm(val_list, desc=f"{ckpt_name} -> {eval_model_type}")
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

                mask_t, masked_t, image_t = prepare_mask_and_masked_image(pil_image, pil_mask)
                image_t = image_t.half().cuda()
                mask_t = mask_t.half().cuda()

                # Apply immunization
                immunized_t = apply_immunization(model, image_t, mask_t)

                # Imperceptibility
                psnr_val = _psnr(immunized_t.float(), image_t.float())
                ssim_imm_orig = _ssim(immunized_t.float(), image_t.float())

                for prompt in prompts:
                    # Edit clean image
                    edited_clean = run_edit(attack_model, image_t, mask_t, prompt, eval_model_type)
                    # Edit immunized image
                    edited_imm = run_edit(
                        attack_model,
                        immunized_t.half(),
                        mask_t, prompt, eval_model_type
                    )

                    ssim_clean_edit = compute_ssim(
                        edited_clean.cuda().float(), image_t.float()
                    )
                    ssim_imm_edit = compute_ssim(
                        edited_imm.cuda().float(), image_t.float()
                    )
                    disrupted = int(ssim_imm_edit < ssim_clean_edit - 0.05)

                    results.append({
                        "checkpoint": ckpt_name,
                        "eval_model": eval_model_type,
                        "image": image_name,
                        "prompt": prompt,
                        "psnr_immunized": round(psnr_val, 3),
                        "ssim_imm_orig": round(ssim_imm_orig, 4),
                        "ssim_clean_edit": round(ssim_clean_edit, 4),
                        "ssim_imm_edit": round(ssim_imm_edit, 4),
                        "disrupted": disrupted,
                    })

    # Write CSV
    csv_path = out_dir / "transfer_edr_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Summary table
    print("\n=== H1 Transfer Summary ===")
    print(f"{'Checkpoint':20s} | {'Eval Model':15s} | EDR   | PSNR  | Imm SSIM")
    print("-" * 70)
    summary = defaultdict(list)
    for r in results:
        key = (r["checkpoint"], r["eval_model"])
        summary[key].append(r)

    for (ckpt, model), rows in sorted(summary.items()):
        edr = sum(r["disrupted"] for r in rows) / len(rows)
        psnr = sum(r["psnr_immunized"] for r in rows) / len(rows)
        ssim = sum(r["ssim_imm_orig"] for r in rows) / len(rows)
        print(f"{ckpt:20s} | {model:15s} | {edr:.3f} | {psnr:.1f} | {ssim:.4f}")

    print(f"\nFull results: {csv_path}")


if __name__ == "__main__":
    main()
