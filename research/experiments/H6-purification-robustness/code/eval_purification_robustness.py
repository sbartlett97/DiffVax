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
PURIFICATION_STEPS = 4   # FLUX.1-schnell is distilled for 4 steps; using 20 degrades quality
EDIT_STEPS = 4    # FLUX.1-schnell edit steps (distilled 4-step model)
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


def purify_with_flux(flux_model: FluxAttack, image_t: torch.Tensor, strength: float = 0.3) -> torch.Tensor:
    """Run FLUX reconstruction over the entire image (empty mask = full reconstruction).

    This is the EditorClean purification strategy from arXiv:2603.13028.
    An all-zero mask means the model reconstructs the whole image from scratch,
    removing the adversarial perturbation in the process.

    Args:
        strength: Purification aggressiveness (0.3=mild, 0.5=moderate, 0.7=strong).
                  Higher = better at removing perturbations but more image distortion.
    """
    empty_mask = torch.zeros(1, 1, RESOLUTION, RESOLUTION, device="cuda")
    with torch.no_grad():
        purified = flux_model.attack(
            prompt=[""],
            masked_image=image_t.half().cuda(),
            mask=empty_mask,
            height=RESOLUTION,
            width=RESOLUTION,
            num_inference_steps=PURIFICATION_STEPS,
            strength=strength,
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
    parser.add_argument(
        "--purify-strengths", nargs="+", type=float, default=[0.3, 0.5, 0.7],
        help="Purification strengths to test (adversary's denoising strength). "
             "Higher = more aggressive purification. Default: 0.3 0.5 0.7"
    )
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
        model.eval()  # sets dropout/batchnorm to inference mode

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
            psnr_imm_vs_orig = compute_psnr(immunized_t, image_t.float())

            # Precompute clean edits once per image (shared across purify_strengths)
            clean_edits = {}
            for prompt in prompts:
                with torch.no_grad():
                    clean_edits[prompt] = flux.attack(
                        prompt=[prompt], masked_image=image_t, mask=mask_t,
                        height=RESOLUTION, width=RESOLUTION,
                        num_inference_steps=EDIT_STEPS, batch_size=1,
                    ).float()
            torch.cuda.empty_cache()

            # Edit immunized directly (no purification)
            direct_edits = {}
            for prompt in prompts:
                with torch.no_grad():
                    direct_edits[prompt] = flux.attack(
                        prompt=[prompt], masked_image=immunized_t.half(), mask=mask_t,
                        height=RESOLUTION, width=RESOLUTION,
                        num_inference_steps=EDIT_STEPS, batch_size=1,
                    ).float()
            torch.cuda.empty_cache()

            # Test each purification strength
            for purify_strength in args.purify_strengths:
                purified_t = purify_with_flux(flux, immunized_t, strength=purify_strength)
                psnr_purified_vs_orig = compute_psnr(purified_t, image_t.float())
                ssim_purified_vs_orig = compute_ssim(purified_t, image_t.float())

                for prompt in prompts:
                    edited_clean = clean_edits[prompt]
                    edited_immunized = direct_edits[prompt]

                    with torch.no_grad():
                        edited_purified = flux.attack(
                            prompt=[prompt], masked_image=purified_t.half(), mask=mask_t,
                            height=RESOLUTION, width=RESOLUTION,
                            num_inference_steps=EDIT_STEPS, batch_size=1,
                        ).float()
                    torch.cuda.empty_cache()

                    ssim_clean_edit = compute_ssim(edited_clean, image_t.float())
                    ssim_direct_edit = compute_ssim(edited_immunized, image_t.float())
                    ssim_purified_edit = compute_ssim(edited_purified, image_t.float())

                    # Disrupted = DiffVax drives output toward zeros, so a successful
                    # immunization produces a more-blanked edit (LOWER SSIM vs original)
                    # than the clean edit. Matches H2 eval convention.
                    direct_disrupted = int(ssim_direct_edit < ssim_clean_edit - 0.05)
                    purified_disrupted = int(ssim_purified_edit < ssim_clean_edit - 0.05)

                    results.append({
                        "checkpoint": ckpt_name,
                        "image": image_name,
                        "prompt": prompt,
                        "purify_strength": purify_strength,
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
    print(f"{'Checkpoint':15s} | {'Strength':8s} | Direct EDR | After-Purify EDR | PSNR Purified")
    print("-" * 70)
    by_key = defaultdict(list)
    for r in results:
        by_key[(r["checkpoint"], r["purify_strength"])].append(r)

    direct_edrs = defaultdict(list)
    for r in results:
        direct_edrs[r["checkpoint"]].append(r["direct_disrupted"])

    for ckpt in sorted(set(r["checkpoint"] for r in results)):
        direct_edr = sum(direct_edrs[ckpt]) / len(direct_edrs[ckpt])
        for strength in sorted(set(r["purify_strength"] for r in results)):
            rows = by_key[(ckpt, strength)]
            purified_edr = sum(r["purified_disrupted"] for r in rows) / len(rows)
            psnr_p = sum(r["psnr_purified"] for r in rows) / len(rows)
            print(f"{ckpt:15s} | {strength:8.1f} | {direct_edr:.3f}      | {purified_edr:.3f}            | {psnr_p:.1f}")

    print(f"\nH6 prediction: at each purify_strength, flux_trained purified_edr > sd15_only purified_edr")
    print(f"(flux-trained immunization resists FLUX-based purification; sd15-only does not)")
    print(f"Full results: {csv_path}")


if __name__ == "__main__":
    main()
