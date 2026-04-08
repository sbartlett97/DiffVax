#!/usr/bin/env python3
"""Cross-model transfer evaluation for DiffVax immunization (H1 + H7).

Tests how well immunizations trained on one set of models transfer to
held-out architectures. Optionally tests compression robustness by applying
JPEG compression to the immunized image before editing (H7 protocol).

Produces a CSV with Edit Disruption Rate per
(checkpoint, eval_model, jpeg_quality, image) combination.

Usage:
    # Standard H1 transfer evaluation
    python eval_transfer.py \
        --checkpoints sd15_only=path/sd15.pth multimodel=path/mm.pth \
        --eval-models sd15 flux_schnell sd35 \
        --data-dir ../../../../data \
        --output-dir results/ \
        --n-images 50

    # H7: also test post-JPEG EDR
    python eval_transfer.py \
        --checkpoints h1a=path/h1a.pth h7_jpeg=path/h7.pth \
        --eval-models sd15 flux_schnell \
        --jpeg-qualities 75 70 \
        --data-dir ../../../../data \
        --output-dir results/
"""

import argparse
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

from diffvax.model import NestedUNet
from diffvax.utils import prepare_mask_and_masked_image, get_train_val_image_prompt_list
from eval_metrics import psnr as _psnr, ssim as _ssim


RESOLUTION = 512
PROMPTS_PER_IMAGE = 3

# Inference steps per model — use the model's native step count so the adversary
# gets the best possible edit quality (makes disruption harder to claim).
MODEL_INFERENCE_STEPS = {
    "sd15": 20,
    "flux_schnell": 4,   # distilled; >4 steps degrades quality
    "flux_dev": 20,
    "sd35": 20,
}


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = (t.float().squeeze(0).cpu().clamp(-1, 1) + 1) / 2
    return TF.to_pil_image(t)


def jpeg_compress(img_t: torch.Tensor, quality: int) -> torch.Tensor:
    """Apply JPEG compression to (1,3,H,W) tensor in [-1,1]. Returns same shape."""
    pil = tensor_to_pil(img_t)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed = TF.to_tensor(Image.open(buf).convert("RGB"))
    return (compressed.unsqueeze(0) * 2 - 1).to(img_t.device, img_t.dtype)


def load_attack_model(model_type: str):
    """Load and return an attack model by type string."""
    if model_type == "sd15":
        from diffvax.attack import Attack
        m = Attack("runwayml/stable-diffusion-inpainting")
        m.model.enable_attention_slicing()
        return m
    elif model_type == "flux_schnell":
        from diffvax.attack_flux import FluxAttack
        return FluxAttack("black-forest-labs/FLUX.1-schnell", guidance_scale=0.0)
    elif model_type == "flux_dev":
        from diffvax.attack_flux import FluxAttack
        return FluxAttack("black-forest-labs/FLUX.1-dev", guidance_scale=3.5)
    elif model_type == "sd35":
        from diffvax.attack_sd3 import SD3Attack
        return SD3Attack("stabilityai/stable-diffusion-3.5-medium")
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")


def run_edit(
    attack_model,
    image_t: torch.Tensor,
    mask_t: torch.Tensor,
    prompt: str,
    model_type: str = "sd15",
):
    """Run editing attack and return output tensor (cpu, float)."""
    n_steps = MODEL_INFERENCE_STEPS.get(model_type, 20)
    with torch.no_grad():
        edited = attack_model.attack(
            prompt=[prompt],
            masked_image=image_t.half().cuda(),
            mask=mask_t.half().cuda(),
            height=RESOLUTION,
            width=RESOLUTION,
            num_inference_steps=n_steps,
            batch_size=1,
        )
    return edited.float().cpu()


def apply_immunization(
    model: torch.nn.Module,
    image_t: torch.Tensor,
    mask_t: torch.Tensor,
    jpeg_quality: Optional[int] = None,
) -> torch.Tensor:
    """Apply NestedUNet immunization. Optionally apply JPEG compression after."""
    img_f = image_t.float().cuda()
    with torch.no_grad():
        perturb = model(img_f)
    perturb = perturb * (1 - mask_t.float().cuda())
    immunized = torch.clamp(img_f + perturb, -1, 1)

    if jpeg_quality is not None:
        immunized = jpeg_compress(immunized, quality=jpeg_quality)

    return immunized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints", nargs="+",
        help="name=path pairs, e.g. sd15_only=path/sd15.pth multimodel=path/mm.pth",
    )
    parser.add_argument(
        "--eval-models", nargs="+", default=["sd15", "flux_schnell", "sd35"],
        help="Models to evaluate transfer on",
    )
    parser.add_argument(
        "--jpeg-qualities", nargs="*", type=int, default=[],
        help="Also evaluate post-JPEG EDR at these quality levels (H7 protocol). "
             "E.g. --jpeg-qualities 75 70 simulates Instagram and Twitter.",
    )
    parser.add_argument("--data-dir", default="../../../../data")
    parser.add_argument("--output-dir", default="results/")
    parser.add_argument("--n-images", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {}
    for spec in (args.checkpoints or []):
        name, path = spec.split("=", 1)
        checkpoints[name] = Path(path)

    if not checkpoints:
        parser.error("Provide at least one --checkpoints name=path")

    # jpeg_modes: None = clean, integer = post-JPEG at that quality
    jpeg_modes = [None] + args.jpeg_qualities

    _, val_list = get_train_val_image_prompt_list(args.data_dir)
    val_list = val_list[:args.n_images]
    data_path = Path(args.data_dir)

    results = []

    for eval_model_type in args.eval_models:
        print(f"\nLoading eval model: {eval_model_type}")
        attack_model = load_attack_model(eval_model_type)

        for ckpt_name, ckpt_path in checkpoints.items():
            print(f"  Checkpoint: {ckpt_name}")
            model = NestedUNet(num_classes=3).cuda()
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            # IMPORTANT: keep model in train() mode for BN consistency.
            # DiffVax was trained with batch_size=1; BN running stats accumulated in
            # train mode. In eval mode, those stats produce near-zero activations (78x
            # weaker signal). Using train() + torch.no_grad() matches training inference.
            model.train()

            pbar = tqdm(val_list, desc=f"{ckpt_name} -> {eval_model_type}")
            for item in pbar:
                image_name = item["image"]
                prompts = item["prompts"][:PROMPTS_PER_IMAGE]

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

                # Precompute clean edits (shared across jpeg_modes and checkpoints)
                clean_edits = {}
                for prompt in prompts:
                    clean_edits[prompt] = run_edit(attack_model, image_t, mask_t, prompt, eval_model_type)
                torch.cuda.empty_cache()

                for jpeg_quality in jpeg_modes:
                    # Apply immunization (optionally JPEG-compressed)
                    immunized_t = apply_immunization(model, image_t, mask_t, jpeg_quality)

                    psnr_val = _psnr(immunized_t.float().cpu(), image_t.float().cpu())
                    ssim_imm_orig = _ssim(immunized_t.float().cpu(), image_t.float().cpu())

                    for prompt in prompts:
                        edited_clean = clean_edits[prompt]
                        edited_imm = run_edit(attack_model, immunized_t.half(), mask_t, prompt, eval_model_type)
                        torch.cuda.empty_cache()

                        ssim_clean_edit = _ssim(
                            edited_clean.cuda().float(), image_t.float()
                        )
                        ssim_imm_edit = _ssim(
                            edited_imm.cuda().float(), image_t.float()
                        )
                        disrupted = int(ssim_imm_edit < ssim_clean_edit - 0.05)

                        results.append({
                            "checkpoint": ckpt_name,
                            "eval_model": eval_model_type,
                            "jpeg_quality": jpeg_quality if jpeg_quality is not None else "none",
                            "image": image_name,
                            "prompt": prompt,
                            "psnr_immunized": round(psnr_val, 3),
                            "ssim_imm_orig": round(ssim_imm_orig, 4),
                            "ssim_clean_edit": round(ssim_clean_edit, 4),
                            "ssim_imm_edit": round(ssim_imm_edit, 4),
                            "disrupted": disrupted,
                        })

                del immunized_t
                torch.cuda.empty_cache()

    # Write CSV
    csv_path = out_dir / "transfer_edr_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Summary table
    print("\n=== Transfer + JPEG Robustness Summary ===")
    print(f"{'Checkpoint':20s} | {'Eval Model':15s} | {'JPEG':5s} | EDR   | PSNR  | Imm SSIM")
    print("-" * 78)
    summary = defaultdict(list)
    for r in results:
        key = (r["checkpoint"], r["eval_model"], r["jpeg_quality"])
        summary[key].append(r)

    for (ckpt, mdl, jpeg_q), rows in sorted(summary.items()):
        edr = sum(r["disrupted"] for r in rows) / len(rows)
        psnr = sum(r["psnr_immunized"] for r in rows) / len(rows)
        ssim = sum(r["ssim_imm_orig"] for r in rows) / len(rows)
        jpeg_str = str(jpeg_q) if jpeg_q != "none" else "  —  "
        print(f"{ckpt:20s} | {mdl:15s} | {jpeg_str:5s} | {edr:.3f} | {psnr:.1f} | {ssim:.4f}")

    print(f"\nFull results: {csv_path}")
    if args.jpeg_qualities:
        print(f"\nH7 interpretation: compare EDR at jpeg_quality=none vs {args.jpeg_qualities}")
        print("  H7 prediction: h7_jpeg checkpoint maintains EDR ≥ 0.7 at q=75; h1a drops to ≤ 0.5")


if __name__ == "__main__":
    main()
