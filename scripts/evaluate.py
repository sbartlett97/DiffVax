"""Calculate image quality metrics for immunization evaluation against SD 1.5.

Standard image-quality + protection metrics: PSNR/SSIM of the perturbation
itself (imperceptibility) and PSNR/SSIM/FSIM/CLIP-score of the SD 1.5
inpainting edit on the immunized image vs. the clean image (protection).

Usage:
    python scripts/evaluate.py --checkpoint <ckpt>.pth
    python scripts/evaluate.py --checkpoint username/diffvax-run --max-images 5
"""

import argparse
import json
import numpy as np
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, "src"))

from diffvax.attack import Attack
from diffvax.metrics import MetricType, create_metric
from diffvax.utils import (
    set_seed_lib,
    recover_image,
    load_image,
    get_train_val_image_prompt_list,
    ensure_dataset_in_data_dir,
    load_perturbation_net,
    immunize_image_pil,
    resolve_device,
    make_generator,
)

SEED = 5


def calculate_metrics_for_image(
    orig_image, adv_image, image_mask, prompt_list, image_name,
    immunization_model, save_path, no_metric=False, sampling_steps=30,
):
    noise_metrics = {}
    edit_metrics = {}
    noise_metrics["psnr"] = create_metric(MetricType.PSNR)
    noise_metrics["ssim"] = create_metric(MetricType.SSIM)
    edit_metrics["psnr"] = create_metric(MetricType.PSNR)
    edit_metrics["ssim"] = create_metric(MetricType.SSIM)
    edit_metrics["fsim"] = create_metric(MetricType.FSIM)
    edit_metrics["clip"] = create_metric(MetricType.CLIP, model='ViT-B-32', pretrained_on='laion2b_s34b_b79k')

    log = {}
    log["image_name"] = image_name
    log["prompt_list"] = prompt_list
    log["immunization_model"] = immunization_model.model_name
    orig_image_np = np.array(orig_image.convert("RGB"))
    print(image_name)
    adv_image_np = np.array(adv_image.convert("RGB"))
    noise_metric_values = {}
    # Difference between the output (image+noise) and the original image: PSNR and SSIM
    psnr_noise = noise_metrics["psnr"]([orig_image_np], [adv_image_np])[0]
    ssim_noise = noise_metrics["ssim"]([orig_image_np], [adv_image_np])[0]
    noise_metric_values["psnr"] = psnr_noise
    noise_metric_values["ssim"] = ssim_noise

    edit_metric_values = {}
    for metric_name in edit_metrics.keys():
        edit_metric_values[metric_name] = {}

    # Difference between the edited output (image+noise) and the edited original image
    for prompt_ind, prompt in enumerate(prompt_list):
        set_seed_lib(SEED)
        edited_adv = immunization_model.edit_image(prompt, adv_image, image_mask, num_inf=sampling_steps)[0]
        set_seed_lib(SEED)
        edited_orig = immunization_model.edit_image(prompt, orig_image, image_mask, num_inf=sampling_steps)[0]

        edited_adv_recovered = recover_image(edited_adv, adv_image, image_mask, background=False)
        edited_orig_recovered = recover_image(edited_orig, orig_image, image_mask, background=False)
        adv_dir = os.path.join(save_path, 'images')
        os.makedirs(adv_dir, exist_ok=True)
        edited_adv_recovered.save(os.path.join(adv_dir, f"{image_name}_prompt_{prompt_ind}_edited_result_adv.png"))
        edited_orig_recovered.save(os.path.join(adv_dir, f"{image_name}_prompt_{prompt_ind}_edited_result_orig.png"))

        if no_metric:
            continue
        # Whole-image metrics on the final composited edit result (generated
        # hole + original background) — the same thing recover_image() above
        # already produces for saving, and what a real viewer would see. A
        # previous "take_background" step here was a no-op (it called
        # recover_image(image, image, mask, ...) with the same image as both
        # args, which always returns that image unchanged regardless of mask
        # content) and was computing metrics on the raw, un-composited model
        # output instead of the actual presented result.
        for metric_name, edit_metric in edit_metrics.items():
            if metric_name == "clip":
                adv_metric = edit_metric([edited_adv_recovered], [prompt])[0]
            else:
                edited_adv_recovered_np = np.array(edited_adv_recovered.convert("RGB"))
                edited_orig_recovered_np = np.array(edited_orig_recovered.convert("RGB"))
                adv_metric = edit_metric([edited_orig_recovered_np], [edited_adv_recovered_np])[0]
            edit_metric_values[metric_name][prompt] = adv_metric

    return noise_metric_values, edit_metric_values


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class _SD15EditEvaluator:
    """Adapter exposing exactly what calculate_metrics_for_image needs
    (.model_name, .edit_image(prompt, image, mask, num_inf=N)) without
    pulling in DiffVaxImmunization's full training-time constructor
    (optimizer, reporter, EoT/loss objects). Mirrors
    DiffVaxImmunization.edit_image()'s exact call to the underlying
    diffusers pipeline.
    """

    def __init__(self, attack_model_link: str, device):
        self.attack = Attack(attack_model_link)
        self.attack.model = self.attack.model.to(device)
        self.model_name = "DiffVaxImmunization"
        self.generator = make_generator(device)

    def edit_image(self, prompt, img, img_mask, num_inf=30, SEED=5):
        self.generator.manual_seed(SEED)
        return self.attack.model(
            prompt=prompt,
            image=img,
            mask_image=img_mask,
            eta=1,
            num_inference_steps=num_inf,
            guidance_scale=7.5,
            strength=1.0,
            generator=self.generator,
        ).images


def _aggregate(dicts, key):
    values = [d[key] for d in dicts if key in d]
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a DiffVax checkpoint against SD 1.5 "
        "(image-quality + protection metrics)"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help=(
            "Local .pth checkpoint, a local save_pretrained() directory, "
            "or a Hugging Face Hub repo id (e.g. 'username/diffvax-run')"
        ),
    )
    parser.add_argument(
        "--data-dir", type=str, default=os.path.join(_project_root, "data"),
        help="Dataset directory (auto-downloads the DiffVax dataset if absent)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(_project_root, "outputs", "evaluate"),
        help="Where to save edited images and the metrics report",
    )
    parser.add_argument(
        "--attack-model-link", type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="SD 1.5 inpainting checkpoint to evaluate protection against",
    )
    parser.add_argument(
        "--mask-type", type=str, default=None,
        help="Mask variant (person|face|head|upper_body); default uses each "
        "image's first available mask type",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--max-images", type=int, default=None, help="Limit number of eval images"
    )
    parser.add_argument(
        "--sampling-steps", type=int, default=30,
        help="Denoising steps for the SD 1.5 edit",
    )
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument(
        "--no-metric", action="store_true",
        help="Save edited images without computing metrics (faster smoke test)",
    )
    parser.add_argument(
        "--mask-gate-perturbation", action="store_true",
        help="Confine the applied perturbation to the subject region (mask==0; "
        "dataset convention 1=background) instead of the whole image. Only use "
        "this for checkpoints actually trained with perturbation_mask_gating: "
        "true — for full-image-trained checkpoints it strips content the "
        "network relied on rather than reflecting how it actually protects.",
    )
    args = parser.parse_args()

    set_seed_lib(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    data_dir = ensure_dataset_in_data_dir(
        repo_id="ozdentarikcan/DiffVaxDataset", data_dir=args.data_dir
    )
    _, val_entries = get_train_val_image_prompt_list(data_dir)
    if args.max_images:
        val_entries = val_entries[: args.max_images]
    print(f"Evaluating {len(val_entries)} validation image(s)")

    device = resolve_device()
    size = (args.resolution, args.resolution)

    print("Loading perturbation network...")
    perturbation_net = load_perturbation_net(args.checkpoint, device=device)

    print(f"Loading SD 1.5 inpainting pipeline ({args.attack_model_link})...")
    evaluator = _SD15EditEvaluator(args.attack_model_link, device)

    report = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "attack_model_link": args.attack_model_link,
            "data_dir": str(data_dir),
            "resolution": args.resolution,
            "seed": args.seed,
            "num_images": len(val_entries),
        },
        "per_image": {},
    }
    all_noise = []
    all_edit_flat = []  # one row per (image, prompt, metric_name) -> value

    for i, entry in enumerate(val_entries):
        image_name = entry["image"][:-4]
        mask_type = args.mask_type or next(
            iter(entry.get("mask_types_available", [])), None
        )
        print(f"[{i + 1}/{len(val_entries)}] {image_name}")

        orig_image = load_image(
            image_name, data_dir, is_mask=False,
            images_subdir="validation/images", masks_subdir="validation/masks",
            size=size, mask_type=mask_type,
        )
        image_mask = load_image(
            image_name, data_dir, is_mask=True,
            images_subdir="validation/images", masks_subdir="validation/masks",
            size=size, mask_type=mask_type,
        )
        adv_image = immunize_image_pil(
            perturbation_net, orig_image, device=device,
            mask_pil=image_mask if args.mask_gate_perturbation else None,
        )

        noise_metrics, edit_metrics = calculate_metrics_for_image(
            orig_image, adv_image, image_mask, entry["prompts"], image_name,
            evaluator, args.output_dir,
            no_metric=args.no_metric, sampling_steps=args.sampling_steps,
        )

        report["per_image"][image_name] = {
            "noise": noise_metrics, "edit": edit_metrics,
        }
        if not args.no_metric:
            all_noise.append(noise_metrics)
            for metric_name, per_prompt in edit_metrics.items():
                for value in per_prompt.values():
                    all_edit_flat.append({"metric": metric_name, "value": value})

    # ---- Save JSON report ----
    report_path = os.path.join(args.output_dir, "evaluate_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved to {report_path}")

    # ---- Print summary ----
    if not args.no_metric and all_noise:
        print("\n" + "=" * 50)
        print("Perturbation invisibility (orig vs. immunized):")
        print(f"  PSNR: {_aggregate(all_noise, 'psnr'):.2f}")
        print(f"  SSIM: {_aggregate(all_noise, 'ssim'):.3f}")

        print("\nEdit protection (edited-orig vs. edited-immunized, masked region):")
        for metric_name in ("ssim", "psnr", "fsim", "clip"):
            values = [
                row["value"] for row in all_edit_flat if row["metric"] == metric_name
            ]
            if values:
                print(f"  {metric_name.upper()}: {float(np.mean(values)):.3f}")
        print("=" * 50)

    print("\nDone!")


if __name__ == "__main__":
    main()
