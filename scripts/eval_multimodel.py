#!/usr/bin/env python3
"""
Evaluate a DiffVax checkpoint against a zoo of editing models.

Loads the perturbation net, immunizes evaluation images, then runs each attack
model sequentially (one on GPU at a time), collecting metrics and visual
comparisons.

Usage:
    # Quick smoke test with one model
    python scripts/eval_multimodel.py --checkpoint ckpt.pth \
        --models "SD 1.5 Inpainting" --max-images 2

    # Full evaluation
    python scripts/eval_multimodel.py --checkpoint ckpt.pth

    # Specific mask type and models
    python scripts/eval_multimodel.py --checkpoint ckpt.pth \
        --mask-type face --models "SD 1.5 Inpainting" "SDXL img2img"

    # Checkpoint pushed to the Hugging Face Hub (e.g. via train.py's hub.enabled)
    python scripts/eval_multimodel.py --checkpoint username/diffvax-run
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, "src"))

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from diffvax.utils import (
    set_seed_lib,
    recover_image,
    resolve_device,
    resolve_dtype,
    empty_cache,
    make_generator,
    load_perturbation_net,
    immunize_image_pil,
)
from diffvax.metrics import MetricType, create_metric

to_pil = T.ToPILImage()

# ---------------------------------------------------------------------------
# Model zoo configuration
# ---------------------------------------------------------------------------


class PipelineType(Enum):
    SD_INPAINTING = "sd_inpainting"
    SD_IMG2IMG = "sd_img2img"
    SDXL_IMG2IMG = "sdxl_img2img"
    SD3_IMG2IMG = "sd3_img2img"
    FLUX_KLEIN = "flux_klein"
    FLUX_IMG2IMG = "flux_img2img"


@dataclass
class ModelConfig:
    name: str
    model_id: str
    pipeline_type: PipelineType
    native_resolution: int
    num_inference_steps: int | None = None  # None = use pipeline default
    guidance_scale: float | None = None     # None = use pipeline default
    strength: float | None = None           # None = use pipeline default
    uses_mask: bool = True
    prompt_key: str = "prompts"  # "prompts" for SD, "flux_prompts" for FLUX
    enabled: bool = True


DEFAULT_ZOO: list[ModelConfig] = [
    ModelConfig(
        name="SD 1.5 Inpainting",
        model_id="runwayml/stable-diffusion-inpainting",
        pipeline_type=PipelineType.SD_INPAINTING,
        native_resolution=1024,
    ),
    ModelConfig(
        name="SD 2.0 Inpainting",
        model_id="sd2-community/stable-diffusion-2-inpainting",
        pipeline_type=PipelineType.SD_INPAINTING,
        native_resolution=1024,
    ),
    ModelConfig(
        name="SD 1.5 img2img",
        model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        pipeline_type=PipelineType.SD_IMG2IMG,
        native_resolution=1024,
        # strength=1.0 destroys the input image (pure noise); use 0.75 so the
        # perturbation can meaningfully condition the output.
        strength=0.75,
        uses_mask=False,
    ),
    ModelConfig(
        name="SDXL img2img",
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        pipeline_type=PipelineType.SDXL_IMG2IMG,
        native_resolution=1024,
        strength=0.75,
        uses_mask=False,
        prompt_key="flux_prompts",
    ),
    ModelConfig(
        name="SD3.5 Medium",
        model_id="stabilityai/stable-diffusion-3.5-medium",
        pipeline_type=PipelineType.SD3_IMG2IMG,
        # Matches SD3Attack.native_resolution in src/diffvax/sd3_attack.py.
        native_resolution=1024,
        # Same reasoning as SD 1.5 img2img above: strength=1.0 destroys the
        # input image (pure noise); 0.75 lets the perturbation meaningfully
        # condition the output.
        strength=0.75,
        uses_mask=False,
    ),
    ModelConfig(
        name="FLUX.2 Klein",
        model_id="black-forest-labs/FLUX.2-klein-4B",
        pipeline_type=PipelineType.FLUX_KLEIN,
        native_resolution=1088,
        uses_mask=False,
        prompt_key="flux_prompts",
    ),
    ModelConfig(
        name="FLUX.1 Dev",
        model_id="black-forest-labs/FLUX.1-dev",
        pipeline_type=PipelineType.FLUX_IMG2IMG,
        native_resolution=1088,
        strength=0.75,
        uses_mask=False,
        prompt_key="flux_prompts",
    ),
    ModelConfig(
        name="FLUX.2 Dev",
        model_id="black-forest-labs/FLUX.2-dev",
        pipeline_type=PipelineType.FLUX_IMG2IMG,
        native_resolution=1088,
        strength=0.75,
        uses_mask=False,
        prompt_key="flux_prompts",
        enabled=False,
    ),
]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


@dataclass
class ImageEntry:
    image_path: str
    mask_path: str
    sd_prompts: list[str]
    flux_prompts: list[str]
    name: str


def load_eval_entries(
    data_dir: str,
    mask_type: str,
    max_images: int | None = None,
    split: str | None = None,
) -> tuple[list[ImageEntry], int]:
    """Load evaluation entries from metadata.jsonl.

    Returns (entries, resolution) where resolution is auto-detected from
    the first image.
    """
    base = Path(data_dir)

    # Auto-detect split directory
    if split:
        split_dir = base / split
    elif (base / "eval").exists():
        split_dir = base / "eval"
    elif (base / "validation").exists():
        split_dir = base / "validation"
    else:
        raise FileNotFoundError(
            f"No eval or validation split found in {data_dir}"
        )

    meta_path = split_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found at {meta_path}")

    entries = []
    with meta_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)

            img_filename = Path(row["file_name"]).name
            stem = Path(img_filename).stem
            img_path = str(split_dir / "images" / img_filename)

            # Resolve mask path: new multi-mask format or legacy format
            if "masks" in row and mask_type in row["masks"]:
                mask_rel = row["masks"][mask_type]
                mask_path = str(split_dir / mask_rel)
            else:
                # Legacy format: mask_image_N.png or mask_type_stem.png
                candidate_new = split_dir / "masks" / f"{mask_type}_{stem}.png"
                candidate_legacy = split_dir / "masks" / f"mask_{stem}.png"
                if candidate_new.exists():
                    mask_path = str(candidate_new)
                elif candidate_legacy.exists():
                    mask_path = str(candidate_legacy)
                else:
                    # Use the mask field from metadata
                    mask_path = str(split_dir / row["mask"])

            sd_prompts = row.get("prompts", [])
            flux_prompts = row.get("flux_prompts", sd_prompts)

            entries.append(
                ImageEntry(
                    image_path=img_path,
                    mask_path=mask_path,
                    sd_prompts=sd_prompts,
                    flux_prompts=flux_prompts,
                    name=stem,
                )
            )

    if max_images and len(entries) > max_images:
        entries = entries[:max_images]

    # Auto-detect resolution from first image
    first_img = Image.open(entries[0].image_path)
    resolution = first_img.size[0]  # Assume square
    first_img.close()

    print(f"Loaded {len(entries)} images from {split_dir.name}/ (resolution={resolution})")
    return entries, resolution


# ---------------------------------------------------------------------------
# Perturbation net loading & immunization
# ---------------------------------------------------------------------------
#
# load_perturbation_net() and immunize_image_pil() now live in diffvax.utils
# (shared with scripts/evaluate.py) — load_perturbation_net() also accepts a
# Hugging Face Hub repo id in addition to a local .pth/save_pretrained() path.


# ---------------------------------------------------------------------------
# Pipeline creation & running
# ---------------------------------------------------------------------------


def create_pipeline(config: ModelConfig):
    """Create and load a diffusion pipeline to the best available device."""
    device = resolve_device()
    dtype = resolve_dtype(device)

    if config.pipeline_type == PipelineType.SD_INPAINTING:
        from diffusers import StableDiffusionInpaintPipeline

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            config.model_id, torch_dtype=dtype, safety_checker=None
        )
    elif config.pipeline_type == PipelineType.SD_IMG2IMG:
        from diffusers import StableDiffusionImg2ImgPipeline

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            config.model_id, torch_dtype=dtype, safety_checker=None
        )
    elif config.pipeline_type == PipelineType.SDXL_IMG2IMG:
        from diffusers import StableDiffusionXLImg2ImgPipeline

        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            config.model_id, torch_dtype=dtype, variant="fp16"
        )
    elif config.pipeline_type == PipelineType.SD3_IMG2IMG:
        from diffusers import StableDiffusion3Img2ImgPipeline

        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            config.model_id, torch_dtype=dtype
        )
    elif config.pipeline_type == PipelineType.FLUX_KLEIN:
        from diffusers import Flux2KleinPipeline

        pipe = Flux2KleinPipeline.from_pretrained(
            config.model_id, torch_dtype=dtype
        )
    elif config.pipeline_type == PipelineType.FLUX_IMG2IMG:
        from diffusers import FluxImg2ImgPipeline

        pipe = FluxImg2ImgPipeline.from_pretrained(
            config.model_id, torch_dtype=dtype
        )
    else:
        raise ValueError(f"Unknown pipeline type: {config.pipeline_type}")

    pipe = pipe.to(device)
    return pipe


def unload_pipeline(pipe):
    """Move pipeline off the accelerator and free memory."""
    pipe.to("cpu")
    del pipe
    empty_cache()
    gc.collect()


def run_model(
    pipe,
    config: ModelConfig,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    seed: int,
    dataset_resolution: int,
) -> Image.Image:
    """Run a single model on an image and return the edited result.

    Handles resolution scaling and pipeline-specific call signatures.
    """
    native = config.native_resolution
    need_scale = native != dataset_resolution

    # Prepare image at native resolution
    if need_scale:
        img_native = image.resize((native, native), Image.LANCZOS)
        mask_native = mask.resize((native, native), Image.NEAREST)
    else:
        img_native = image
        mask_native = mask

    generator = make_generator(pipe.device, seed)

    # Build kwargs, omitting None values so pipeline defaults are used
    kwargs = {"prompt": prompt, "image": img_native, "generator": generator}
    if config.num_inference_steps is not None:
        kwargs["num_inference_steps"] = config.num_inference_steps
    if config.guidance_scale is not None:
        kwargs["guidance_scale"] = config.guidance_scale

    if config.pipeline_type == PipelineType.SD_INPAINTING:
        kwargs["mask_image"] = mask_native
        kwargs["height"] = native
        kwargs["width"] = native
        result = pipe(**kwargs).images[0]
        # Composite with input (recover masked region from result, bg from input)
        result = recover_image(result, img_native, mask_native, background=False)

    elif config.pipeline_type in (
        PipelineType.SD_IMG2IMG,
        PipelineType.SDXL_IMG2IMG,
        PipelineType.SD3_IMG2IMG,
        PipelineType.FLUX_IMG2IMG,
    ):
        if config.strength is not None:
            kwargs["strength"] = config.strength
        result = pipe(**kwargs).images[0]

    elif config.pipeline_type == PipelineType.FLUX_KLEIN:
        # Flux2KleinPipeline: image for img2img, no strength parameter
        result = pipe(**kwargs).images[0]
    else:
        raise ValueError(f"Unknown pipeline type: {config.pipeline_type}")

    # Scale back to dataset resolution
    if need_scale:
        result = result.resize((dataset_resolution, dataset_resolution), Image.LANCZOS)

    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def create_metrics():
    """Create metric instances."""
    return {
        "psnr": create_metric(MetricType.PSNR),
        "ssim": create_metric(MetricType.SSIM),
        "fsim": create_metric(MetricType.FSIM),
        "clip": create_metric(
            MetricType.CLIP, model="ViT-B-32", pretrained_on="laion2b_s34b_b79k"
        ),
    }


def compute_image_metrics(
    metrics: dict,
    original: Image.Image,
    immunized: Image.Image,
    edited_orig: Image.Image,
    edited_imm: Image.Image,
    prompt: str,
) -> dict:
    """Compute all metrics for one image-model pair.

    Edit-quality metrics are computed over the WHOLE edited image, for every
    model uniformly — including inpainting models, where run_model() has
    already composited the generated hole back into the original background
    before this function ever sees it. (A previous "extract mask region"
    step here was a no-op — it called recover_image(image, image, mask, ...)
    with the same image as both args, which returns that image unchanged
    regardless of mask content — so this was already the effective behavior;
    this makes it explicit instead of pretending to restrict to a region.)
    """
    edited_orig_np = np.array(edited_orig.convert("RGB"))
    edited_imm_np = np.array(edited_imm.convert("RGB"))

    # Perturbation visibility (full image)
    orig_np = np.array(original.convert("RGB"))
    imm_np = np.array(immunized.convert("RGB"))

    result = {}
    result["orig_vs_imm_psnr"] = float(metrics["psnr"]([orig_np], [imm_np])[0])
    result["orig_vs_imm_ssim"] = float(metrics["ssim"]([orig_np], [imm_np])[0])

    # Edit quality metrics (whole image)
    result["edit_ssim"] = float(
        metrics["ssim"]([edited_orig_np], [edited_imm_np])[0]
    )
    result["edit_psnr"] = float(
        metrics["psnr"]([edited_orig_np], [edited_imm_np])[0]
    )
    result["edit_fsim"] = float(
        metrics["fsim"]([edited_orig_np], [edited_imm_np])[0]
    )

    # CLIP scores (whole image)
    result["clip_no_defense"] = float(
        metrics["clip"]([edited_orig], [prompt])[0]
    )
    result["clip_with_defense"] = float(
        metrics["clip"]([edited_imm], [prompt])[0]
    )
    result["clip_delta"] = result["clip_no_defense"] - result["clip_with_defense"]

    return result


# ---------------------------------------------------------------------------
# Output: table, JSON, visual grid
# ---------------------------------------------------------------------------


def print_summary_table(report: dict):
    """Print a formatted summary table to the terminal."""
    header = (
        f"{'Model':<25} | {'SSIM':>6} | {'PSNR':>6} | {'FSIM':>6} | "
        f"{'CLIP-Orig':>9} | {'CLIP-Def':>8} | {'CLIP-D':>7}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for model_name, model_data in report["models"].items():
        s = model_data["summary"]
        print(
            f"{model_name:<25} | {s['mean_edit_ssim']:>6.3f} | "
            f"{s['mean_edit_psnr']:>6.1f} | {s['mean_edit_fsim']:>6.3f} | "
            f"{s['mean_clip_no_defense']:>9.1f} | "
            f"{s['mean_clip_with_defense']:>8.1f} | "
            f"{s['mean_clip_delta']:>7.1f}"
        )

    print(sep)

    # Print perturbation invisibility (same across models, take first)
    first_model = next(iter(report["models"].values()))
    s = first_model["summary"]
    print(
        f"\nPerturbation invisibility:  "
        f"PSNR={s['mean_orig_vs_imm_psnr']:.1f}  "
        f"SSIM={s['mean_orig_vs_imm_ssim']:.3f}"
    )


def build_visual_comparison(
    entries: list[ImageEntry],
    immunized_images: list[Image.Image],
    model_results: dict[str, list[tuple[Image.Image, Image.Image]]],
    save_path: str,
    max_rows: int = 5,
):
    """Build a visual comparison grid PNG.

    Columns: Original | Mask | Immunized | [Model_i NoDef | Model_i Def] ...
    Rows: one per image (up to max_rows).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_images = min(len(entries), max_rows)
    model_names = list(model_results.keys())
    n_models = len(model_names)
    # 3 fixed columns + 2 per model
    n_cols = 3 + 2 * n_models

    fig, axes = plt.subplots(
        n_images, n_cols, figsize=(3.5 * n_cols, 4.0 * n_images)
    )
    if n_images == 1:
        axes = axes[None, :]

    # Column labels
    fixed_labels = ["Original", "Mask", "Immunized"]
    model_labels = []
    for name in model_names:
        short = name.replace("Inpainting", "Inp").replace("img2img", "i2i")
        model_labels.extend([f"{short}\nNo Def", f"{short}\nDefended"])

    all_labels = fixed_labels + model_labels

    for col_idx, label in enumerate(all_labels):
        axes[0, col_idx].set_title(label, fontsize=9, fontweight="bold", pad=8)

    for row_idx in range(n_images):
        entry = entries[row_idx]
        orig_pil = Image.open(entry.image_path).convert("RGB")
        mask_pil = Image.open(entry.mask_path).convert("RGB")
        imm_pil = immunized_images[row_idx]

        axes[row_idx, 0].imshow(orig_pil)
        axes[row_idx, 1].imshow(mask_pil)
        axes[row_idx, 2].imshow(imm_pil)

        for model_idx, model_name in enumerate(model_names):
            edited_orig, edited_imm = model_results[model_name][row_idx]
            col_base = 3 + 2 * model_idx
            axes[row_idx, col_base].imshow(edited_orig)
            axes[row_idx, col_base + 1].imshow(edited_imm)

        # Row label
        axes[row_idx, 0].text(
            -0.05, 0.5, entry.name,
            transform=axes[row_idx, 0].transAxes,
            fontsize=7, va="center", ha="right", rotation=90,
        )

    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    plt.subplots_adjust(wspace=0.02, hspace=0.05)
    fig.savefig(save_path, bbox_inches="tight", dpi=150, pad_inches=0.1)
    plt.close(fig)
    print(f"Visual comparison saved to {save_path}")


def save_per_model_images(
    entries: list[ImageEntry],
    immunized_images: list[Image.Image],
    model_name: str,
    edited_pairs: list[tuple[Image.Image, Image.Image]],
    output_dir: str,
):
    """Save individual edited images for one model."""
    model_dir = Path(output_dir) / "images" / model_name.replace(" ", "_").replace(".", "")
    model_dir.mkdir(parents=True, exist_ok=True)

    for i, (entry, (edited_orig, edited_imm)) in enumerate(
        zip(entries, edited_pairs)
    ):
        edited_orig.save(model_dir / f"{entry.name}_no_defense.png")
        edited_imm.save(model_dir / f"{entry.name}_defended.png")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DiffVax checkpoint against a zoo of editing models"
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
        help="Dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(_project_root, "outputs", "eval_report"),
        help="Output directory",
    )
    parser.add_argument(
        "--mask-type", type=str, default="person",
        help="Mask variant: person|face|head|upper_body",
    )
    parser.add_argument(
        "--max-images", type=int, default=None, help="Limit number of eval images"
    )
    parser.add_argument("--seed", type=int, default=5, help="Random seed")
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Only run specific models by name",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override inference steps for all models",
    )
    parser.add_argument(
        "--skip-visual", action="store_true", help="Skip generating comparison PNG"
    )
    parser.add_argument(
        "--mask-gate-perturbation", action="store_true",
        help="Confine the applied perturbation to the subject region (mask==0; "
        "dataset convention 1=background) instead of the whole image. Only use "
        "this for checkpoints actually trained with perturbation_mask_gating: "
        "true — for full-image-trained checkpoints it strips content the "
        "network relied on rather than reflecting how it actually protects.",
    )
    parser.add_argument(
        "--split", type=str, default=None,
        help='Dataset split: "validation", "eval", or auto-detect',
    )
    args = parser.parse_args()

    set_seed_lib(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Select models ----
    zoo = [m for m in DEFAULT_ZOO if m.enabled]
    if args.models:
        requested = {n.lower() for n in args.models}
        zoo = [m for m in DEFAULT_ZOO if m.name.lower() in requested]
        if not zoo:
            all_names = [m.name for m in DEFAULT_ZOO]
            print(f"No matching models. Available: {all_names}")
            sys.exit(1)

    if args.steps:
        for m in zoo:
            m.num_inference_steps = args.steps

    print("=" * 60)
    print("  DiffVax Multi-Model Evaluation")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Models: {[m.name for m in zoo]}")
    print(f"  Mask type: {args.mask_type}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    # ---- Load dataset ----
    entries, resolution = load_eval_entries(
        args.data_dir, args.mask_type, args.max_images, args.split
    )

    # ---- Load perturbation net and immunize all images ----
    print("\nLoading perturbation network...")
    perturbation_net = load_perturbation_net(args.checkpoint)

    print("Immunizing images..." + (
        " (mask-gated to subject region)" if args.mask_gate_perturbation else ""
    ))
    immunized_images = []
    for i, entry in enumerate(entries):
        image_pil = Image.open(entry.image_path).convert("RGB")
        mask_pil = (
            Image.open(entry.mask_path).convert("RGB")
            if args.mask_gate_perturbation else None
        )
        imm_pil = immunize_image_pil(perturbation_net, image_pil, mask_pil=mask_pil)
        immunized_images.append(imm_pil)
        if (i + 1) % 10 == 0 or (i + 1) == len(entries):
            print(f"  Immunized {i + 1}/{len(entries)}")

    # ---- Initialize metrics ----
    metrics = create_metrics()

    # ---- Evaluate each model ----
    report = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "data_dir": args.data_dir,
            "resolution": resolution,
            "seed": args.seed,
            "mask_type": args.mask_type,
            "num_images": len(entries),
        },
        "models": {},
    }
    # Store edited image pairs for visual grid: model_name -> [(edited_orig, edited_imm)]
    all_model_results: dict[str, list[tuple[Image.Image, Image.Image]]] = {}

    for model_cfg in zoo:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_cfg.name}")
        print(f"  Model ID: {model_cfg.model_id}")
        print(f"  Pipeline: {model_cfg.pipeline_type.value}")
        print(f"{'='*60}")

        # Load pipeline
        t0 = time.time()
        print(f"Loading pipeline...")
        try:
            pipe = create_pipeline(model_cfg)
        except Exception as e:
            print(f"  FAILED to load {model_cfg.name}: {e}")
            print(f"  Skipping this model.")
            continue
        print(f"  Pipeline loaded in {time.time() - t0:.1f}s")

        per_image_metrics = {}
        edited_pairs = []

        for img_idx, entry in enumerate(entries):
            image_pil = Image.open(entry.image_path).convert("RGB")
            mask_pil = Image.open(entry.mask_path).convert("RGB")
            imm_pil = immunized_images[img_idx]

            # Ensure all inputs are at dataset resolution
            res = (resolution, resolution)
            if image_pil.size != res:
                image_pil = image_pil.resize(res, Image.LANCZOS)
            if mask_pil.size != res:
                mask_pil = mask_pil.resize(res, Image.NEAREST)
            if imm_pil.size != res:
                imm_pil = imm_pil.resize(res, Image.LANCZOS)

            # Select prompt based on model type
            if model_cfg.prompt_key == "flux_prompts":
                prompts = entry.flux_prompts
            else:
                prompts = entry.sd_prompts
            prompt = prompts[0] if prompts else "a person"

            print(f"  [{img_idx + 1}/{len(entries)}] {entry.name}: \"{prompt[:50]}\"")
            t1 = time.time()

            # Edit original (no defense)
            set_seed_lib(args.seed)
            edited_orig = run_model(
                pipe, model_cfg, image_pil, mask_pil, prompt, args.seed, resolution
            )

            # Edit immunized
            set_seed_lib(args.seed)
            edited_imm = run_model(
                pipe, model_cfg, imm_pil, mask_pil, prompt, args.seed, resolution
            )

            # Ensure outputs match dataset resolution for metrics
            if edited_orig.size != res:
                edited_orig = edited_orig.resize(res, Image.LANCZOS)
            if edited_imm.size != res:
                edited_imm = edited_imm.resize(res, Image.LANCZOS)

            edited_pairs.append((edited_orig, edited_imm))

            # Compute metrics
            img_metrics = compute_image_metrics(
                metrics, image_pil, imm_pil, edited_orig, edited_imm, prompt
            )
            per_image_metrics[entry.name] = img_metrics

            print(
                f"    SSIM={img_metrics['edit_ssim']:.3f}  "
                f"CLIP-D={img_metrics['clip_delta']:.1f}  "
                f"({time.time() - t1:.1f}s)"
            )

            empty_cache()

        # Save per-model images
        save_per_model_images(entries, immunized_images, model_cfg.name, edited_pairs, args.output_dir)

        # Compute summary
        if per_image_metrics:
            metric_keys = list(next(iter(per_image_metrics.values())).keys())
            summary = {}
            for key in metric_keys:
                values = [m[key] for m in per_image_metrics.values()]
                summary[f"mean_{key}"] = float(np.mean(values))
                summary[f"std_{key}"] = float(np.std(values))
        else:
            summary = {}

        report["models"][model_cfg.name] = {
            "config": {
                "model_id": model_cfg.model_id,
                "pipeline_type": model_cfg.pipeline_type.value,
                "native_resolution": model_cfg.native_resolution,
                "num_inference_steps": model_cfg.num_inference_steps or "default",
                "guidance_scale": model_cfg.guidance_scale if model_cfg.guidance_scale is not None else "default",
                "strength": model_cfg.strength if model_cfg.strength is not None else "default",
                "uses_mask": model_cfg.uses_mask,
            },
            "per_image": per_image_metrics,
            "summary": summary,
        }
        all_model_results[model_cfg.name] = edited_pairs

        # Unload pipeline
        print(f"  Unloading {model_cfg.name}...")
        unload_pipeline(pipe)

    # ---- Save JSON report ----
    report_path = os.path.join(args.output_dir, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved to {report_path}")

    # ---- Print summary table ----
    if report["models"]:
        print_summary_table(report)

    # ---- Visual comparison ----
    if not args.skip_visual and report["models"]:
        visual_path = os.path.join(args.output_dir, "visual_comparison.png")
        max_visual = min(len(entries), 5)
        build_visual_comparison(
            entries[:max_visual],
            immunized_images[:max_visual],
            {name: pairs[:max_visual] for name, pairs in all_model_results.items()},
            visual_path,
            max_rows=max_visual,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
