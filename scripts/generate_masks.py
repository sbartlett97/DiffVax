#!/usr/bin/env python3
"""Generate human-subject masks for DiffVax training data.

Uses a SegFormer human-parsing model (mattmdjaga/segformer_b2_clothes) to
produce pixel-level body-part segmentation, then derives multiple mask
variants from the single label map:

  - person:     full-body silhouette (all body-part labels)
  - face:       face region only
  - head:       face + hair + hat + sunglasses
  - upper_body: head + upper clothes + arms + scarf

Usage:
    python scripts/generate_masks.py --src /path/to/jpgs --dst /path/to/output
    python scripts/generate_masks.py --src imgs/ --dst data_masked --split 0.8
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

MODEL_ID = "mattmdjaga/segformer_b2_clothes"

# Label indices from the model's id2label mapping:
#  0: Background   1: Hat          2: Hair         3: Sunglasses
#  4: Upper-clothes 5: Skirt       6: Pants        7: Dress
#  8: Belt         9: Left-shoe   10: Right-shoe  11: Face
# 12: Left-leg    13: Right-leg   14: Left-arm    15: Right-arm
# 16: Bag         17: Scarf

PERSON_LABELS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17}
FACE_LABELS = {11}
HEAD_LABELS = {1, 2, 3, 11}  # hat, hair, sunglasses, face
UPPER_BODY_LABELS = {1, 2, 3, 4, 7, 8, 11, 14, 15, 17}  # head + torso + arms

MASK_DEFS = {
    "person": PERSON_LABELS,
    "face": FACE_LABELS,
    "head": HEAD_LABELS,
    "upper_body": UPPER_BODY_LABELS,
}

PROMPT_POOL_FILE = Path(__file__).parent / "prompt_pools.json"
PROMPTS_PER_IMAGE = 2


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_model(device):
    print(f"Loading {MODEL_ID} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model.to(device).eval()
    return processor, model


def get_seg_map(image, processor, model, device, conf_threshold=0.8):
    """Run SegFormer and return the full-resolution label map (H, W) ndarray.

    Pixels where the model's confidence (softmax probability) is below
    *conf_threshold* are mapped to 0 (Background).
    """
    w, h = image.size
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits  # (1, C, h', w')

    # Upsample to original resolution
    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    probs = torch.softmax(logits, dim=1)  # (1, C, H, W)
    confidence, seg_map = probs.max(dim=1)  # (1, H, W) each

    seg_map = seg_map.squeeze(0).cpu().numpy()
    confidence = confidence.squeeze(0).cpu().numpy()

    # Zero out low-confidence pixels (treat as background)
    seg_map[confidence < conf_threshold] = 0
    return seg_map


def masks_from_seg_map(seg_map):
    """Derive all mask variants from the label map.

    Convention: 255 = background, 0 = foreground (region of interest).
    This matches the existing DiffVax dataset format.
    """
    masks = {}
    for name, label_ids in MASK_DEFS.items():
        foreground = np.isin(seg_map, list(label_ids))
        # Light morphological cleanup
        fg_u8 = foreground.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, kernel)
        # Invert: background=255, foreground=0
        masks[name] = 255 - fg_u8
    return masks


def process_image(image_path, processor, model, device):
    """Return dict of {mask_type: ndarray} for one image."""
    image = Image.open(image_path).convert("RGB")
    seg_map = get_seg_map(image, processor, model, device)
    return masks_from_seg_map(seg_map)


def load_prompt_pools():
    """Load SD inpainting and FLUX i2i prompt pools from JSON."""
    with open(PROMPT_POOL_FILE) as f:
        pools = json.load(f)
    sd = pools["sd_prompts"]
    flux = pools["flux_prompts"]
    print(f"Loaded prompt pools: {len(sd)} SD, {len(flux)} FLUX")
    return sd, flux


def sample_prompts(sd_pool, flux_pool, n=PROMPTS_PER_IMAGE):
    """Return (sd_prompts, flux_prompts) — n of each, no duplicates."""
    return random.sample(sd_pool, n), random.sample(flux_pool, n)


def main():
    parser = argparse.ArgumentParser(
        description="Generate human-subject masks for DiffVax training"
    )
    parser.add_argument("--src", required=True, help="Source directory containing .jpg images")
    parser.add_argument("--dst", required=True, help="Output dataset directory")
    parser.add_argument("--split", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split")
    parser.add_argument("--size", type=int, default=512, help="Output image (and mask) size in pixels (default: 512)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Discover source images
    src = Path(args.src)
    images = sorted(src.glob("*.jpg"))
    if not images:
        print(f"No .jpg files found in {src}")
        return

    print(f"Found {len(images)} images in {src}")

    # 80/20 split
    random.shuffle(images)
    n_train = int(len(images) * args.split)
    splits = {
        "train": images[:n_train],
        "eval": images[n_train:],
    }
    print(f"Split: train={len(splits['train'])}, eval={len(splits['eval'])}")

    # Output dirs
    dst = Path(args.dst)
    for name in splits:
        (dst / name / "images").mkdir(parents=True, exist_ok=True)
        (dst / name / "masks").mkdir(parents=True, exist_ok=True)

    # Model & prompts
    device = setup_device()
    processor, model = load_model(device)
    sd_pool, flux_pool = load_prompt_pools()

    # Process each split
    for split_name, split_images in splits.items():
        metadata = []

        for img_path in tqdm(split_images, desc=split_name):
            stem = img_path.stem

            try:
                masks = process_image(img_path, processor, model, device)
            except Exception as e:
                print(f"\nSkipping {img_path.name}: {e}")
                continue

            # Save source image as .png, resized to target size
            out_name = stem + ".png"
            src_img = Image.open(img_path).convert("RGB")
            if args.size != src_img.width or args.size != src_img.height:
                src_img = src_img.resize((args.size, args.size), Image.LANCZOS)
            src_img.save(dst / split_name / "images" / out_name)

            # Save each mask type (skip pure-white masks with no foreground)
            mask_paths = {}
            available = []
            for mtype, mdata in masks.items():
                if (mdata == 255).all():
                    continue
                fname = f"{mtype}_{stem}.png"
                mask_img = Image.fromarray(mdata)
                if args.size != mask_img.width or args.size != mask_img.height:
                    mask_img = mask_img.resize((args.size, args.size), Image.NEAREST)
                mask_img.save(dst / split_name / "masks" / fname)
                mask_paths[mtype] = f"masks/{fname}"
                available.append(mtype)

            # Sample editing prompts
            sd_prompts, flux_prompts = sample_prompts(sd_pool, flux_pool)

            entry = {
                "file_name": f"images/{out_name}",
                "mask": mask_paths.get("person", ""),
                "masks": mask_paths,
                "prompts": sd_prompts,
                "flux_prompts": flux_prompts,
                "mask_types_available": available,
                "is_validation": split_name == "eval",
            }
            metadata.append(entry)

        # Write metadata.jsonl
        meta_path = dst / split_name / "metadata.jsonl"
        with open(meta_path, "w") as f:
            for entry in metadata:
                f.write(json.dumps(entry) + "\n")
        print(f"  Wrote {len(metadata)} entries -> {meta_path}")

    print("Done!")


if __name__ == "__main__":
    main()
