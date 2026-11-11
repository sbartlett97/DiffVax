#!/usr/bin/env python3
"""One-off dataset enrichment: classify each image's subject (age bracket +
gender) via CLIP zero-shot classification, then substitute a specific
descriptor into the generic "A person" prompt template.

Motivation: every prompt in this dataset follows "A person in a {scene}",
with zero subject metadata anywhere — at typical img2img strengths (~0.75),
SD3.5/SD1.5 have no anchor for who's actually in the photo, so edits drift
toward the model's default (adult) bias even when the subject is a young
child. Injecting an accurate subject descriptor gives the edit model
something concrete to condition on, both during training (better-anchored
prompts) and eval (fairer, more interpretable comparisons).

The fix lives entirely in the data (metadata.jsonl's `prompts` field) rather
than new code — every consumer (ImmunizationDataset for training,
eval_multimodel.py / evaluate.py for eval) already just reads that field as
a plain string, so nothing else needs to change.

Only rewrites `prompts` (used by SD 1.5/SD3.5). `flux_prompts` in this
dataset are style-transfer instructions ("Render this in a photorealistic
style"), not scene descriptions — the "a person" pattern doesn't apply
there, and FLUX isn't in use in this training track anyway.

Usage:
    # Dry run (default) — prints a before/after review table, writes nothing
    python scripts/add_subject_descriptors.py --data-dir data

    # Write metadata.jsonl.new next to the original (review before adopting)
    python scripts/add_subject_descriptors.py --data-dir data --write

    # Overwrite metadata.jsonl directly (keeps a one-time .bak of the original)
    python scripts/add_subject_descriptors.py --data-dir data --write --in-place
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, "src"))

from diffvax.utils import resolve_device  # noqa: E402

# Grid spanning both dimensions that visibly drift without an anchor (age,
# gender) rather than age alone — a richer candidate set also gives CLIP's
# zero-shot argmax more to discriminate against, generally improving top-1
# accuracy even for the dimension you care most about.
SUBJECT_CANDIDATES = [
    "a baby boy",
    "a baby girl",
    "a toddler boy",
    "a toddler girl",
    "a young boy",
    "a young girl",
    "a teenage boy",
    "a teenage girl",
    "a young man",
    "a young woman",
    "a middle-aged man",
    "a middle-aged woman",
    "an elderly man",
    "an elderly woman",
]

_PERSON_RE = re.compile(r"\ba\s+person\b", re.IGNORECASE)


def substitute_subject(prompt: str, subject_candidate: str) -> str:
    """Replace the first 'a person'/'A person' with subject_candidate,
    matching the original occurrence's capitalization but using the
    candidate's own grammatically-correct article (a/an) rather than
    blindly reusing whatever article was originally matched.
    """
    def _replace(match):
        replacement = subject_candidate
        if match.group(0)[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]
        return replacement

    return _PERSON_RE.sub(_replace, prompt, count=1)


def crop_to_mask_bbox(image: Image.Image, mask: Image.Image, pad_frac: float = 0.1) -> Image.Image:
    """Crop to the bounding box of the mask's nonzero region (+ small pad),
    so CLIP classifies the subject without irrelevant background diluting
    the signal. Falls back to the full image if the mask is empty.
    """
    mask_np = np.array(mask.convert("L"))
    ys, xs = np.nonzero(mask_np > 127)
    if len(xs) == 0:
        return image
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w, h = image.size
    pad_x = int((x1 - x0) * pad_frac)
    pad_y = int((y1 - y0) * pad_frac)
    x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    x1, y1 = min(w, x1 + pad_x), min(h, y1 + pad_y)
    return image.crop((x0, y0, x1, y1))


class SubjectClassifier:
    """CLIP zero-shot classifier over SUBJECT_CANDIDATES. Same model/weights
    already used elsewhere in this codebase (src/diffvax/metrics/clip_score.py),
    so no new dependency."""

    def __init__(self, model: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.device = resolve_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        tokenizer = open_clip.get_tokenizer(model)
        with torch.no_grad():
            text = tokenizer(SUBJECT_CANDIDATES).to(self.device)
            text_features = self.model.encode_text(text)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def classify(self, image: Image.Image) -> str:
        img_t = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(img_t)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ self.text_features.T).squeeze(0)
        return SUBJECT_CANDIDATES[int(sims.argmax())]


def process_split(split_dir: Path, classifier: SubjectClassifier, write: bool, in_place: bool):
    meta_path = split_dir / "metadata.jsonl"
    if not meta_path.exists():
        print(f"  (skip: no metadata.jsonl in {split_dir})")
        return

    rows = [json.loads(line) for line in meta_path.open() if line.strip()]
    updated = []
    n_changed = 0

    for row in rows:
        image = Image.open(split_dir / row["file_name"]).convert("RGB")

        mask_rel = row.get("masks", {}).get("person") or row.get("mask")
        subject_image = image
        if mask_rel and (split_dir / mask_rel).exists():
            subject_image = crop_to_mask_bbox(image, Image.open(split_dir / mask_rel))

        subject = classifier.classify(subject_image)
        old_prompts = row["flux_prompts"]
        new_prompts = [substitute_subject(p, subject) for p in old_prompts]
        changed = new_prompts != old_prompts
        n_changed += changed

        print(f"{row['file_name']:55} -> {subject:20} {'(changed)' if changed else '(no \"a person\" match)'}")
        for old, new in zip(old_prompts, new_prompts):
            if old != new:
                print(f"    - {old}")
                print(f"    + {new}")

        row["prompts"] = new_prompts
        updated.append(row)

    print(f"  {n_changed}/{len(rows)} prompts updated in {split_dir.name}/")

    if not write:
        return None

    if in_place:
        backup_path = meta_path.parent / (meta_path.name + ".bak")
        if not backup_path.exists():
            backup_path.write_text(meta_path.read_text())
            print(f"  Backed up original to {backup_path}")
        out_path = meta_path
    else:
        out_path = meta_path.parent / (meta_path.name + ".new")
    with out_path.open("w") as f:
        for row in updated:
            f.write(json.dumps(row) + "\n")
    print(f"  Wrote {out_path}")
    return out_path


def push_metadata_to_hub(split_to_path: dict, repo_id: str, private: bool = True):
    """Push updated metadata.jsonl file(s) to a Hugging Face Hub dataset repo.

    Only uploads the specific metadata.jsonl path_in_repo for each split —
    does not touch images/masks or anything else already in the repo.
    Authenticates via the HF_TOKEN env var / cached `hf auth login`, same as
    every other Hub interaction in this codebase.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    for split, local_path in split_to_path.items():
        if local_path is None:
            print(f"  Skipping Hub push for {split}/ — nothing was written locally "
                  f"(pass --write, and --in-place to actually update metadata.jsonl).")
            continue
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"{split}/metadata.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=(
                f"Add subject descriptors (age/gender) to {split} prompts, "
                f"via CLIP zero-shot classification"
            ),
        )
        print(f"  Pushed {split}/metadata.jsonl to {repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", default=os.path.join(_project_root, "data"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument(
        "--write", action="store_true",
        help="Actually write output (default: dry run / preview only, writes nothing)",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Overwrite metadata.jsonl directly (keeps a one-time .bak). "
        "Default (without this flag) writes metadata.jsonl.new instead.",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Push the updated metadata.jsonl for each split to --hub-repo-id "
        "after writing locally. Requires --write --in-place (pushes exactly "
        "what was just written to metadata.jsonl, not a .new sidecar).",
    )
    parser.add_argument(
        "--hub-repo-id", default="sbartlett97/diffvax-high-res",
        help="Hugging Face Hub dataset repo id to push to",
    )
    args = parser.parse_args()

    if args.push_to_hub and not (args.write and args.in_place):
        print("--push-to-hub requires --write --in-place (nothing to push otherwise).")
        sys.exit(1)

    print("Loading CLIP model for subject classification...")
    classifier = SubjectClassifier()

    data_dir = Path(args.data_dir)
    split_to_path = {}
    for split in args.splits:
        split_dir = data_dir / split
        print(f"\n=== {split} ===")
        split_to_path[split] = process_split(
            split_dir, classifier, write=args.write, in_place=args.in_place
        )

    if not args.write:
        print("\nDry run only — nothing written. Re-run with --write to save changes.")
        return

    if args.push_to_hub:
        print(f"\n=== Pushing to {args.hub_repo_id} ===")
        push_metadata_to_hub(split_to_path, args.hub_repo_id)


if __name__ == "__main__":
    main()
