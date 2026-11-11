#!/usr/bin/env python3
"""Rewrite the scene-DESCRIPTION flux_prompts into edit INSTRUCTIONS.

About 64% of this dataset's flux_prompts read "A person on a rain-slicked urban
alleyway": a caption, not an instruction. FLUX.2 Klein, FLUX.1 Kontext and
Qwen-Image-Edit are instruction editors, so a caption leaves the edit
ambiguous (they may barely change anything or re-imagine the whole image). The
other ~36% are already instructions ("Turn this into a pencil sketch") and are
left untouched. Each caption is mapped to an instruction that says what to
change and what to keep, which is how people actually prompt these editors.

    python scripts/rewrite_edit_prompts.py --data-dir data            # dry run
    python scripts/rewrite_edit_prompts.py --data-dir data --write    # backs up, then rewrites

The original list is kept in each row's `flux_prompts_orig`. `prompts` (used by
the SD 1.5 / SD3.5 surrogates) is not touched.
"""
import argparse
import json
import shutil
from pathlib import Path

KEEP = "keeping the person, their pose and their clothing unchanged"

# caption -> instruction (all 40 distinct captions in the dataset)
SETTING = {
    "A person in an urban environment on a clear sunny day": "Change the scene to a clear sunny day",
    "A person in an underpass with dramatic urban architecture": "Move the person into a city underpass with dramatic concrete architecture",
    "A person on a city sidewalk in overcast midday light": "Change the lighting to flat overcast midday light",
    "A person in an urban environment at dusk with warm ambient lighting": "Change the scene to dusk with warm ambient lighting",
    "A person standing downtown in harsh midday sun casting sharp shadows": "Change the lighting to harsh midday sun with sharp cast shadows",
    "A person standing on a rooftop with a city skyline behind them": "Move the person onto a rooftop with a city skyline behind them",
    "A person on a rain-slicked urban alleyway": "Move the person into a rain-slicked urban alleyway",
    "A person standing in front of a graffiti-covered wall in a city": "Change the background to a graffiti-covered city wall",
    "A person in a crowded city plaza surrounded by pedestrians": "Move the person into a crowded city plaza with pedestrians around them",
    "A person on a city street in the soft light of early morning": "Change the lighting to the soft light of early morning",
    "A person on a city sidewalk in spring with blooming trees": "Change the season to spring with blooming trees along the sidewalk",
    "A person standing downtown on a foggy morning": "Make it a foggy morning",
    "A person standing in an urban scene in light rain with wet pavement reflections": "Add light rain and wet pavement with reflections",
    "A person standing near a subway entrance in a metropolitan area": "Move the person next to a subway entrance",
    "A person in an urban scene in summer with lush green trees": "Change the season to summer with lush green trees",
    "A person on a city sidewalk under heavy overcast skies": "Change the sky to heavy overcast clouds",
    "A person standing near a city park with skyscrapers in the background": "Change the background to a city park with skyscrapers behind it",
    "A person on an empty urban side street at night": "Change the scene to an empty side street at night",
    "A person standing at a crosswalk in a busy downtown district": "Move the person to a crosswalk in a busy downtown district",
    "A person near a construction site in a dense urban area": "Change the background to a construction site in a dense city area",
    "A person standing in a city, vibrant saturated colors": "Make the colors vibrant and highly saturated",
    "A person standing downtown in winter with snow on the ground": "Change the season to winter with snow on the ground",
    "A person standing on a city street in autumn with fallen leaves": "Change the season to autumn with fallen leaves on the street",
    "A person standing near a busy intersection in a modern city": "Move the person next to a busy intersection in a modern city",
    "A person standing in an urban scene at night with neon lights": "Change the scene to night with glowing neon lights",
    "A person on a city street during a snowstorm": "Add a snowstorm with falling snow",
    "A person on an elevated walkway overlooking a cityscape": "Move the person onto an elevated walkway overlooking the city",
    "A person on a city sidewalk, soft bokeh background": "Blur the background into soft bokeh",
    "A person standing on a city street at golden hour": "Change the lighting to golden hour",
}
STYLE = {
    "A person on a city sidewalk, high-key bright urban photography": "Re-photograph this as high-key bright urban photography",
    "A person standing in a city, wide-angle architectural photography": "Re-photograph this as wide-angle architectural photography",
    "A person in an urban scene, noir black-and-white style": "Make this a noir black-and-white photograph",
    "A person in an urban environment, vibrant editorial fashion photography": "Make this look like vibrant editorial fashion photography",
    "A person standing downtown, documentary photojournalism style": "Make this look like documentary photojournalism",
    "A person standing in an urban scene, cinematic film photography style": "Give this a cinematic film photography look",
    "A person in an urban scene with a teal and orange color grade": "Apply a teal and orange color grade",
    "A person on a city street with a desaturated moody color palette": "Apply a desaturated, moody color palette",
    "A person in an urban scene, drone aerial perspective": "Change the camera angle to a drone aerial perspective",
    "A person on a city street, moody street photography with high contrast": "Make this moody high-contrast street photography",
    "A person in an urban environment with a vintage faded film look": "Give this a vintage faded film look",
}


def rewrite(prompt: str) -> str:
    if prompt in SETTING:
        return f"{SETTING[prompt]}, {KEEP}"
    if prompt in STYLE:
        return f"{STYLE[prompt]}, {KEEP}"
    return prompt   # already an instruction (or unknown): leave it alone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    for split in ("train", "validation"):
        path = Path(args.data_dir) / split / "metadata.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        changed = unknown = 0
        for r in rows:
            r.setdefault("flux_prompts_orig", list(r["flux_prompts"]))
            new = [rewrite(p) for p in r["flux_prompts_orig"]]
            changed += sum(a != b for a, b in zip(new, r["flux_prompts_orig"]))
            unknown += sum(p.startswith("A person") and p not in SETTING and p not in STYLE
                           for p in r["flux_prompts_orig"])
            r["flux_prompts"] = new
        print(f"{split}: {len(rows)} rows, {changed} prompts rewritten, {unknown} unmapped captions")
        if args.write:
            bak = path.with_suffix(".jsonl.bak_prompts")
            if not bak.exists():
                shutil.copy2(path, bak)
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if not args.write:
        print("dry run; pass --write to apply")


if __name__ == "__main__":
    main()
