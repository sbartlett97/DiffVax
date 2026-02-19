"""
Adds a new 'img2img_prompts' field to each line in metadata.jsonl files,
randomly sampled from a diverse pool of urban scene editing prompts.
"""

import json
import random
from pathlib import Path

IMG2IMG_PROMPT_POOL = [
    # Time of day / lighting
    "A person standing on a city street at golden hour",
    "A person standing in an urban scene at night with neon lights",
    "A person on a city sidewalk in overcast midday light",
    "A person in an urban environment at dusk with warm ambient lighting",
    "A person standing downtown in harsh midday sun casting sharp shadows",
    "A person on a city street in the soft light of early morning",
    # Weather
    "A person standing in an urban scene in light rain with wet pavement reflections",
    "A person on a city street during a snowstorm",
    "A person standing downtown on a foggy morning",
    "A person in an urban environment on a clear sunny day",
    "A person on a city sidewalk under heavy overcast skies",
    # Season
    "A person standing on a city street in autumn with fallen leaves",
    "A person in an urban scene in summer with lush green trees",
    "A person standing downtown in winter with snow on the ground",
    "A person on a city sidewalk in spring with blooming trees",
    # Style / rendering
    "A person standing in an urban scene, cinematic film photography style",
    "A person on a city street, moody street photography with high contrast",
    "A person in an urban environment, vibrant editorial fashion photography",
    "A person standing downtown, documentary photojournalism style",
    "A person on a city sidewalk, soft bokeh background",
    "A person in an urban scene, drone aerial perspective",
    "A person standing in a city, wide-angle architectural photography",
    # Scene type
    "A person standing near a busy intersection in a modern city",
    "A person on an empty urban side street at night",
    "A person standing in front of a graffiti-covered wall in a city",
    "A person in a crowded city plaza surrounded by pedestrians",
    "A person standing near a subway entrance in a metropolitan area",
    "A person on an elevated walkway overlooking a cityscape",
    "A person standing on a rooftop with a city skyline behind them",
    "A person in an underpass with dramatic urban architecture",
    "A person standing near a city park with skyscrapers in the background",
    "A person on a rain-slicked urban alleyway",
    "A person standing at a crosswalk in a busy downtown district",
    "A person near a construction site in a dense urban area",
    # Artistic / color grade
    "A person in an urban scene with a teal and orange color grade",
    "A person on a city street with a desaturated moody color palette",
    "A person standing in a city, vibrant saturated colors",
    "A person in an urban environment with a vintage faded film look",
    "A person on a city sidewalk, high-key bright urban photography",
    "A person in an urban scene, noir black-and-white style",
]

NUM_PROMPTS_PER_SAMPLE = 2


def add_img2img_prompts(jsonl_path: Path, num_prompts: int = NUM_PROMPTS_PER_SAMPLE) -> int:
    lines = jsonl_path.read_text().splitlines()
    updated = []
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        record["flux_prompts"] = random.sample(IMG2IMG_PROMPT_POOL, k=num_prompts)
        updated.append(json.dumps(record))

    jsonl_path.write_text("\n".join(updated) + "\n")
    return len(updated)


def main():
    data_root = Path(__file__).parent / "data"
    jsonl_files = list(data_root.rglob("metadata.jsonl"))

    if not jsonl_files:
        print("No metadata.jsonl files found.")
        return

    for path in jsonl_files:
        count = add_img2img_prompts(path)
        print(f"Updated {count} records in {path.relative_to(data_root.parent)}")


if __name__ == "__main__":
    main()
