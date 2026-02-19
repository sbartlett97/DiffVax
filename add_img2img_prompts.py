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
    "A person on a city street in a heavy thunderstorm with puddles on the ground",
    "A person standing in an urban scene during a summer heatwave with heat shimmer",
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
    # Background swap — non-urban / natural
    "A person standing in a lush green forest",
    "A person on a sandy beach with ocean waves behind them",
    "A person standing in an open field under a dramatic cloudy sky",
    "A person standing in a desert landscape at sunset",
    "A person in a snow-covered mountain scene",
    "A person standing beside a calm lake with trees reflected in the water",
    "A person in a meadow with wildflowers in the background",
    "A person standing on rocky coastal cliffs overlooking the sea",
    # Background swap — architectural / interior exterior
    "A person standing in front of a grand neoclassical building",
    "A person in front of a sleek modern glass skyscraper",
    "A person standing at the entrance of an industrial warehouse",
    "A person in front of a colorful row of European townhouses",
    "A person standing outside a neon-lit Japanese street at night",
    "A person in front of an ancient stone ruins site",
    "A person standing at a train station platform with a train arriving",
    "A person outside a vibrant street market with vendor stalls",
    # Adding objects / props to the scene
    "A person standing on a city street next to a vintage red telephone box",
    "A person on a sidewalk beside a row of parked motorcycles",
    "A person standing in an urban scene with colorful street art murals on the walls",
    "A person on a city street with food carts and market stalls nearby",
    "A person standing next to a large illuminated billboard at night",
    "A person in an urban scene with pigeons gathered on the pavement around them",
    "A person standing on a street corner beside a glowing streetlamp",
    "A person on a city street with a yellow taxi cab passing by",
    "A person standing near a fountain in an urban plaza",
    "A person on a sidewalk with flower planters and café tables nearby",
    "A person standing in a street with hanging string lights overhead",
    "A person near a fire hydrant on a wet city street",
    "A person on a city street with autumn leaves blowing around them",
    "A person standing beside a parked bicycle in an urban setting",
    "A person on a city sidewalk with a hot dog stand in the background",
    "A person standing in an urban scene with scaffolding and tarps on nearby buildings",
    # Adding people / crowds
    "A person standing on a busy city street surrounded by a diverse crowd of pedestrians",
    "A person in a city plaza with street performers and onlookers nearby",
    "A person on a sidewalk with commuters rushing past in motion blur",
    "A person standing at a city crosswalk with cyclists and pedestrians around them",
    # Atmospheric / VFX additions
    "A person standing on a city street with light fog rolling in",
    "A person in an urban scene with dramatic storm clouds building overhead",
    "A person on a city sidewalk with confetti falling around them",
    "A person standing on a street with autumn leaves swirling in the wind",
    "A person in an urban scene with light rays breaking through buildings",
    "A person on a city street with bokeh lights of traffic in the background",
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
