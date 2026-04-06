"""Modal.com launcher for DiffVax training experiments.

Runs DiffVax training on an A100-80GB GPU in the cloud.

Prerequisites:
    pip install modal
    modal setup  # authenticate

Usage:
    # Dry run (check setup)
    modal run research/src/run_on_modal.py --config configs/train_multimodel.yml --dry-run

    # Launch H1a experiment
    modal run research/src/run_on_modal.py --config configs/train_multimodel.yml

    # Launch H3 (1088 fine-tuning)
    modal run research/src/run_on_modal.py --config configs/train_1088.yml
"""

import os
import sys
from pathlib import Path

import modal

# Modal image: Python 3.12 + full ML stack
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.4.1",
        "torchvision",
        "diffusers>=0.31.0",
        "transformers>=4.44.0",
        "accelerate",
        "xformers",
        "huggingface_hub",
        "numpy",
        "pillow",
        "tqdm",
        "pyyaml",
        "scipy",
        "opencv-python-headless",
        "sentencepiece",
        "protobuf",
    )
    .run_commands(
        "pip install flash-attn --no-build-isolation || true",
    )
)

app = modal.App("diffvax-training", image=image)

# Persistent volume for checkpoints and dataset
volume = modal.Volume.from_name("diffvax-data", create_if_missing=True)

PROJECT_DIR = "/diffvax"
DATA_DIR = "/diffvax/data"
OUTPUT_DIR = "/diffvax/outputs"
CHECKPOINT_DIR = "/diffvax/checkpoints"


@app.function(
    gpu="A100-80GB",
    timeout=3600 * 12,  # 12 hours max
    volumes={PROJECT_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-token", required=False)],
)
def train(config_path: str, data_dir: str = DATA_DIR, output_dir: str = OUTPUT_DIR):
    """Run DiffVax training on Modal A100."""
    import subprocess
    import sys

    # Mount project code into the container
    # (The code is bundled by Modal via the mounts below)
    result = subprocess.run(
        [
            sys.executable, "scripts/train.py",
            "--config", config_path,
            "--data-dir", data_dir,
            "--output-dir", output_dir,
        ],
        cwd=PROJECT_DIR,
        capture_output=False,
    )
    return result.returncode


@app.local_entrypoint()
def main(
    config: str = "configs/train_multimodel.yml",
    dry_run: bool = False,
):
    """Launch training on Modal.

    Args:
        config: path to training config YAML (relative to project root).
        dry_run: if True, just print the config and exit.
    """
    config_path = Path(config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}")
        sys.exit(1)

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"Training config: {config_path}")
    print(f"  Project: {cfg.get('project_name', 'unknown')}")
    print(f"  Iterations: {cfg.get('iter_num', '?')}")
    print(f"  Batch size: {cfg.get('batch_size', '?')}")
    print(f"  Resolution: {cfg.get('resolution', 512)}")
    print(f"  SD prob: {cfg.get('sd_probability', 'N/A')}")
    print(f"  FLUX prob: {cfg.get('flux_probability', 'N/A')}")
    print(f"  SD3 prob: {cfg.get('sd3_probability', 'N/A')}")

    if dry_run:
        print("\nDry run — not launching.")
        return

    print("\nLaunching on Modal A100-80GB...")
    ret = train.remote(config_path=config, data_dir=DATA_DIR, output_dir=OUTPUT_DIR)
    print(f"Training finished with exit code: {ret}")
