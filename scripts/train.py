#!/usr/bin/env python3
"""Train DiffVax immunization model."""

import argparse
import os
import sys
import yaml

# Add src to path for package imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, "src"))

from diffvax.attack import Attack
from diffvax.attack_manager import AttackModelManager
from diffvax.immunization import DiffVaxImmunization
from diffvax.utils import (
    get_train_val_image_prompt_list,
    ensure_dataset_in_data_dir,
)


def immunize_image_list(image_prompt_list, config, data_dir, output_dir):
    iter_num = config["iter_num"]
    immunization_model_name = config["immunization_model"]
    alpha = config["alpha"]
    batch_size = config["batch_size"]
    train_all = config["train_all"]
    attack_model_link = config["attack_model_link"]
    resolution = config.get("resolution", 512)

    # Build attack models and manager
    models = {}
    probabilities = {}

    # SD attack (always created)
    sd_prob = config.get("sd_probability", 1.0)
    if sd_prob > 0:
        sd_attack = Attack(attack_model_link)
        models["sd"] = sd_attack
        probabilities["sd"] = sd_prob

    # FLUX attack (optional)
    flux_model_link = config.get("flux_model_link")
    flux_prob = config.get("flux_probability", 0.0)
    if flux_model_link and flux_prob > 0:
        from diffvax.flux_attack import FluxAttack
        flux_attack = FluxAttack(flux_model_link)
        models["flux"] = flux_attack
        probabilities["flux"] = flux_prob

    attack_manager = AttackModelManager(models, probabilities)

    immunization_config = {
        "iter_num": iter_num,
        "learning_rate": config["learning_rate"],
        "immunization_model": immunization_model_name,
    }

    load_existing = config.get("load_existing", False)
    load_path = config.get("load_path")

    immunization_mdl = DiffVaxImmunization(
        config=immunization_config,
        attack_manager=attack_manager,
        load_existing=load_existing,
        load_path=load_path,
        output_dir=output_dir,
    )

    if not train_all:
        index_list = config["image_index_list"]
        image_prompt_list = [image_prompt_list[index] for index in index_list]

    image_name_list = [image_prompt["image"][:-4] for image_prompt in image_prompt_list]
    prompt_list = [image_prompt["prompts"] for image_prompt in image_prompt_list]
    flux_prompt_list = [image_prompt.get("flux_prompts", image_prompt["prompts"]) for image_prompt in image_prompt_list]

    images_subdir = config.get("images_subdir", "train/images")
    masks_subdir = config.get("masks_subdir", "train/masks")
    size = (resolution, resolution)

    entries = []
    for image_ind, image_name in enumerate(image_name_list):
        cur_prompt_list = prompt_list[image_ind]
        cur_flux_prompt_list = flux_prompt_list[image_ind]
        for prompt_idx, prompt in enumerate(cur_prompt_list):
            flux_prompt = (
                cur_flux_prompt_list[prompt_idx]
                if prompt_idx < len(cur_flux_prompt_list)
                else prompt
            )
            entry = {"image_name": image_name, "prompt": prompt, "flux_prompt": flux_prompt}
            if "mask_types_available" in image_prompt_list[image_ind]:
                entry["mask_types_available"] = image_prompt_list[image_ind]["mask_types_available"]
            entries.append(entry)

    sd_target_resolutions = config.get("sd_target_resolutions", [512])
    whole_image_probability = config.get("whole_image_probability", 0.0)

    immunized_img, immunization_model_path = (
        immunization_mdl.train_immunization_all_images_batch(
            entries,
            data_dir,
            images_subdir,
            masks_subdir,
            size,
            target_image=None,
            alpha=alpha,
            iter_num=iter_num,
            batch_size=batch_size,
            sd_target_resolutions=sd_target_resolutions,
            whole_image_probability=whole_image_probability,
        )
    )

    return immunization_model_path


def main():
    parser = argparse.ArgumentParser(description="Train DiffVax immunization model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Path to output directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to an existing perturbation net checkpoint (.pth) to resume from",
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    config["data_dir"] = args.data_dir
    config["output_dir"] = args.output_dir

    # CLI --checkpoint overrides config file values
    if args.checkpoint is not None:
        config["load_existing"] = True
        config["load_path"] = args.checkpoint

    data_dir = config["data_dir"]

    data_dir = ensure_dataset_in_data_dir(
        repo_id="ozdentarikcan/DiffVaxDataset",
        data_dir=data_dir,
    )

    train_list, val_list = get_train_val_image_prompt_list(data_dir)

    immunization_model_path = immunize_image_list(
        train_list, config, data_dir, config["output_dir"]
    )
    print(f"Training complete. Model saved to: {immunization_model_path}")


if __name__ == "__main__":
    main()
