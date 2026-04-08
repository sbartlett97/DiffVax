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
from diffvax.immunization import DiffVaxImmunization
from diffvax.utils import (
    load_image,
    prepare_mask_and_masked_image,
    get_train_val_image_prompt_list,
    ensure_dataset_in_data_dir,
)


def _build_attack_model(config):
    """Construct the appropriate attack model from config.

    Supports three modes:
    - Single SD 1.5 model (original): ``attack_model_link`` key only
    - Single FLUX model: ``flux_model_link`` key only
    - Multi-model: both keys present, selected randomly per batch according to
      ``sd_probability`` / ``flux_probability`` (and optionally ``sd3_probability``
      / ``sd3_model_link`` for SD 3.5)
    """
    has_sd = "attack_model_link" in config
    has_flux = "flux_model_link" in config
    has_sd3 = "sd3_model_link" in config

    if has_sd and not has_flux and not has_sd3:
        return Attack(config["attack_model_link"])

    if has_flux and not has_sd and not has_sd3:
        from diffvax.attack_flux import FluxAttack
        return FluxAttack(config["flux_model_link"])

    # Multi-model: assemble spec list
    from diffvax.attack_multi import MultiAttack
    specs = []
    if has_sd:
        specs.append({
            "type": "sd15",
            "link": config["attack_model_link"],
            "prob": config.get("sd_probability", 0.5),
            "loss_scale": config.get("sd_loss_scale", 1.0),
        })
    if has_flux:
        specs.append({
            "type": "flux",
            "link": config["flux_model_link"],
            "prob": config.get("flux_probability", 0.5),
            "loss_scale": config.get("flux_loss_scale", 1.0),
        })
    if has_sd3:
        specs.append({
            "type": "sd3",
            "link": config["sd3_model_link"],
            "prob": config.get("sd3_probability", 0.1),
            "loss_scale": config.get("sd3_loss_scale", 1.0),
        })
    return MultiAttack(specs)


def immunize_image_list(image_prompt_list, config, data_dir, output_dir):
    iter_num = config["iter_num"]
    immunization_model_name = config["immunization_model"]
    alpha = config["alpha"]
    batch_size = config["batch_size"]
    train_all = config["train_all"]

    attack_model = _build_attack_model(config)

    immunization_config = {
        "iter_num": iter_num,
        "learning_rate": config["learning_rate"],
        "immunization_model": immunization_model_name,
        "vae_loss_beta": config.get("vae_loss_beta", 0.0),
        "max_steps": config.get("max_steps", None),
        # H7: JPEG augmentation — must be forwarded or training silently runs without it
        "jpeg_augment_prob": config.get("jpeg_augment_prob", 0.0),
        "jpeg_quality_range": config.get("jpeg_quality_range", [70, 85]),
        # Checkpoint / stop-file controls
        "checkpoint_every": config.get("checkpoint_every", 5),
        "stop_file": config.get("stop_file", "/tmp/diffvax_stop"),
    }
    immunization_mdl = DiffVaxImmunization(
        attack_model, immunization_config, output_dir=output_dir
    )

    if not train_all:
        index_list = config["image_index_list"]
        image_prompt_list = [image_prompt_list[index] for index in index_list]

    image_name_list = [image_prompt["image"][:-4] for image_prompt in image_prompt_list]
    prompt_list = [image_prompt["prompts"] for image_prompt in image_prompt_list]
    image_torch_list = []
    mask_torch_list = []
    prompt_train_list = []

    # Support both dataset layouts: images/masks or cropped_images/sam_masks
    images_subdir = config.get("images_subdir", "train/images")
    masks_subdir = config.get("masks_subdir", "train/masks")
    resolution = config.get("resolution", 512)

    for image_ind, image_name in enumerate(image_name_list):
        image = load_image(
            image_name,
            data_dir,
            is_mask=False,
            images_subdir=images_subdir,
            masks_subdir=masks_subdir,
            resolution=resolution,
        )
        image_mask = load_image(
            image_name,
            data_dir,
            is_mask=True,
            images_subdir=images_subdir,
            masks_subdir=masks_subdir,
            resolution=resolution,
        )
        mask_torch, image_torch, non_masked_image_torch = prepare_mask_and_masked_image(
            image, image_mask
        )
        image_torch = image_torch.half().cuda()
        non_masked_image_torch = non_masked_image_torch.half().cuda()
        mask_torch = mask_torch.half().cuda()

        cur_prompt_list = prompt_list[image_ind]
        for prompt in cur_prompt_list:
            image_torch_list.append(image_torch.squeeze(0))
            mask_torch_list.append(mask_torch.squeeze(0))
            prompt_train_list.append(prompt)

    immunized_img, immunization_model_path = (
        immunization_mdl.train_immunization_all_images_batch(
            image_torch_list,
            mask_torch_list,
            prompt_train_list,
            target_image=None,
            alpha=alpha,
            iter_num=iter_num,
            batch_size=batch_size,
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
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    config["data_dir"] = args.data_dir
    config["output_dir"] = args.output_dir

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
