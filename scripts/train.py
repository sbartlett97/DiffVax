#!/usr/bin/env python3
"""Train DiffVax immunization model."""

import argparse
import os
import sys
import traceback
import yaml

# Add src to path for package imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, "src"))

from diffvax.attack import Attack
from diffvax.attack_manager import AttackModelManager
from diffvax.distributed import (
    barrier, cleanup_distributed, get_rank, get_world_size, init_distributed,
    is_distributed, is_main_process,
)
from diffvax.immunization import DiffVaxImmunization
from diffvax.utils import (
    get_train_val_image_prompt_list,
    ensure_dataset_in_data_dir,
)


def _build_surrogate_specs(config):
    """Enumerate the enabled surrogates as (name, probability, builder) triples.

    Builders are deferred so that under DDP each rank constructs ONLY the one
    surrogate it owns — instantiating all of them on every rank would defeat
    the entire point (and OOM immediately).
    """
    specs = []

    sd_prob = config.get("sd_probability", 1.0)
    if sd_prob > 0:
        attack_model_link = config["attack_model_link"]
        specs.append(("sd", sd_prob, lambda: Attack(attack_model_link)))

    flux_model_link = config.get("flux_model_link")
    flux_prob = config.get("flux_probability", 0.0)
    if flux_model_link and flux_prob > 0:
        flux_cfg = config.get("flux_attack", {})

        def _build_flux():
            from diffvax.flux_attack import FluxAttack
            return FluxAttack(
                flux_model_link,
                gradient_timestep_fraction=flux_cfg.get("gradient_timestep_fraction", 1.0),
                token_gradient_regularization=flux_cfg.get("token_gradient_regularization", False),
                use_gradient_checkpointing=flux_cfg.get("use_gradient_checkpointing", True),
            )

        specs.append(("flux", flux_prob, _build_flux))

    sd3_model_link = config.get("sd3_model_link")
    sd3_prob = config.get("sd3_probability", 0.0)
    if sd3_model_link and sd3_prob > 0:
        sd3_cfg = config.get("sd3_attack", {})

        def _build_sd3():
            from diffvax.sd3_attack import SD3Attack
            return SD3Attack(
                sd3_model_link,
                gradient_timestep_fraction=sd3_cfg.get("gradient_timestep_fraction", 1.0),
                token_gradient_regularization=sd3_cfg.get("token_gradient_regularization", False),
                use_gradient_checkpointing=sd3_cfg.get("use_gradient_checkpointing", True),
                offload_text_encoders=sd3_cfg.get("offload_text_encoders", True),
            )

        specs.append(("sd3", sd3_prob, _build_sd3))

    if not specs:
        raise ValueError(
            "No attack models configured. Set sd_probability > 0 or provide "
            "flux_model_link/sd3_model_link with their probabilities."
        )
    return specs


def _build_attack_manager(config):
    """Construct the AttackModelManager for this process.

    Single-process: builds every enabled surrogate and samples between them
    per batch, weighted by the configured probabilities (original behaviour).

    Distributed: each rank builds exactly ONE surrogate, assigned round-robin
    by rank, and keeps it resident for the whole run. The configured
    probabilities are ignored in this mode — the effective ensemble mixture is
    set by how many ranks are assigned to each surrogate. With a single model
    the manager's swap logic is inherently a no-op, so no model is ever moved
    off its device mid-run.
    """
    specs = _build_surrogate_specs(config)
    adaptive_cfg = config.get("adaptive_ensemble", {})
    adaptive_enabled = adaptive_cfg.get("enabled", False)

    if not is_distributed():
        models = {name: build() for name, _, build in specs}
        probabilities = {name: prob for name, prob, _ in specs}
        return AttackModelManager(
            models,
            probabilities,
            adaptive=adaptive_enabled,
            adaptive_cfg=adaptive_cfg,
        )

    rank, world_size = get_rank(), get_world_size()
    name, _, build = specs[rank % len(specs)]

    if rank == 0:
        assignment = {
            r: specs[r % len(specs)][0] for r in range(world_size)
        }
        print(f"[DDP] Surrogate assignment by rank: {assignment}")
        if world_size < len(specs):
            unused = [s[0] for s in specs[world_size:]]
            print(
                f"[DDP] WARNING: world_size={world_size} < {len(specs)} configured "
                f"surrogates; these are UNUSED this run: {unused}. "
                f"Launch with at least {len(specs)} ranks to train against all."
            )
    print(f"[DDP] rank {rank}/{world_size} loading surrogate '{name}'")

    # Adaptive ensemble weighting is a *sampling* strategy over multiple
    # resident surrogates; with one surrogate per rank there is nothing to
    # sample, so it is disabled regardless of config.
    return AttackModelManager(
        {name: build()},
        {name: 1.0},
        adaptive=False,
    )


def build_immunization_config(config: dict) -> dict:
    """Curate the subset of the full YAML config that DiffVaxImmunization
    reads via self._config. Every key DiffVaxImmunization reads with
    self._config.get(...) MUST be listed here, or it silently falls back to
    that call's hardcoded default regardless of what the YAML says (bit us
    once already: sd3_attack.masked_attack_probability and
    num_inference_steps were both missing here and silently no-opped/
    defaulted for every run until this was caught) — see
    tests/test_train_config_passthrough.py, which cross-checks this dict's
    keys against every self._config.get(...) call in diffvax_immunization.py
    so a newly-added config read can't reintroduce the same gap unnoticed.
    """
    return {
        "iter_num": config["iter_num"],
        "learning_rate": config["learning_rate"],
        "immunization_model": config["immunization_model"],
        # Pass through all v2 phase configs so DiffVaxImmunization can read them
        "eot": config.get("eot", {}),
        "clip_loss": config.get("clip_loss", {}),
        "beta": config.get("beta", 0.5),
        "curriculum": config.get("curriculum", {}),
        "resolution": config.get("resolution", 512),
        "batch_size": config["batch_size"],
        "adaptive_ensemble": config.get("adaptive_ensemble", {}),
        "flat_minima": config.get("flat_minima", {}),
        "attention_loss": config.get("attention_loss", {}),
        "noise_target": config.get("noise_target", {}),
        "spectral_loss": config.get("spectral_loss", {}),
        "latent_loss": config.get("latent_loss", {}),
        "nb_filter": config.get("nb_filter"),
        "num_inference_steps": config.get("num_inference_steps", 4),
        "sd3_attack": config.get("sd3_attack", {}),
        "flux_attack": config.get("flux_attack", {}),
        "perturbation_mask_gating": config.get("perturbation_mask_gating", False),
        "max_grad_norm": config.get("max_grad_norm", 5.0),
        "dataloader": config.get("dataloader", {}),
        "hub": config.get("hub", {}),
        "reporting": config.get("reporting", {}),
    }


def immunize_image_list(image_prompt_list, config, data_dir, output_dir):
    iter_num = config["iter_num"]
    alpha = config["alpha"]
    batch_size = config["batch_size"]
    train_all = config["train_all"]
    resolution = config.get("resolution", 512)

    attack_manager = _build_attack_manager(config)

    immunization_config = build_immunization_config(config)

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
    flux_prompt_list = [
        image_prompt.get("flux_prompts", image_prompt["prompts"])
        for image_prompt in image_prompt_list
    ]

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
            entry = {
                "image_name": image_name,
                "prompt": prompt,
                "flux_prompt": flux_prompt,
            }
            if "mask_types_available" in image_prompt_list[image_ind]:
                entry["mask_types_available"] = image_prompt_list[image_ind][
                    "mask_types_available"
                ]
            entries.append(entry)

    sd_target_resolutions = config.get("sd_target_resolutions", [512])
    strength_range = config.get("strength_range", [0.5, 1.0])

    try:
        result = immunization_mdl.train_immunization_all_images_batch(
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
            strength_range=strength_range,
        )
        # train_immunization_all_images_batch returns bare None on a NaN/Inf
        # abort (see its early `return` in the NaN-guard block) rather than
        # the usual (img_adv, path) tuple — unpacking that unconditionally
        # raises an unrelated-looking TypeError that stomps the actually
        # informative [NaN] message already printed above it.
        if result is None:
            raise RuntimeError(
                "Training aborted before completing (NaN/Inf loss on some "
                "batch) — see the [NaN] message above and training_log.json "
                "for which loss term was at fault. An emergency checkpoint "
                "was saved at the point of failure."
            )
        immunized_img, immunization_model_path = result
    except Exception as exc:
        tb_str = traceback.format_exc()
        immunization_mdl.reporter.report_error(
            "fatal",
            f"{type(exc).__name__}: {exc}\n\n{tb_str}",
        )
        raise

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

    # Multi-GPU: no-ops unless launched under torchrun, so the single-process
    # invocation is completely unchanged.
    #   torchrun --nproc_per_node=3 scripts/train.py --config configs/full_v2.yml
    # Each rank pins one surrogate; only the NestedUNet gradients are
    # all-reduced. See src/diffvax/distributed.py.
    init_distributed()
    try:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)

        config["data_dir"] = args.data_dir
        config["output_dir"] = args.output_dir

        # CLI --checkpoint overrides config file values
        if args.checkpoint is not None:
            config["load_existing"] = True
            config["load_path"] = args.checkpoint

        data_dir = config["data_dir"]

        # Only rank 0 downloads the dataset; the others wait, then read the
        # already-populated directory. Concurrent snapshot_download calls into
        # the same target would race.
        if is_main_process():
            data_dir = ensure_dataset_in_data_dir(
                repo_id="ozdentarikcan/DiffVaxDataset",
                data_dir=data_dir,
            )
        barrier()
        if not is_main_process():
            data_dir = ensure_dataset_in_data_dir(
                repo_id="ozdentarikcan/DiffVaxDataset",
                data_dir=data_dir,
            )

        train_list, val_list = get_train_val_image_prompt_list(data_dir)

        immunization_model_path = immunize_image_list(
            train_list, config, data_dir, config["output_dir"]
        )
        if is_main_process():
            print(f"Training complete. Model saved to: {immunization_model_path}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
