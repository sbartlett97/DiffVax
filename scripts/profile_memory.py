#!/usr/bin/env python3
"""Profile peak accelerator memory for one training step, per curriculum
resolution stage and batch size.

Loads the real surrogate(s) for a config (via scripts/train.py's own
_build_attack_manager, so it's exactly what a real run would load) and drives
one manual forward+backward+optimizer step per (resolution, batch_size)
combination using synthetic random image/mask tensors in place of a real
dataset. Memory usage depends on tensor shape/dtype/graph structure, not
pixel content, so synthetic data is a faithful stand-in for this purpose and
avoids needing a downloaded dataset just to answer "what batch size fits".

The per-step body mirrors the core of
DiffVaxImmunization.train_immunization_all_images_batch's per-batch logic
(unet forward -> EoT -> attack -> losses -> backward), with the
dataset/dataloader/curriculum-epoch machinery stripped out so each
(resolution, batch_size) can be probed directly and cheaply. If that
function's core forward path changes, update run_one_step() to match.

Usage:
    python scripts/profile_memory.py --config configs/research_v3_sd3_only.yml
    python scripts/profile_memory.py --config configs/research_v3_sd3_only.yml \
        --resolutions 512 768 1024 1088 --batch-sizes 1 2 3 4 6 8
"""

import argparse
import gc
import os
import random
import sys

import torch
import torch.nn.functional as F
import yaml

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, "src"))
sys.path.insert(0, _script_dir)  # to import train.py for _build_attack_manager

from diffvax.immunization import DiffVaxImmunization  # noqa: E402
from diffvax.utils import resolve_device, resolve_dtype, empty_cache  # noqa: E402

import train as train_mod  # scripts/train.py — reuse surrogate construction  # noqa: E402


def _device_memory_gb(device):
    """(allocated_gb, reserved_gb) for the active accelerator.

    MPS has no CUDA-style peak-memory-stats API, so current_allocated_memory()
    read immediately after the step (before any empty_cache) is the best
    available proxy for that step's peak.
    """
    if device.type == "cuda":
        return (
            torch.cuda.max_memory_allocated(device) / 2**30,
            torch.cuda.max_memory_reserved(device) / 2**30,
        )
    if device.type == "mps":
        return (
            torch.mps.current_allocated_memory() / 2**30,
            torch.mps.driver_allocated_memory() / 2**30,
        )
    import resource
    return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, None)


def _reset_peak_stats(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def run_one_step(immun, resolution, batch_size, sd_target_resolutions, strength_range, alpha):
    """One forward+backward+optimizer step on synthetic data."""
    device = immun.device
    dtype = resolve_dtype(device)

    # float32 like the real loader (see ImmunizationDataset.__getitem__)
    img_batch = torch.rand(batch_size, 3, resolution, resolution, device=device) * 2 - 1
    mask_batch = (torch.rand(batch_size, 1, resolution, resolution, device=device) > 0.5).to(dtype)
    prompt_batch = ["a photo of a scene, high detail"] * batch_size

    immun.optimizer.zero_grad()
    model_name, attack_model = immun.attack_manager.select_and_load()
    cur_prompt = prompt_batch

    ones = torch.ones_like(mask_batch)
    img_f = img_batch.float()
    unet_out = immun.unetmodel(img_f)
    img_adv = torch.clamp(img_f + unet_out, immun.clamp_min, immun.clamp_max)

    img_adv_aug = immun._eot(img_adv) if immun._eot is not None else img_adv
    strength = random.uniform(*strength_range)

    attn_only_with_dit = bool(immun._config.get("attention_loss", {}).get("only_with_dit", True))
    attn_active = immun._attention_loss is not None and (
        not attn_only_with_dit or model_name in {"sd3", "flux"}
    )
    if attn_active:
        transformer = getattr(getattr(attack_model, "pipe", None), "transformer", None)
        if transformer is not None:
            immun._attention_loss.register_hooks(transformer)

    h, w = img_batch.shape[2], img_batch.shape[3]
    actual_bs = img_batch.shape[0]
    num_inference_steps = int(immun._config.get("num_inference_steps", 4))

    if attack_model.is_inpainting:
        sd_target = random.choice(sd_target_resolutions)
        attack_mask = mask_batch
        if sd_target < h:
            img_adv_resized = F.interpolate(img_adv_aug, (sd_target, sd_target), mode="bilinear", align_corners=False)
            mask_resized = F.interpolate(attack_mask, (sd_target, sd_target), mode="nearest")
            img_adv_sd = img_adv_resized * (1.0 - mask_resized)
            img_out_small = attack_model.attack(
                prompt=cur_prompt, image=img_adv_sd, mask=mask_resized,
                height=sd_target, width=sd_target,
                num_inference_steps=num_inference_steps,
                batch_size=actual_bs, strength=strength,
            )
            img_out = F.interpolate(img_out_small, (h, w), mode="bilinear", align_corners=False)
        else:
            img_adv_sd = img_adv_aug * (1.0 - attack_mask)
            img_out = attack_model.attack(
                prompt=cur_prompt, image=img_adv_sd, mask=attack_mask,
                height=h, width=w,
                num_inference_steps=num_inference_steps,
                batch_size=actual_bs, strength=strength,
            )
    else:
        native_res = attack_model.native_resolution
        if h > native_res:
            img_input = F.interpolate(img_adv_aug, (native_res, native_res), mode="bilinear", align_corners=False)
            img_out_native = attack_model.attack(
                prompt=cur_prompt, image=img_input, height=native_res, width=native_res,
                num_inference_steps=num_inference_steps,
                batch_size=actual_bs, strength=strength,
            )
            img_out = F.interpolate(img_out_native, (h, w), mode="bilinear", align_corners=False)
        else:
            img_out = attack_model.attack(
                prompt=cur_prompt, image=img_adv_aug, height=h, width=w,
                num_inference_steps=num_inference_steps,
                batch_size=actual_bs, strength=strength,
            )

    if immun._config.get("noise_target", {}).get("enabled", False):
        target_image_t = torch.randint(0, 2, img_out.shape, device=img_out.device, dtype=img_out.dtype) * 2 - 1
    else:
        target_image_t = torch.zeros_like(img_out)

    loss1_weight = mask_batch if attack_model.loss_uses_mask_weighting else ones
    loss1_weight_norm = loss1_weight / h
    loss2_weight_norm = ones / h

    loss1 = ((img_out - target_image_t) * loss1_weight_norm).norm(p=1) / loss1_weight_norm.sum()
    loss2 = (alpha * (img_adv - img_batch) * loss2_weight_norm).norm(p=1) / loss2_weight_norm.sum()

    loss_extra = torch.tensor(0.0, device=device)
    if immun._loss_composer is not None and immun._loss_composer.has_terms():
        loss_extra, _ = immun._loss_composer.compute(
            img_orig=img_batch, img_adv=img_adv, img_out=img_out, prompts=cur_prompt,
        )

    loss_latent = torch.tensor(0.0, device=device)
    if immun._config.get("latent_loss", {}).get("enabled", False):
        vae = attack_model.get_vae()
        if vae is not None:
            from diffvax.losses.latent_loss import latent_disruption_loss
            loss_latent = latent_disruption_loss(
                vae, img_batch, img_adv,
                mode=immun._config.get("latent_loss", {}).get("mode", "cosine"),
            )

    loss_hs = torch.tensor(0.0, device=device)
    hs_cfg = immun._config.get("hidden_state_loss", {})
    if hs_cfg.get("enabled", False) and hasattr(attack_model, "hidden_state_loss"):
        loss_hs = attack_model.hidden_state_loss(
            img_batch, img_adv, cur_prompt,
            sigma=float(hs_cfg.get("sigma", 1.0)),
            group=hs_cfg.get("group", "single"),
            target_blocks=hs_cfg.get("target_blocks", "middle"),
            num_hooks=int(hs_cfg.get("num_hooks", 4)),
        ).float()

    loss_attn = torch.tensor(0.0, device=device)
    if attn_active:
        loss_attn = immun._attention_loss.compute()
    if immun._attention_loss is not None:
        immun._attention_loss.remove_hooks()

    attn_weight = float(immun._config.get("attention_loss", {}).get("weight", 0.3))
    latent_loss_weight = float(immun._config.get("latent_loss", {}).get("weight", 1.0))
    loss = (loss1 + loss2 + loss_extra + attn_weight * loss_attn
            + latent_loss_weight * loss_latent
            + float(hs_cfg.get("weight", 0.5)) * loss_hs)

    loss.backward()
    immun.optimizer.step()
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description="Profile peak accelerator memory per resolution/batch size")
    parser.add_argument("--config", type=str, default="configs/research_v3_sd3_only.yml")
    parser.add_argument(
        "--resolutions", type=int, nargs="+", default=None,
        help="Resolutions to test (default: curriculum.stages resolutions from --config)",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    resolutions = args.resolutions
    if resolutions is None:
        stages = config.get("curriculum", {}).get("stages")
        resolutions = [s["resolution"] for s in stages] if stages else [config["resolution"]]

    device = resolve_device()
    print(f"[profile_memory] device={device}")
    if device.type != "mps":
        print(
            f"[profile_memory] WARNING: resolved device is '{device.type}', not 'mps' — "
            f"these numbers won't reflect unified-memory (Apple Silicon) behaviour."
        )

    print("[profile_memory] Building attack manager + immunization model "
          "(loads surrogate weights — can take a while on first run)...")
    attack_manager = train_mod._build_attack_manager(config)
    immun = DiffVaxImmunization(
        config=config, attack_manager=attack_manager, output_dir="outputs/profile_memory_tmp",
    )

    sd_target_resolutions = config.get("sd_target_resolutions", [512])
    strength_range = config.get("strength_range", [0.5, 1.0])
    alpha = config.get("alpha", 4)

    results = {}
    for res in resolutions:
        results[res] = {"max_ok_batch": None, "peak_gb": None}
        for bs in sorted(args.batch_sizes):
            empty_cache(device)
            gc.collect()
            _reset_peak_stats(device)
            try:
                run_one_step(immun, res, bs, sd_target_resolutions, strength_range, alpha)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                print(f"[{res}px, bs={bs}] OOM")
                immun.optimizer.zero_grad(set_to_none=True)
                empty_cache(device)
                break  # larger batch sizes at this resolution will only get worse
            allocated_gb, reserved_gb = _device_memory_gb(device)
            reserved_str = f", reserved={reserved_gb:.2f}GB" if reserved_gb is not None else ""
            print(f"[{res}px, bs={bs}] OK — allocated={allocated_gb:.2f}GB{reserved_str}")
            results[res]["max_ok_batch"] = bs
            results[res]["peak_gb"] = allocated_gb

    print("\n=== Summary ===")
    print(f"{'Resolution':>10} | {'Max batch_size':>15} | {'Peak allocated (GB)':>20}")
    for res, r in results.items():
        bs_str = str(r["max_ok_batch"]) if r["max_ok_batch"] is not None else "OOM at bs=1"
        peak_str = f"{r['peak_gb']:.2f}" if r["peak_gb"] is not None else "-"
        print(f"{res:>10} | {bs_str:>15} | {peak_str:>20}")


if __name__ == "__main__":
    main()
