"""DiffVax immunization against diffusion attacks — v2 unified training loop.

Integrates all seven phases of the DiffVax v2 implementation plan:
  Phase 1: EoT augmentation (src/diffvax/eot.py)
  Phase 2: CLIP-based disruption loss (src/diffvax/losses/clip_loss.py)
  Phase 3: SD3/FLUX 16-ch VAE surrogate (src/diffvax/sd3_attack.py)
  Phase 4: Multi-resolution curriculum (src/diffvax/curriculum.py)
  Phase 5: Adaptive ensemble weighting (src/diffvax/attack_manager.py)
  Phase 6: Flat-minima regularization (src/diffvax/losses/flat_minima.py)
  Phase 7: Cross-attention disruption loss (src/diffvax/losses/attention_loss.py)

All new features are gated by config flags. Setting all enabled flags to False
reproduces the original v1 behaviour exactly (loss1 + loss2 only).
"""

import random

import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm

from diffvax.model import NestedUNet
from diffvax.utils import set_seed_lib, load_image

scaler = torch.cuda.amp.GradScaler()


class ImmunizationDataset(torch.utils.data.Dataset):
    """Dataset for immunization training — streams images from disk on demand.

    Supports dynamic resolution updates via set_resolution() for Phase 4
    multi-resolution curriculum training. When set_resolution() is called
    between epochs, the next __getitem__ call uses the updated size.
    """

    def __init__(self, entries, data_dir, images_subdir, masks_subdir, size):
        self.entries = entries          # list of {"image_name", "prompt", "flux_prompt"}
        self.data_dir = data_dir
        self.images_subdir = images_subdir
        self.masks_subdir = masks_subdir
        self._current_size = size       # (H, W) tuple, updated by set_resolution()

    @property
    def size(self):
        return self._current_size

    def set_resolution(self, resolution: int) -> None:
        """Update the target load resolution for subsequent __getitem__ calls.

        Args:
            resolution: Square resolution in pixels (must be multiple of 16).
        """
        self._current_size = (resolution, resolution)

    def __getitem__(self, index):
        entry = self.entries[index]
        mask_type = None
        if "mask_types_available" in entry:
            mask_type = random.choice(entry["mask_types_available"])
        image = load_image(
            entry["image_name"], self.data_dir, is_mask=False,
            images_subdir=self.images_subdir, masks_subdir=self.masks_subdir,
            size=self._current_size,
        )
        image_mask = load_image(
            entry["image_name"], self.data_dir, is_mask=True,
            images_subdir=self.images_subdir, masks_subdir=self.masks_subdir,
            size=self._current_size, mask_type=mask_type,
        )
        img_np = np.array(image.convert("RGB"), dtype=np.float32)
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1)) / 127.5 - 1.0
        mask_np = np.array(image_mask.convert("L"), dtype=np.uint8)
        mask_t = torch.from_numpy((mask_np >= 128).astype(np.float32)[None])
        return (
            img_t.half(),
            mask_t.half(),
            entry["prompt"],
            entry["flux_prompt"],
        )

    def __len__(self):
        return len(self.entries)


class DiffVaxImmunization:
    def __init__(
        self,
        attack_model=None,
        config=None,
        load_existing=False,
        existing_iter_num=0,
        load_path=None,
        output_dir=None,
        attack_manager=None,
    ):
        self.model_name = "DiffVaxImmunization"
        self.step_size = 1
        self.eps = 32 / 255
        self.clamp_min = -1
        self.clamp_max = 1
        self.output_dir = output_dir or "outputs"
        self._config = config or {}

        # Support both single attack_model (backward compat) and attack_manager
        if attack_manager is not None:
            self.attack_manager = attack_manager
            self.attack_model = None
        elif attack_model is not None:
            from diffvax.attack_manager import AttackModelManager
            self.attack_manager = AttackModelManager(
                models={"sd": attack_model},
                probabilities={"sd": 1.0},
            )
            self.attack_model = attack_model
        else:
            raise ValueError("Either attack_model or attack_manager must be provided")

        unetmodel = NestedUNet(num_classes=3)
        self.unetmodel = unetmodel.to("cuda")
        learning_rate = config["learning_rate"]
        self.optimizer = torch.optim.Adam(unetmodel.parameters(), lr=learning_rate)

        self.load_existing = load_existing
        self.existing_iter_num = existing_iter_num

        if self.load_existing:
            if not load_path:
                raise ValueError("load_existing=True but no load_path was provided")
            self.unetmodel.load_state_dict(torch.load(load_path, weights_only=True))
        self.model = self.unetmodel

        for param in self.unetmodel.parameters():
            param.requires_grad = True

        generator = torch.Generator(device="cuda")
        self.generator = generator

        # ---- Phase 1: EoT augmentation ----
        self._eot = None
        if self._config.get("eot", {}).get("enabled", False):
            from diffvax.eot import DifferentiableEoT
            self._eot = DifferentiableEoT(self._config)

        # ---- Phase 2: CLIP loss via LossComposer ----
        self._loss_composer = None
        if self._config.get("clip_loss", {}).get("enabled", False):
            from diffvax.losses import LossComposer
            self._loss_composer = LossComposer(self._config)

        # ---- Phase 4: Resolution curriculum ----
        from diffvax.curriculum import ResolutionCurriculum
        self._curriculum = ResolutionCurriculum(self._config)

        # ---- Phase 6: Flat-minima regularization ----
        self._flat_minima = None
        if self._config.get("flat_minima", {}).get("enabled", False):
            from diffvax.losses.flat_minima import FlatMinimaRegularizer
            self._flat_minima = FlatMinimaRegularizer(self._config)

        # ---- Phase 7: Cross-attention disruption loss ----
        self._attention_loss = None
        if self._config.get("attention_loss", {}).get("enabled", False):
            from diffvax.losses.attention_loss import AttentionDisruptionLoss
            self._attention_loss = AttentionDisruptionLoss(self._config)

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    def immunize_img(self, img, img_mask, epsilon=32):
        """Apply immunization perturbation to image."""
        img_f = img.float().cuda()
        unet_out = self.unetmodel.forward(img_f)
        unet_out = unet_out.half().cuda()
        img_adv = torch.clamp(img + unet_out, self.clamp_min, self.clamp_max)
        return img_adv, unet_out

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_immunization_all_images_batch(
        self,
        entries,
        data_dir,
        images_subdir,
        masks_subdir,
        size,
        target_image=None,
        iter_num=2000,
        SEED=5,
        batch_size=2,
        alpha=1,
        loss_type="l2",
        sd_target_resolutions=None,
        strength_range=None,
    ):
        if sd_target_resolutions is None:
            sd_target_resolutions = [512]
        if strength_range is None:
            strength_range = [0.5, 1.0]
        set_seed_lib(SEED)

        # Memory-efficient SDPA backward is not implemented in all PyTorch
        # versions; disable it so the runtime prefers flash attention (which
        # supports first-order backward) with math as a fallback.
        if torch.cuda.is_available():
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        total_iter = 0

        models_dir = os.path.join(self.output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        existing_folders = [
            d for d in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, d)) and d.isdigit()
        ]
        last_idx = max([int(x) for x in existing_folders], default=0) + 1
        run_dir = os.path.join(models_dir, str(last_idx))
        os.makedirs(run_dir, exist_ok=True)

        if self.load_existing:
            path_of_models = os.path.join(
                models_dir,
                f"sd15_all_images_half_mult_img_mult_prompt_immunization_model_"
                f"{self.model_name}_iter_{iter_num + self.existing_iter_num}_"
                f"alpha_{alpha}_loss_{loss_type}",
            )
        else:
            path_of_models = os.path.join(
                run_dir,
                f"sd15_all_images_{self.model_name}_iter_{iter_num}_alpha_{alpha}"
                f"_loss_{loss_type}_batch_{batch_size}",
            )

        # Pull adaptive ensemble settings
        adaptive_cfg = self._config.get("adaptive_ensemble", {})
        adaptive_enabled = adaptive_cfg.get("enabled", False)
        update_period = int(adaptive_cfg.get("update_period", 50))

        # Pull flat-minima lambda (Phase 6)
        lambda_flat = float(
            self._config.get("flat_minima", {}).get("lambda_flat", 0.01)
        )

        # Pull attention loss weight (Phase 7)
        attn_weight = float(
            self._config.get("attention_loss", {}).get("weight", 0.3)
        )
        attn_only_with_dit = bool(
            self._config.get("attention_loss", {}).get("only_with_dit", True)
        )
        dit_model_names = {"sd3", "flux"}

        dataset = ImmunizationDataset(
            entries, data_dir, images_subdir, masks_subdir, size
        )
        dl_cfg = self._config.get("dataloader", {})
        num_workers = int(dl_cfg.get("num_workers", 4))

        def _make_dataloader(ds, bs):
            """Create a fresh DataLoader with the given dataset and batch size.

            Must be recreated on every curriculum stage change so that:
            1. Worker subprocesses are forked after set_resolution() and pick up
               the new _current_size (stale workers hold the old value).
            2. The new per-stage batch size is applied immediately.
            """
            pf = int(dl_cfg.get("prefetch_factor", 4)) if num_workers > 0 else None
            return torch.utils.data.DataLoader(
                ds, batch_size=bs, shuffle=True,
                num_workers=num_workers, pin_memory=True,
                prefetch_factor=pf,
            )

        # Initialise with the epoch-0 curriculum resolution and batch size.
        _curr_resolution = self._curriculum.get_resolution(0)
        _curr_dl_batch = self._curriculum.get_batch_size(0)
        dataset.set_resolution(_curr_resolution)
        dataloader = _make_dataloader(dataset, _curr_dl_batch)

        # ---- Hub + reporting setup ----
        from diffvax.reporter import TrainingReporter
        reporter = TrainingReporter(self._config, self.output_dir)

        hub_cfg = self._config.get("hub", {})
        hub_enabled = hub_cfg.get("enabled", False)
        hub_repo_id = hub_cfg.get("repo_id") or None
        hub_private = bool(hub_cfg.get("private", True))
        hub_token = hub_cfg.get("token") or os.environ.get("HF_TOKEN") or None
        hub_upload_every_n = int(hub_cfg.get("upload_every_n_epochs", 10000))

        best_loss = float("inf")
        best_model_path: str = path_of_models + "_best.pth"
        epoch_avg_loss = float("inf")  # updated each epoch for use after the loop

        batch_iter_count = 0  # global batch counter for adaptive weight updates

        for epoch_i in range(iter_num):
            # ---- Phase 4: curriculum resolution + batch-size update ----
            curriculum_resolution = self._curriculum.get_resolution(epoch_i)
            curriculum_batch_size = self._curriculum.get_batch_size(epoch_i)
            if (curriculum_resolution != _curr_resolution
                    or curriculum_batch_size != _curr_dl_batch):
                dataset.set_resolution(curriculum_resolution)
                _curr_resolution = curriculum_resolution
                _curr_dl_batch = curriculum_batch_size
                # Recreate DataLoader: spawns fresh workers that inherit the
                # updated _current_size, and applies the new per-stage batch size.
                dataloader = _make_dataloader(dataset, _curr_dl_batch)
                tqdm.write(
                    f"[Curriculum] Stage change → {curriculum_resolution}px "
                    f"batch={curriculum_batch_size}"
                )

            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            epoch_losses = []
            epoch_losses1 = []
            epoch_losses2 = []
            per_model_losses = {}  # {model_name: {"loss": [], "loss1": [], "loss2": []}}

            for i, (img_batch, mask_batch, prompt_batch, flux_prompt_batch) in enumerate(dataloader):
                self.optimizer.zero_grad()
                losses = []
                losses1 = []
                losses2 = []
                cur_iter = i + self.existing_iter_num

                # Select model for this batch
                model_name, attack_model = self.attack_manager.select_and_load()

                # Pick prompt set based on active model
                cur_prompt = flux_prompt_batch if model_name == "flux" else prompt_batch

                img_batch = img_batch.cuda()
                mask_batch = mask_batch.cuda()

                mask_batch.requires_grad = False
                img_batch.requires_grad_()

                ones = torch.ones_like(mask_batch)

                # Perturbation: always full image (no mask gating)
                img_f = img_batch.float().cuda()
                unet_out = self.unetmodel.forward(img_f)
                unet_out = unet_out.half().cuda()
                img_adv = torch.clamp(
                    img_batch + unet_out, self.clamp_min, self.clamp_max
                )

                # ---- Phase 1: EoT augmentation ----
                # EoT is applied to img_adv before passing to the attack model.
                # The noise loss (loss2) is still computed against the un-augmented
                # img_adv to penalize perturbation magnitude in pixel space.
                if self._eot is not None:
                    img_adv_aug = self._eot(img_adv)
                else:
                    img_adv_aug = img_adv

                # Sample strength for this batch
                strength = random.uniform(*strength_range)

                # ---- Phase 7: Register attention hooks for DiT models ----
                attn_active = (
                    self._attention_loss is not None
                    and (not attn_only_with_dit or model_name in dit_model_names)
                )
                if attn_active:
                    transformer = getattr(
                        getattr(attack_model, "pipe", None), "transformer", None
                    )
                    if transformer is not None:
                        self._attention_loss.register_hooks(transformer)

                # Differentiable resize and attack (Phases 3 + 4)
                h, w = img_batch.shape[2], img_batch.shape[3]
                actual_bs = img_batch.shape[0]

                if attack_model.is_inpainting:
                    # Inpainting model: mask-conditioned, multi-resolution downsample.
                    # Apply mask on GPU: zero out inpaint region so the pipeline sees a
                    # masked image (inpaint region = 0) as expected by inpainting models.
                    attack_mask = mask_batch
                    sd_target = random.choice(sd_target_resolutions)
                    if sd_target < h:
                        img_adv_resized = F.interpolate(
                            img_adv_aug, (sd_target, sd_target),
                            mode="bilinear", align_corners=False,
                        )
                        mask_resized = F.interpolate(
                            attack_mask, (sd_target, sd_target), mode="nearest"
                        )
                        img_adv_sd = img_adv_resized * (1.0 - mask_resized)
                        img_out_small = attack_model.attack(
                            prompt=cur_prompt,
                            image=img_adv_sd,
                            mask=mask_resized,
                            height=sd_target,
                            width=sd_target,
                            num_inference_steps=4,
                            batch_size=actual_bs,
                            strength=strength,
                        )
                        img_out = F.interpolate(
                            img_out_small, (h, w), mode="bilinear", align_corners=False
                        )
                    else:
                        img_adv_sd = img_adv_aug * (1.0 - attack_mask)
                        img_out = attack_model.attack(
                            prompt=cur_prompt,
                            image=img_adv_sd,
                            mask=attack_mask,
                            height=h,
                            width=w,
                            num_inference_steps=4,
                            batch_size=actual_bs,
                            strength=strength,
                        )
                else:
                    # FLUX, SD3, or other full-image models
                    # Phase 4: native-resolution dispatch
                    native_res = attack_model.native_resolution
                    if h > native_res:
                        # Resize down to native resolution for the attack pass
                        img_input = F.interpolate(
                            img_adv_aug, (native_res, native_res),
                            mode="bilinear", align_corners=False,
                        )
                        img_out_native = attack_model.attack(
                            prompt=cur_prompt,
                            image=img_input,
                            height=native_res,
                            width=native_res,
                            num_inference_steps=4,
                            batch_size=actual_bs,
                            strength=strength,
                        )
                        img_out = F.interpolate(
                            img_out_native, (h, w), mode="bilinear", align_corners=False
                        )
                    else:
                        img_out = attack_model.attack(
                            prompt=cur_prompt,
                            image=img_adv_aug,
                            height=h,
                            width=w,
                            num_inference_steps=4,
                            batch_size=actual_bs,
                            strength=strength,
                        )

                # ---- Loss computation ----
                resolution = h
                target_image_t = torch.zeros_like(img_out).cuda()

                if attack_model.loss_uses_mask_weighting:
                    loss1_weight = mask_batch
                else:
                    loss1_weight = ones

                loss1_weight_norm = loss1_weight / resolution
                loss2_weight_norm = ones / resolution

                # Existing losses (unchanged)
                loss1 = (
                    ((img_out - target_image_t) * loss1_weight_norm).norm(p=1)
                    / loss1_weight_norm.sum()
                )
                loss2 = (
                    (alpha * (img_adv - img_batch) * loss2_weight_norm).norm(p=2)
                    / loss2_weight_norm.sum()
                )

                # ---- Phase 2: CLIP disruption loss ----
                loss_clip_val = 0.0
                if self._loss_composer is not None and self._loss_composer.has_terms():
                    loss_extra, clip_breakdown = self._loss_composer.compute(
                        img_orig=img_batch,
                        img_adv=img_adv,
                        img_out=img_out,
                        prompts=cur_prompt,
                    )
                    loss_clip_val = clip_breakdown.get("clip", 0.0)
                else:
                    loss_extra = torch.tensor(0.0, device="cuda")

                # ---- Phase 7: Attention disruption loss ----
                loss_attn = torch.tensor(0.0, device="cuda")
                if attn_active:
                    loss_attn = self._attention_loss.compute()
                # Always remove hooks after the forward pass to avoid
                # stale hooks accumulating when the next batch uses a different model.
                if self._attention_loss is not None:
                    self._attention_loss.remove_hooks()

                # Aggregate loss
                loss = loss1 + loss2 + loss_extra + attn_weight * loss_attn

                # Log scalar values (after building the full computation graph)
                loss1_val = loss1.item()
                loss2_val = loss2.item()

                losses.append(loss.item())
                losses1.append(loss1_val)
                losses2.append(loss2_val)
                epoch_losses.append(loss.item())
                epoch_losses1.append(loss1_val)
                epoch_losses2.append(loss2_val)

                if model_name not in per_model_losses:
                    per_model_losses[model_name] = {"loss": [], "loss1": [], "loss2": []}
                per_model_losses[model_name]["loss"].append(loss.item())
                per_model_losses[model_name]["loss1"].append(loss1_val)
                per_model_losses[model_name]["loss2"].append(loss2_val)

                scaler.scale(loss).backward()

                # ---- Phase 6: Flat-minima regularization (post-backward) ----
                # Applied after backward so only first-order derivatives are
                # needed, allowing flash attention instead of O(N²) math SDPA.
                if self._flat_minima is not None:
                    self._flat_minima.apply(self.unetmodel, lambda_flat)

                # ---- Phase 5: Record gradient for adaptive weighting ----
                if adaptive_enabled and self.attack_manager.adaptive:
                    grad_vecs = [
                        p.grad.detach().flatten()
                        for p in self.unetmodel.parameters()
                        if p.grad is not None
                    ]
                    if grad_vecs:
                        grad_vec = torch.cat(grad_vecs)
                        self.attack_manager.record_gradient(model_name, grad_vec)

                scaler.step(self.optimizer)
                scaler.update()

                batch_iter_count += 1

                # ---- Phase 5: Periodically update adaptive weights ----
                if (
                    adaptive_enabled
                    and self.attack_manager.adaptive
                    and batch_iter_count % update_period == 0
                ):
                    self.attack_manager.update_weights()

                total_iter += batch_size

                pbar.set_description_str(
                    f"AVG Loss: {np.mean(losses):.5f} "
                    f"Loss1: {np.mean(losses1):.5f} "
                    f"Loss2: {np.mean(losses2):.5f}"
                    + (f" CLIP: {loss_clip_val:.5f}" if loss_clip_val else "")
                )
                pbar.update(1)

                if torch.isnan(loss):
                    if self._attention_loss is not None:
                        self._attention_loss.remove_hooks()
                    torch.save(
                        self.model.state_dict(),
                        path_of_models + f"iter_{cur_iter}_early.pth",
                    )
                    return

                losses = []
                losses1 = []
                losses2 = []

            # ---- Per-epoch summary ----
            epoch_avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
            per_model_avg = {
                mn: float(np.mean(per_model_losses[mn]["loss"]))
                for mn in per_model_losses
                if per_model_losses[mn]["loss"]
            }
            parts = [f"Epoch {epoch_i}  avg={epoch_avg_loss:.5f}"]
            for mn in sorted(per_model_losses):
                m = per_model_losses[mn]
                parts.append(
                    f"[{mn}] loss={np.mean(m['loss']):.4f} "
                    f"loss1={np.mean(m['loss1']):.4f} "
                    f"loss2={np.mean(m['loss2']):.4f} "
                    f"(n={len(m['loss'])})"
                )
            tqdm.write("  ".join(parts))
            reporter.report_epoch(epoch_i, epoch_avg_loss, per_model_avg)

            # ---- Periodic local checkpoint ----
            if (epoch_i + 1) % reporter.checkpoint_every == 0:
                ckpt_path = path_of_models + f"_epoch{epoch_i}.pth"
                torch.save(self.model.state_dict(), ckpt_path)
                reporter.report_checkpoint(epoch_i, epoch_avg_loss, ckpt_path, is_best=False)
                tqdm.write(f"[Checkpoint] Saved periodic checkpoint: {ckpt_path}")

            # ---- Best-model checkpoint + Hub upload ----
            if epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                torch.save(self.model.state_dict(), best_model_path)
                reporter.report_checkpoint(epoch_i, best_loss, best_model_path, is_best=True)
                tqdm.write(f"[Checkpoint] New best model (loss={best_loss:.5f}): {best_model_path}")
                if hub_enabled and hub_repo_id:
                    try:
                        self.unetmodel.push_to_hub(
                            repo_id=hub_repo_id,
                            private=hub_private,
                            token=hub_token,
                            commit_message=(
                                f"Epoch {epoch_i}: best checkpoint (loss={best_loss:.5f})"
                            ),
                        )
                        tqdm.write(f"[Hub] Uploaded best checkpoint to {hub_repo_id}")
                    except Exception as exc:
                        tqdm.write(f"[Hub] Best-model upload failed: {exc}")

            # ---- Periodic Hub upload ----
            if hub_enabled and hub_repo_id and (epoch_i + 1) % hub_upload_every_n == 0:
                try:
                    self.unetmodel.push_to_hub(
                        repo_id=hub_repo_id,
                        private=hub_private,
                        token=hub_token,
                        commit_message=(
                            f"Epoch {epoch_i}: periodic checkpoint (loss={epoch_avg_loss:.5f})"
                        ),
                    )
                    tqdm.write(f"[Hub] Uploaded periodic checkpoint to {hub_repo_id}")
                except Exception as exc:
                    tqdm.write(f"[Hub] Periodic upload failed: {exc}")

        # ---- Final save ----
        final_path = path_of_models + "_final.pth"
        torch.save(self.model.state_dict(), final_path)
        if hub_enabled and hub_repo_id:
            try:
                self.unetmodel.push_to_hub(
                    repo_id=hub_repo_id,
                    private=hub_private,
                    token=hub_token,
                    commit_message=f"Training complete: final model (loss={epoch_avg_loss:.5f})",
                )
                tqdm.write(f"[Hub] Uploaded final model to {hub_repo_id}")
            except Exception as exc:
                tqdm.write(f"[Hub] Final upload failed: {exc}")
        reporter.report_complete(iter_num, epoch_avg_loss, final_path)
        tqdm.write(
            f"[Training complete] {iter_num} epochs | final loss={epoch_avg_loss:.5f} | "
            f"best loss={best_loss:.5f} | model: {final_path}"
        )

        return img_adv, final_path

    def edit_image(
        self,
        prompt,
        img,
        img_mask,
        num_inf=30,
        SEED=5,
        generator=None,
    ):
        """Edit image using the diffusion model."""
        strength = 1.0
        guidance_scale = 7.5
        self.generator.manual_seed(SEED)

        # Use first available attack model for editing
        if self.attack_model is not None:
            model = self.attack_model.model
        else:
            # Pick the SD model from the manager if available
            for name, m in self.attack_manager.models.items():
                if name == "sd":
                    model = m.model
                    break
            else:
                # Fallback: use first model
                _, m = next(iter(self.attack_manager.models.items()))
                model = m.model

        edited_image = model(
            prompt=prompt,
            image=img,
            mask_image=img_mask,
            eta=1,
            num_inference_steps=num_inf,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=self.generator,
        ).images

        return edited_image
