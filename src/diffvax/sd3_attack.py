"""Differentiable SD3 / SD3.5 attack for 16-channel VAE training (Phase 3).

SD3 and FLUX share a 16-channel VAE family that is fundamentally incompatible
with SD 1.5's 4-channel VAE. Perturbations optimized exclusively against
SD 1.5 provide minimal protection against SD3 or FLUX because the latent
spaces are disjoint. This module wraps SD3/SD3.5 as a differentiable
BaseAttack so the training ensemble can include it alongside SD 1.5 and FLUX.

Key architectural differences from the SD 1.5 Attack class:
  - 16-channel VAE (no quant_conv layers)
  - Model-specific scaling/shift factors (not 0.18215)
  - MM-DiT transformer (joint bidirectional attention)
  - Triple text encoders: T5-XXL + CLIP-G + CLIP-L
  - Rectified flow scheduler (velocity-prediction, logit-normal timesteps)

Gradient flow: input image → 16-ch VAE encode (mode()) → rectified flow
noise injection → MM-DiT denoising → 16-ch VAE decode → output.
"""

import torch
from typing import Union, List

from diffvax.attack_base import BaseAttack


class SD3Attack(BaseAttack):
    """Differentiable img2img forward pass through SD3 / SD3.5.

    Uses StableDiffusion3Img2ImgPipeline from diffusers. The VAE encode
    uses .latent_dist.mode() (mean of the posterior) instead of .sample()
    for deterministic, gradient-preserving latent computation.

    All pipeline parameters (VAE, transformer, text encoders) are frozen.
    Only the image path carries gradients.

    Args:
        model_link: HuggingFace model ID or local path for the SD3/SD3.5 model.
        strength:   Default img2img denoising strength (unused directly;
                    per-call strength is passed to attack()).
    """

    def __init__(self, model_link: str, strength: float = 0.75,
                 gradient_timestep_fraction: float = 1.0,
                 token_gradient_regularization: bool = False):
        from diffusers import StableDiffusion3Img2ImgPipeline

        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            model_link, torch_dtype=torch.float16
        )
        self.model_link = model_link
        self.strength = strength
        # H2: partial-timestep gradient — fraction of timesteps that backprop.
        # Early (high-sigma) timesteps determine global structure and are most
        # critical for protection. 0.5 = first 50% of steps get gradients.
        # Basis: "Distraction Is All You Need" CVPR 2024.
        self._gradient_timestep_fraction = float(gradient_timestep_fraction)
        # H4: TGR — token-wise gradient normalization during backward pass.
        # Reduces gradient variance across the 18k-token joint attention at 1088px.
        # Basis: Token Gradient Regularization, CVPR 2023 (arXiv:2303.15754).
        self._tgr_enabled = bool(token_gradient_regularization)
        self._tgr_hooks: list = []

        # Freeze all parameters — gradient flows through activations only
        self.pipe.vae.requires_grad_(False)
        self.pipe.transformer.requires_grad_(False)
        for enc_attr in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
            enc = getattr(self.pipe, enc_attr, None)
            if enc is not None:
                enc.requires_grad_(False)

    # ------------------------------------------------------------------
    # H4: Token Gradient Regularization helpers
    # ------------------------------------------------------------------

    def _register_tgr_hooks(self, transformer) -> None:
        """Register backward hooks on transformer blocks for TGR.

        Each hook normalizes the per-token gradient to unit norm, reducing
        the token-to-token variance that causes poor adversarial transfer
        in high-token-count attention (TGR, CVPR 2023, arXiv:2303.15754).
        """
        self._remove_tgr_hooks()
        blocks = getattr(transformer, "transformer_blocks", None) or []
        for block in blocks:
            # Hook on the block output so gradient normalization applies
            # to the full residual stream exiting each transformer block.
            h = block.register_backward_hook(self._tgr_backward_hook)
            self._tgr_hooks.append(h)

    @staticmethod
    def _tgr_backward_hook(module, grad_input, grad_output):
        """Normalize per-token gradient magnitude to reduce variance.

        grad_output[0] shape: (B, seq_len, dim)
        We normalize across the feature dim so each token's gradient
        vector has unit L2 norm, following TGR (arXiv:2303.15754).
        """
        normed = []
        for g in grad_output:
            if g is None or g.ndim < 3:
                normed.append(g)
                continue
            norm = g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            normed.append(g / norm)
        return tuple(normed)

    def _remove_tgr_hooks(self) -> None:
        for h in self._tgr_hooks:
            h.remove()
        self._tgr_hooks = []

    # ------------------------------------------------------------------
    # BaseAttack interface
    # ------------------------------------------------------------------

    def to_device(self, device: str) -> None:
        self.pipe = self.pipe.to(device)

    def to_cpu(self) -> None:
        self.pipe.to("cpu")
        torch.cuda.empty_cache()

    @property
    def loss_uses_mask_weighting(self) -> bool:
        # SD3 img2img edits the full image; no mask-specific loss weighting.
        return False

    @property
    def is_inpainting(self) -> bool:
        # SD3 is a full-image img2img model; no mask is used.
        return False

    @property
    def vae_channels(self) -> int:
        """SD3 uses a 16-channel VAE."""
        return 16

    def get_vae(self):
        """Return the 16-channel VAE for latent-space disruption loss."""
        return self.pipe.vae

    @property
    def native_resolution(self) -> int:
        """SD3 operates natively at 1024×1024."""
        return 1024

    # ------------------------------------------------------------------
    # Main attack
    # ------------------------------------------------------------------

    def attack(
        self,
        prompt: Union[str, List[str]],
        image: torch.FloatTensor,
        mask: torch.FloatTensor = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 4,
        batch_size: int = 1,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """Differentiable SD3 img2img forward pass.

        Gradient path:
            image → vae.encode().mode() → noise injection → MM-DiT denoising
            → vae.decode() → output (fp16)

        Args:
            prompt:             Edit prompts (list of strings, length batch_size).
            image:              Adversarial image tensor (B, 3, H, W), in [-1, 1].
            mask:               Not used; defaults to None. SD3 img2img edits
                                the full image without a mask.
            height, width:      Output resolution. SD3 supports 512/768/1024.
            num_inference_steps:Number of rectified-flow denoising steps.
            batch_size:         Number of images in the batch.
            strength:           Denoising strength (1.0 = full noise from scratch).

        Returns:
            Generated image tensor (B, 3, H, W) in float16.
        """
        device = self.pipe.device
        vae = self.pipe.vae
        transformer = self.pipe.transformer
        scheduler = self.pipe.scheduler
        dtype = next(vae.parameters()).dtype

        # ------ 1. Text encoding (detached — no gradient through text path) ------
        if isinstance(prompt, (tuple, list)):
            prompt_list = list(prompt)
        else:
            prompt_list = [str(prompt)] * batch_size

        with torch.no_grad():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt=prompt_list,
                prompt_2=None,
                prompt_3=None,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=None,
            )
            # Concatenate unconditional + conditional for CFG
            prompt_embeds_cfg = torch.cat(
                [negative_prompt_embeds, prompt_embeds], dim=0
            )
            pooled_embeds_cfg = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )

        # Offload text encoders to CPU — gradient flows only through VAE + transformer,
        # so T5-XXL + CLIP-G/L (~10-12 GB) don't need to be in VRAM during backprop.
        for enc_attr in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
            enc = getattr(self.pipe, enc_attr, None)
            if enc is not None:
                enc.to("cpu")
        torch.cuda.empty_cache()

        # ------ 2. VAE encode with gradient flow via mode() ------
        image_input = image.to(device=device, dtype=dtype)
        latents = vae.encode(image_input).latent_dist.mode()

        # SD3 VAE scaling factors (read from config; defaults match SD3.5)
        vae_scaling_factor = float(
            getattr(vae.config, "scaling_factor", 1.5305)
        )
        vae_shift_factor = float(
            getattr(vae.config, "shift_factor", 0.0609)
        )
        latents = (latents - vae_shift_factor) * vae_scaling_factor

        # ------ 3. Timestep schedule setup ------
        scheduler.set_timesteps(num_inference_steps, device=device)
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:]

        # ------ 4. Add noise using rectified flow interpolation ------
        noise = torch.randn_like(latents)
        if t_start < len(scheduler.sigmas):
            sigma = scheduler.sigmas[t_start].to(dtype)
            noisy_latents = (1.0 - sigma) * latents + sigma * noise
        else:
            noisy_latents = latents

        guidance_scale = 7.0

        # H2: partial-timestep gradient — how many of the denoising steps receive
        # gradients. Early steps (high sigma) shape global structure; later steps
        # refine fine detail. Using only the first k steps for backprop saves VRAM
        # proportionally to the skipped fraction.
        n_grad_steps = max(1, int(len(timesteps) * self._gradient_timestep_fraction))

        # H4: TGR — register backward hooks on transformer blocks that normalize
        # per-token gradient magnitude, reducing variance across the joint-attention
        # sequence at high resolution (18k+ tokens at 1088px).
        if self._tgr_enabled:
            self._register_tgr_hooks(transformer)

        # ------ 5. MM-DiT denoising loop ------
        from torch.utils.checkpoint import checkpoint as grad_checkpoint

        for step_idx, t in enumerate(timesteps):
            # Only backpropagate through the first n_grad_steps timesteps.
            # Later steps run under no_grad to reduce backward-graph VRAM.
            use_grad = step_idx < n_grad_steps
            grad_ctx = torch.enable_grad() if use_grad else torch.no_grad()

            with grad_ctx:
                # CFG: duplicate latents for uncond + cond
                latent_input = torch.cat([noisy_latents] * 2, dim=0)
                timestep = t.expand(latent_input.shape[0])

                if use_grad:
                    # Gradient checkpointing: discard transformer intermediate
                    # activations during forward and recompute them during backward.
                    # Reduces activation VRAM from ~6 GB/step to ~200 MB/step for
                    # the 24-block MM-DiT at 512px batch=4 with CFG doubling.
                    def _transformer_fwd(hs, ts, enc_hs, pooled):
                        return transformer(
                            hidden_states=hs,
                            timestep=ts,
                            encoder_hidden_states=enc_hs,
                            pooled_projections=pooled,
                            return_dict=False,
                        )[0]
                    noise_pred = grad_checkpoint(
                        _transformer_fwd,
                        latent_input, timestep, prompt_embeds_cfg, pooled_embeds_cfg,
                        use_reentrant=False,
                    )
                else:
                    noise_pred = transformer(
                        hidden_states=latent_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds_cfg,
                        pooled_projections=pooled_embeds_cfg,
                        return_dict=False,
                    )[0]

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                noisy_latents = scheduler.step(
                    noise_pred, t, noisy_latents, return_dict=False
                )[0]

        if self._tgr_enabled:
            self._remove_tgr_hooks()

        # ------ 6. VAE decode ------
        latents_out = noisy_latents / vae_scaling_factor + vae_shift_factor
        output = vae.decode(latents_out.to(dtype), return_dict=False)[0]

        return output.half()
