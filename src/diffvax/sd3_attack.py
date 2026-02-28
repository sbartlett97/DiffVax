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

    def __init__(self, model_link: str, strength: float = 0.75):
        from diffusers import StableDiffusion3Img2ImgPipeline

        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            model_link, torch_dtype=torch.float16
        )
        self.model_link = model_link
        self.strength = strength

        # Freeze all parameters — gradient flows through activations only
        self.pipe.vae.requires_grad_(False)
        self.pipe.transformer.requires_grad_(False)
        for enc_attr in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
            enc = getattr(self.pipe, enc_attr, None)
            if enc is not None:
                enc.requires_grad_(False)

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
    def vae_channels(self) -> int:
        """SD3 uses a 16-channel VAE."""
        return 16

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
        mask: torch.FloatTensor,
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
            mask:               Not used (SD3 img2img edits the full image).
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

        # ------ 5. MM-DiT denoising loop ------
        for t in timesteps:
            # CFG: duplicate latents for uncond + cond
            latent_input = torch.cat([noisy_latents] * 2, dim=0)
            timestep = t.expand(latent_input.shape[0])

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

        # ------ 6. VAE decode ------
        latents_out = noisy_latents / vae_scaling_factor + vae_shift_factor
        output = vae.decode(latents_out.to(dtype), return_dict=False)[0]

        return output.half()
