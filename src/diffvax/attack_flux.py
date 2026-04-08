"""Differentiable FLUX inpainting attack wrapper for DiffVax training.

Wraps the FLUX.1 / FLUX.2 inpainting pipeline to expose a differentiable
forward pass suitable for backpropagating gradients through the immunization
network. Mirrors the interface of attack.py (SD 1.5).

FLUX architecture notes:
  - 16-channel VAE, 8× spatial compression
  - Packs (H/8, W/8) spatial latents into a token sequence for the transformer
  - Uses FlowMatching scheduler (not DDPM/DDIM)
  - Text conditioning: dual-stream CLIP + T5 embeddings
  - No negative prompt / CFG in distilled variants (Schnell, Klein)
"""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import Union, List, Optional


def _pack_latents(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """Pack (B, C, H, W) into transformer sequence (B, H/p * W/p, C*p*p)."""
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    latents = latents.permute(0, 2, 4, 1, 3, 5)  # B, H/p, W/p, C, p, p
    latents = latents.reshape(B, (H // patch_size) * (W // patch_size), C * patch_size * patch_size)
    return latents


def _unpack_latents(
    latents: torch.Tensor, height: int, width: int, vae_scale_factor: int = 8, patch_size: int = 2
) -> torch.Tensor:
    """Unpack transformer sequence (B, seq, C*p*p) back to (B, C, H, W)."""
    B, seq, inner = latents.shape
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor
    C = inner // (patch_size * patch_size)
    latents = latents.view(B, latent_h // patch_size, latent_w // patch_size, C, patch_size, patch_size)
    latents = latents.permute(0, 3, 1, 4, 2, 5)  # B, C, H/p, p, W/p, p
    latents = latents.reshape(B, C, latent_h, latent_w)
    return latents


def _prepare_image_ids(height: int, width: int, vae_scale_factor: int = 8, patch_size: int = 2) -> torch.Tensor:
    """Build image positional IDs expected by the FLUX transformer."""
    latent_h = height // vae_scale_factor // patch_size
    latent_w = width // vae_scale_factor // patch_size
    ids = torch.zeros(latent_h * latent_w, 3, dtype=torch.float32)
    ids[:, 1] = torch.arange(latent_h).repeat_interleave(latent_w)
    ids[:, 2] = torch.arange(latent_w).repeat(latent_h)
    return ids  # (seq_len, 3)


class FluxAttack:
    """Differentiable FLUX inpainting forward pass for DiffVax training.

    Usage::

        attack = FluxAttack("black-forest-labs/FLUX.1-schnell")
        edited = attack.attack(
            prompt="a dog in a park",
            masked_image=img_tensor,
            mask=mask_tensor,
            height=512, width=512,
        )
        loss = (edited * mask - target).norm()
        loss.backward()

    Args:
        model_link: HuggingFace repo ID for a FLUX inpainting pipeline.
        guidance_scale: Guidance scale. Set 0 for distilled (Schnell/Klein) variants.
        dtype: Model dtype (bfloat16 recommended for FLUX).
    """

    def __init__(
        self,
        model_link: str,
        guidance_scale: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        from diffusers import FluxInpaintPipeline

        # guidance_scale=0.0 is correct for distilled FLUX models (schnell, fill-schnell).
        # For non-distilled models (dev), use guidance_scale=3.5.
        # Auto-detect if not explicitly set: distilled if "schnell" in link.
        if guidance_scale == 0.0 and "schnell" not in model_link.lower():
            guidance_scale = 3.5

        try:
            pipe = FluxInpaintPipeline.from_pretrained(model_link, torch_dtype=dtype)
        except ValueError as e:
            if "were passed" in str(e):
                # Some FLUX variants (T5-only, no CLIP) lack text_encoder_2/tokenizer_2.
                # Load with optional components set to None — prompt encoding falls back to T5 only.
                pipe = FluxInpaintPipeline.from_pretrained(
                    model_link, torch_dtype=dtype,
                    text_encoder_2=None, tokenizer_2=None,
                    feature_extractor=None, image_encoder=None,
                )
            else:
                raise

        pipe = pipe.to("cuda")
        pipe.vae.enable_slicing()
        # Gradient checkpointing: recompute transformer activations during backprop
        # instead of storing them. Trades ~10× compute for ~8-10× activation memory
        # reduction. Required when backpropagating through FLUX during training.
        pipe.transformer.enable_gradient_checkpointing()

        self.pipe = pipe
        self.guidance_scale = guidance_scale
        self.dtype = dtype
        self.model_link = model_link

        # VAE scale factor (typically 8 for FLUX)
        self.vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        self.patch_size = 2  # FLUX uses 2×2 patch packing

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ):
        """Encode text prompt(s) with CLIP + T5 encoders.

        Returns:
            prompt_embeds: (B, seq, hidden) — T5 token embeddings
            pooled_prompt_embeds: (B, hidden) — CLIP pooled embedding
        """
        device = device or self.pipe.device
        if isinstance(prompt, str):
            prompt = [prompt] * batch_size

        prompt_embeds, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=device,
            num_images_per_prompt=1,
        )
        return prompt_embeds.detach(), pooled_prompt_embeds.detach()

    # ------------------------------------------------------------------
    # Differentiable attack
    # ------------------------------------------------------------------

    def attack(
        self,
        prompt: Union[str, List[str]],
        masked_image: torch.Tensor,
        mask: torch.Tensor,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 4,
        strength: float = 0.85,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Differentiable forward pass through FLUX inpainting.

        Args:
            prompt: editing prompt(s).
            masked_image: image tensor in [-1, 1], shape (B, 3, H, W).
            mask: mask tensor in [0, 1], shape (B, 1, H, W), 1 = edit region.
            height: image height (must be divisible by vae_scale_factor * patch_size).
            width: image width.
            num_inference_steps: denoising steps (4 for training efficiency).
            strength: inpainting strength in [0, 1].
            batch_size: number of images.

        Returns:
            Decoded edited image tensor in [-1, 1], shape (B, 3, H, W).
        """
        pipe = self.pipe
        device = pipe.device

        if isinstance(prompt, (list, tuple)) and len(prompt) == 1:
            prompt = prompt * batch_size

        # --- Text embeddings ---
        prompt_embeds, pooled_embeds = self.encode_prompt(prompt, batch_size, device)

        # --- VAE encode masked image ---
        masked_image_f = masked_image.to(dtype=self.dtype, device=device)
        # Scale from [-1, 1] to VAE input convention
        masked_latents = pipe.vae.encode(masked_image_f).latent_dist.sample()
        masked_latents = (masked_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

        # --- Encode original (unmasked) image for flow matching start ---
        image_latents = pipe.vae.encode(
            masked_image_f
        ).latent_dist.sample()
        image_latents = (image_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

        # --- Prepare mask in latent space ---
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        mask_latent = F.interpolate(
            mask.to(device=device, dtype=self.dtype),
            size=(latent_h, latent_w),
            mode="nearest",
        )

        # --- Add noise (flow matching: x_t = (1-t)*x_0 + t*noise) ---
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        # Honour `strength`: start from later timesteps
        num_steps_with_strength = max(1, int(num_inference_steps * strength))
        timesteps = pipe.scheduler.timesteps[-num_steps_with_strength:]

        noise = torch.randn(image_latents.shape, generator=generator,
                           device=image_latents.device, dtype=image_latents.dtype)
        t_start = timesteps[0]
        sigma = t_start.float() / pipe.scheduler.config.num_train_timesteps
        latents = (1.0 - sigma) * image_latents + sigma * noise

        # --- Image positional IDs ---
        img_ids = _prepare_image_ids(height, width, self.vae_scale_factor, self.patch_size)
        img_ids = img_ids.to(device=device, dtype=self.dtype)

        txt_ids = torch.zeros(
            prompt_embeds.shape[1], 3, device=device, dtype=self.dtype
        )

        # --- Denoising loop ---
        for i, t in enumerate(timesteps):
            # Apply mask: keep unmasked region from clean latents
            latents = latents * mask_latent + image_latents * (1 - mask_latent)

            # Pack for transformer
            packed = _pack_latents(latents, self.patch_size)  # (B, seq, C*p*p)

            # Guidance (set 0 for distilled models, use actual value otherwise)
            guidance = (
                torch.full((batch_size,), self.guidance_scale, device=device, dtype=self.dtype)
                if self.guidance_scale > 1
                else None
            )

            t_tensor = t.expand(batch_size).to(device)

            # Transformer forward
            # img_ids / txt_ids: diffusers now expects 2D (seq, 3) — no batch dimension.
            noise_pred_packed = pipe.transformer(
                hidden_states=packed,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                timestep=t_tensor / 1000.0,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=False,
            )[0]

            noise_pred = _unpack_latents(
                noise_pred_packed, height, width, self.vae_scale_factor, self.patch_size
            )

            # Scheduler step
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Final mask apply
        latents = latents * mask_latent + image_latents * (1 - mask_latent)

        # --- VAE decode ---
        latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents).sample
        return image
