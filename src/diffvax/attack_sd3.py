"""Differentiable SD 3.5 inpainting attack wrapper for DiffVax training.

Wraps the Stable Diffusion 3.5 inpainting pipeline (MM-DiT architecture) to
expose a differentiable forward pass for backpropagating gradients through the
immunization network. Mirrors the interface of attack.py (SD 1.5) and
attack_flux.py.

SD 3.5 architecture notes:
  - 16-channel VAE with shift_factor (different from SD 1.5's 4-channel)
  - MM-DiT transformer: interleaved image + text transformer blocks
  - FlowMatching scheduler (same as FLUX)
  - Text conditioning: CLIP-L + CLIP-G + T5-XXL tri-encoder
"""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import Union, List, Optional


class SD3Attack:
    """Differentiable SD 3.5 inpainting forward pass for DiffVax training.

    Args:
        model_link: HuggingFace repo ID for a SD 3.5 inpainting pipeline.
            E.g. "stabilityai/stable-diffusion-3.5-large-inpainting"
        guidance_scale: Classifier-free guidance scale.
        dtype: Model dtype (float16 or bfloat16).
    """

    def __init__(
        self,
        model_link: str,
        guidance_scale: float = 7.0,
        dtype: torch.dtype = torch.float16,
    ):
        from diffusers import StableDiffusion3InpaintPipeline

        pipe = StableDiffusion3InpaintPipeline.from_pretrained(
            model_link, torch_dtype=dtype
        )
        pipe = pipe.to("cuda")
        pipe.enable_model_cpu_offload()

        self.pipe = pipe
        self.guidance_scale = guidance_scale
        self.dtype = dtype
        self.model_link = model_link

        self.vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        negative_prompt: str = "",
    ):
        """Encode text prompt(s) with SD 3.5 tri-encoder (CLIP-L + CLIP-G + T5).

        Returns:
            prompt_embeds: concatenated text embeddings (B, seq, hidden)
            pooled_embeds: pooled CLIP embeddings (B, hidden)
            neg_embeds: negative prompt embeddings (B, seq, hidden)
            neg_pooled: negative pooled embeddings (B, hidden)
        """
        device = device or self.pipe.device
        if isinstance(prompt, str):
            prompt = [prompt] * batch_size
        neg_prompts = [negative_prompt] * batch_size

        (
            prompt_embeds,
            neg_embeds,
            pooled_embeds,
            neg_pooled,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt=neg_prompts,
            negative_prompt_2=neg_prompts,
            negative_prompt_3=neg_prompts,
            device=device,
            num_images_per_prompt=1,
        )
        return (
            prompt_embeds.detach(),
            pooled_embeds.detach(),
            neg_embeds.detach(),
            neg_pooled.detach(),
        )

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
    ) -> torch.Tensor:
        """Differentiable forward pass through SD 3.5 inpainting.

        Args:
            prompt: editing prompt(s).
            masked_image: image tensor in [-1, 1], shape (B, 3, H, W).
            mask: mask tensor in [0, 1], shape (B, 1, H, W), 1 = edit region.
            height: image height.
            width: image width.
            num_inference_steps: denoising steps.
            strength: inpainting strength.
            batch_size: number of images.

        Returns:
            Decoded edited image tensor in [-1, 1], shape (B, 3, H, W).
        """
        pipe = self.pipe
        device = pipe.device

        if isinstance(prompt, (list, tuple)) and len(prompt) == 1:
            prompt = prompt * batch_size

        # --- Text embeddings ---
        prompt_embeds, pooled_embeds, neg_embeds, neg_pooled = self.encode_prompt(
            prompt, batch_size, device
        )

        # --- VAE encode ---
        masked_image_f = masked_image.to(dtype=self.dtype, device=device)
        image_latents = pipe.vae.encode(masked_image_f).latent_dist.sample()
        image_latents = (image_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

        # --- Prepare latent mask ---
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        mask_latent = F.interpolate(
            mask.to(device=device, dtype=self.dtype),
            size=(latent_h, latent_w),
            mode="nearest",
        )
        # Expand to match 16-channel VAE
        mask_latent_16 = mask_latent.expand(-1, image_latents.shape[1], -1, -1)

        # --- Initialize noisy latents ---
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        num_steps_with_strength = max(1, int(num_inference_steps * strength))
        timesteps = pipe.scheduler.timesteps[-num_steps_with_strength:]

        noise = torch.randn_like(image_latents)
        t_start = timesteps[0]
        sigma = t_start.float() / pipe.scheduler.config.num_train_timesteps
        latents = (1.0 - sigma) * image_latents + sigma * noise

        # CFG: concatenate unconditional + conditional embeddings
        do_cfg = self.guidance_scale > 1.0
        if do_cfg:
            cond_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
            cond_pooled = torch.cat([neg_pooled, pooled_embeds], dim=0)
        else:
            cond_embeds = prompt_embeds
            cond_pooled = pooled_embeds

        # --- Denoising loop ---
        for i, t in enumerate(timesteps):
            # Apply inpainting mask: keep unmasked region from clean latents
            latents = latents * mask_latent_16 + image_latents * (1 - mask_latent_16)

            t_tensor = t.expand(batch_size).to(device)

            if do_cfg:
                latent_input = torch.cat([latents] * 2)
                t_input = t_tensor.repeat(2)
            else:
                latent_input = latents
                t_input = t_tensor

            noise_pred = pipe.transformer(
                hidden_states=latent_input,
                encoder_hidden_states=cond_embeds,
                pooled_projections=cond_pooled,
                timestep=t_input / 1000.0,
                return_dict=False,
            )[0]

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Final mask apply
        latents = latents * mask_latent_16 + image_latents * (1 - mask_latent_16)

        # --- VAE decode ---
        latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents).sample
        return image
