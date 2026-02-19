"""Differentiable FLUX.2 Klein attack for adversarial training.

Requires diffusers with FLUX.2 Klein support. Install from source if needed:
    pip install git+https://github.com/huggingface/diffusers.git
"""

import torch
from typing import Union, List

from optimum.quanto import freeze, qint8, quantize
from transformers import T5EncoderModel

from diffvax.attack_base import BaseAttack


def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Compute empirical mu for FlowMatch scheduler timestep shifting.

    Matches compute_empirical_mu in pipeline_flux2_klein.py.
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


class FluxAttack(BaseAttack):
    """Differentiable img2img forward pass through FLUX.2 Klein.

    The attack encodes the input image through the FLUX VAE (maintaining
    gradient flow via mode() instead of sample()), adds noise based on
    the strength parameter, denoises through the transformer, and decodes
    back to pixel space.

    All model parameters are frozen — only the image path carries gradients.
    Output is converted to float16 for consistent loss computation with GradScaler.
    """

    def __init__(self, model_link: str, strength: float = 0.75, transformer_repo: str = None):
        # Lazy import — only fail when someone actually uses FLUX
        try:
            from diffusers import Flux2KleinPipeline as PipeClass
            from diffusers import Flux2Transformer2DModel
        except ImportError:
            try:
                from diffusers import Flux2KleinPipeline as PipeClass
                from diffusers import Flux2Transformer2DModel
            except ImportError:
                raise ImportError(
                    "FLUX support requires diffusers with FLUX pipeline support. "
                    "Install from source: pip install git+https://github.com/huggingface/diffusers.git"
                )

        if transformer_repo is not None:
            # Load the quantized transformer first, then swap into the pipeline
            try:
                transformer = Flux2Transformer2DModel.from_single_file(
                    "https://huggingface.co/vistralis/FLUX.2-klein-4b-INT8-transformer/blob/main/flux-2-klein-4b-int8.safetensors",
                    torch_dtype="bfloat16"
                )
            except Exception:
                transformer = Flux2Transformer2DModel.from_single_file(
                    "https://huggingface.co/vistralis/FLUX.2-klein-4b-INT8-transformer/blob/main/flux-2-klein-4b-int8.safetensors",
                    torch_dtype="float32"
                )
            # Load VAE, text encoder, tokenizer, scheduler from the original
            # repo but swap in the quantized transformer.
            self.pipe = PipeClass.from_pretrained(
                model_link,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )
        else:
            self.pipe = PipeClass.from_pretrained(
                model_link, torch_dtype="bfloat16"
            )

        self.model_link = model_link
        self.strength = strength

        # vae_scale_factor matches pipeline convention: 2**(len-1)
        # The pipeline then uses vae_scale_factor*2 to account for 2x2 patchification.
        self.vae_scale_factor = (
            2 ** (len(self.pipe.vae.config.block_out_channels) - 1)
            if hasattr(self.pipe.vae.config, "block_out_channels")
            else 8
        )

        # Freeze all model parameters (gradient flows through activations only)
        self.pipe.vae.requires_grad_(False)
        self.pipe.transformer.requires_grad_(False)
        # Flux2KleinPipeline uses a single Qwen3 text_encoder, no text_encoder_2
        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
            self.pipe.text_encoder.requires_grad_(False)

    def to_device(self, device: str):
        self.pipe = self.pipe.to(device)

    def to_cpu(self):
        self.pipe.to("cpu")
        torch.cuda.empty_cache()

    @property
    def loss_uses_mask_weighting(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Internal helpers — matching pipeline_flux2_klein.py
    # ------------------------------------------------------------------

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, C*4, H//2, W//2) — 2x2 spatial patchification.

        Matches Flux2KleinPipeline._patchify_latents.
        """
        B, C, H, W = latents.shape
        latents = latents.view(B, C, H // 2, 2, W // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)   # (B, C, 2, 2, H//2, W//2)
        return latents.reshape(B, C * 4, H // 2, W // 2)

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """(B, C*4, H//2, W//2) -> (B, C, H, W) — inverse of _patchify_latents.

        Matches Flux2KleinPipeline._unpatchify_latents.
        """
        B, Cp, Hh, Wh = latents.shape
        latents = latents.reshape(B, Cp // 4, 2, 2, Hh, Wh)
        latents = latents.permute(0, 1, 4, 2, 5, 3)   # (B, C, Hh, 2, Wh, 2)
        return latents.reshape(B, Cp // 4, Hh * 2, Wh * 2)

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, H*W, C) flat spatial-to-sequence packing.

        Matches Flux2KleinPipeline._pack_latents.
        """
        B, C, H, W = latents.shape
        return latents.reshape(B, C, H * W).permute(0, 2, 1)

    @staticmethod
    def _prepare_latent_ids(latents: torch.Tensor, device=None) -> torch.Tensor:
        """4D position IDs (T, H, W, L) for patchified latents (B, C, H, W).

        Matches Flux2KleinPipeline._prepare_latent_ids.
        """
        batch_size, _, height, width = latents.shape
        latent_ids = torch.cartesian_prod(
            torch.arange(1),
            torch.arange(height),
            torch.arange(width),
            torch.arange(1),
        )   # (H*W, 4)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)  # (B, H*W, 4)
        return latent_ids if device is None else latent_ids.to(device)

    def _encode_prompt(self, prompt, device):
        """Encode text prompt via Qwen3, detached from gradient graph.

        Flux2KleinPipeline.encode_prompt returns (prompt_embeds, text_ids).
        There is no pooled embedding and no prompt_2.
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)

        prompt_embeds, text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            device=device,
        )
        return prompt_embeds.detach().to(device), text_ids.detach().to(device)

    # ------------------------------------------------------------------
    # Main attack
    # ------------------------------------------------------------------

    def attack(
        self,
        prompt: Union[str, List[str]],
        image: torch.FloatTensor,
        mask: torch.FloatTensor,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 4,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Differentiable FLUX.2 Klein img2img forward pass.

        Gradient flows: input image → VAE encode → patchify → BN normalize →
        pack → noise mixing → denoising loop → unpack → BN denormalize →
        unpatchify → VAE decode → output image.
        """
        device = self.pipe.device
        dtype = torch.bfloat16
        vae = self.pipe.vae
        transformer = self.pipe.transformer
        scheduler = self.pipe.scheduler

        # ----- 1. Text encoding (detached) -----
        with torch.no_grad():
            prompt_embeds, text_ids = self._encode_prompt(prompt, device)

        # ----- 2. VAE encode image (gradient maintained via mode()) -----
        image_input = image.to(device=device, dtype=dtype)
        latents = vae.encode(image_input).latent_dist.mode()

        # ----- 3. Patchify: (B, C, H, W) -> (B, C*4, H//2, W//2) -----
        latents = self._patchify_latents(latents)

        # ----- 4. BatchNorm normalization (matches _encode_vae_image in pipeline) -----
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=dtype)
        bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        ).to(device=device, dtype=dtype)
        latents = (latents - bn_mean) / bn_std

        # ----- 5. Position IDs from patchified latent shape -----
        latent_ids = self._prepare_latent_ids(latents, device=device)

        # ----- 6. Pack: (B, C*4, H//2, W//2) -> (B, (H//2)*(W//2), C*4) -----
        packed = self._pack_latents(latents)

        # ----- 7. Timesteps with empirical mu (matches pipeline __call__) -----
        image_seq_len = packed.shape[1]
        mu = _compute_empirical_mu(image_seq_len, num_inference_steps)
        try:
            scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
        except TypeError:
            scheduler.set_timesteps(num_inference_steps, device=device)

        init_timestep = min(int(num_inference_steps * self.strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:]

        # ----- 8. Add noise at strength level (flow matching interpolation) -----
        noise = torch.randn(packed.shape, device=device, dtype=dtype)
        sigma = scheduler.sigmas[t_start].to(dtype)
        noisy_latents = (1.0 - sigma) * packed + sigma * noise

        # ----- 9. Denoising loop -----
        for t in timesteps:
            timestep = t.expand(batch_size).to(dtype)

            noise_pred = transformer(
                hidden_states=noisy_latents,
                timestep=timestep / 1000,
                guidance=None,                      # always None for Flux2Klein
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,                   # (B, text_seq, 4)
                img_ids=latent_ids,                 # (B, img_seq, 4)
                return_dict=False,
            )[0]

            noisy_latents = scheduler.step(
                noise_pred, t, noisy_latents, return_dict=False
            )[0]

        # ----- 10. Unpack: (B, seq, C*4) -> (B, C*4, H//2, W//2) -----
        # Use differentiable reshape (row-major order is preserved through the loop)
        h_half = height // (self.vae_scale_factor * 2)
        w_half = width // (self.vae_scale_factor * 2)
        latents_out = noisy_latents.permute(0, 2, 1).reshape(batch_size, -1, h_half, w_half)

        # ----- 11. Inverse BN normalization (matches post-loop in pipeline __call__) -----
        latents_out = latents_out * bn_std + bn_mean

        # ----- 12. Unpatchify: (B, C*4, H//2, W//2) -> (B, C, H, W) -----
        latents_out = self._unpatchify_latents(latents_out)

        # ----- 13. VAE decode -----
        output = vae.decode(latents_out.to(dtype), return_dict=False)[0]

        # Convert to float16 for consistent loss computation with GradScaler
        return output.half()
