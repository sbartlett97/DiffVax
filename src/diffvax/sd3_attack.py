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
from typing import Optional, Union, List

from diffvax.attack_base import BaseAttack, suppress_full_backward_hook_kwarg_warning
from diffvax.utils import empty_cache, resolve_device, resolve_dtype


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
                 token_gradient_regularization: bool = False,
                 use_gradient_checkpointing: bool = True,
                 dtype: Optional[torch.dtype] = None):
        from diffusers import StableDiffusion3Img2ImgPipeline

        # Defaults to fp16 on CUDA, bf16 on MPS (fp16 has incomplete/unreliable
        # kernel coverage there), fp32 on CPU. Pass dtype explicitly to override.
        dtype = dtype or resolve_dtype(resolve_device())
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            model_link, torch_dtype=dtype
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
        # Gradient checkpointing trades VRAM for a recomputed forward.
        # Non-reentrant checkpointing (use_reentrant=False) keeps hook-captured
        # activations (Phase 7 attention loss) connected to the graph — only
        # no_grad-skipped timesteps produce detached captures. Knob exposed for
        # profiling/debugging; default True.
        self._use_grad_ckpt = bool(use_gradient_checkpointing)

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
        """Register full backward PRE-hooks on transformer blocks for TGR.

        ``register_full_backward_pre_hook`` fires before the block's backward
        and its return value replaces grad_output — the correct interception
        point for token-gradient normalization. (The previous implementation
        used ``register_backward_hook``, whose return value replaces
        grad_input, silently corrupting gradients; it was force-disabled.)
        Hooks are persistent: they fire on every backward, including the
        recomputed forward graphs produced by non-reentrant gradient
        checkpointing (verified by tests/test_attack_gradient_flow.py).

        Real MM-DiT blocks are called with all-keyword arguments
        (``block(hidden_states=..., encoder_hidden_states=..., temb=...)``),
        which triggers PyTorch's benign "no inputs require gradients"
        full-backward-hook warning on every backward — see
        suppress_full_backward_hook_kwarg_warning() for why this is safe
        to silence.
        """
        suppress_full_backward_hook_kwarg_warning()
        self._remove_tgr_hooks()
        blocks = getattr(transformer, "transformer_blocks", None) or []
        for block in blocks:
            h = block.register_full_backward_pre_hook(self._tgr_backward_pre_hook)
            self._tgr_hooks.append(h)

    @staticmethod
    def _tgr_backward_pre_hook(module, grad_output):
        """Equalize per-token gradient magnitude, preserving overall scale.

        grad_output[0] shape: (B, seq_len, dim). Each token's gradient vector
        is rescaled to the mean token norm — this removes the token-to-token
        variance that hurts adversarial transfer in high-token-count attention
        (TGR, CVPR 2023, arXiv:2303.15754) without changing the global
        gradient magnitude.
        """
        normed = []
        for g in grad_output:
            if g is None or g.ndim < 3:
                normed.append(g)
                continue
            tok_norm = g.norm(dim=-1, keepdim=True)
            mean_norm = tok_norm.mean(dim=1, keepdim=True)
            normed.append(g / tok_norm.clamp(min=1e-8) * mean_norm)
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
        empty_cache()

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
        vae = self.pipe.vae
        transformer = self.pipe.transformer
        scheduler = self.pipe.scheduler
        dtype = next(vae.parameters()).dtype
        # NOT self.pipe.device: DiffusionPipeline.device returns whichever
        # component diffusers finds first in the pipeline's signature (often
        # a text encoder). Since text encoders are deliberately moved to CPU
        # below and never moved back, pipe.device silently reports "cpu" on
        # every call after the first, while vae/transformer stay on the
        # accelerator — vae's own device is the only thing safe to trust.
        device = next(vae.parameters()).device

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
        empty_cache(device)

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

        # H2: partial-timestep gradient — straight-through latent path.
        # The latent chain latents_0 → latents_1 → … → latents_N is the ONLY
        # gradient route from the loss back to img_adv (transformer weights and
        # prompt embeddings are frozen/detached). Running any whole step under
        # no_grad detaches noisy_latents and silently zeroes loss1's gradient.
        # Instead, for skipped steps the TRANSFORMER runs under no_grad (its
        # Jacobian is not backpropagated — this is where the VRAM lives), while
        # scheduler.step always executes with grad enabled so the additive
        # integration path stays connected end to end.
        # The first n_grad_steps (early, high-sigma) get transformer gradients,
        # per "Distraction Is All You Need" (CVPR 2024): early steps set global
        # structure and carry the most protection-relevant signal.
        n_steps = len(timesteps)
        n_grad_steps = max(1, int(n_steps * self._gradient_timestep_fraction))

        # H4: TGR — persistent full-backward-pre-hooks normalize per-token
        # gradient magnitude during every backward pass. Registered lazily on
        # first use; they remain attached for the lifetime of the attack.
        if self._tgr_enabled and not self._tgr_hooks:
            self._register_tgr_hooks(transformer)

        # ------ 5. MM-DiT denoising loop ------
        from torch.utils.checkpoint import checkpoint as grad_checkpoint

        for step_idx, t in enumerate(timesteps):
            # Transformer Jacobian only for the first n_grad_steps; the
            # scheduler integration below always runs with grad enabled so the
            # latent chain from vae.encode to vae.decode is never severed.
            use_grad = step_idx < n_grad_steps

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
                if self._use_grad_ckpt:
                    noise_pred = grad_checkpoint(
                        _transformer_fwd,
                        latent_input, timestep, prompt_embeds_cfg,
                        pooled_embeds_cfg,
                        use_reentrant=False,
                    )
                else:
                    noise_pred = _transformer_fwd(
                        latent_input, timestep, prompt_embeds_cfg,
                        pooled_embeds_cfg,
                    )
            else:
                # Skipped step: transformer output is detached (no activation
                # memory, no Jacobian), but the scheduler step below still
                # propagates gradient through the latent path.
                with torch.no_grad():
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

        # Match whichever compute dtype the pipeline was loaded in (fp16 on
        # CUDA, bf16 on MPS, fp32 on CPU) rather than hardcoding fp16.
        return output.to(dtype)
