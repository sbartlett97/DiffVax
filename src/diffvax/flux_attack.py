"""Differentiable FLUX.2 Klein attack for adversarial training.

Requires diffusers with FLUX.2 Klein support. Install from source if needed:
    pip install git+https://github.com/huggingface/diffusers.git
"""

import torch
from typing import Union, List

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

    def __init__(self, model_link: str, strength: float = 0.75,
                 gradient_timestep_fraction: float = 1.0,
                 token_gradient_regularization: bool = False):
        # Lazy import — only fail when someone actually uses FLUX
        try:
            from diffusers import Flux2KleinPipeline as PipeClass
        except ImportError:
            try:
                from diffusers import Flux2KleinPipeline as PipeClass
            except ImportError:
                raise ImportError(
                    "FLUX support requires diffusers with FLUX pipeline support. "
                    "Install from source: pip install git+https://github.com/huggingface/diffusers.git"
                )

        self.pipe = PipeClass.from_pretrained(
            model_link, torch_dtype=torch.float16
        )

        self.model_link = model_link
        self.strength = strength
        # H2: partial-timestep gradient — early timesteps carry the most gradient
        # signal for global structure protection. 0.5 = backprop only first 50%.
        self._gradient_timestep_fraction = float(gradient_timestep_fraction)
        # H4: TGR token gradient regularization (CVPR 2023, arXiv:2303.15754)
        self._tgr_enabled = bool(token_gradient_regularization)
        self._tgr_hooks: list = []

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

    @property
    def is_inpainting(self) -> bool:
        # FLUX is a full-image img2img model; no mask is used.
        return False

    @property
    def vae_channels(self) -> int:
        return 16

    def get_vae(self):
        """Return the 16-channel VAE for latent-space disruption loss."""
        return self.pipe.vae

    @property
    def native_resolution(self) -> int:
        return 1024

    # ------------------------------------------------------------------
    # H4: Token Gradient Regularization helpers
    # ------------------------------------------------------------------

    def _register_tgr_hooks(self, transformer) -> None:
        """Register backward hooks on single_transformer_blocks for TGR."""
        self._remove_tgr_hooks()
        # FLUX.2 Klein uses single_transformer_blocks (MMDiT single-stream)
        blocks = (
            getattr(transformer, "single_transformer_blocks", None)
            or getattr(transformer, "transformer_blocks", None)
            or []
        )
        for block in blocks:
            h = block.register_backward_hook(self._tgr_backward_hook)
            self._tgr_hooks.append(h)

    @staticmethod
    def _tgr_backward_hook(module, grad_input, grad_output):
        """Normalize per-token gradient magnitude (TGR, CVPR 2023)."""
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
        mask: torch.FloatTensor = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 4,
        batch_size: int = 1,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """Differentiable FLUX.2 Klein img2img forward pass.

        Gradient flows: input image → VAE encode → patchify → BN normalize →
        pack → noise mixing → denoising loop → unpack → BN denormalize →
        unpatchify → VAE decode → output image.
        """
        device = self.pipe.device
        if device.type == "cpu" and torch.cuda.is_available():
            raise RuntimeError(
                "FluxAttack pipeline is on CPU. Call to_device('cuda') before calling attack()."
            )
        vae = self.pipe.vae
        dtype = next(vae.parameters()).dtype
        transformer = self.pipe.transformer
        scheduler = self.pipe.scheduler

        # ----- 1. Text encoding (detached) -----
        with torch.no_grad():
            prompt_embeds, text_ids = self._encode_prompt(prompt, device)

        # Offload text encoder to CPU — gradient flows only through VAE + transformer,
        # so the Qwen3 encoder doesn't need to be in VRAM during backprop.
        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
            self.pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()

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

        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:]

        # ----- 8. Add noise at strength level (flow matching interpolation) -----
        noise = torch.randn(packed.shape, device=device, dtype=dtype)
        sigma = scheduler.sigmas[t_start].to(dtype)
        noisy_latents = (1.0 - sigma) * packed + sigma * noise

        # H2: partial-timestep gradient — straight-through latent path.
        # The latent chain is the ONLY gradient route back to img_adv (the
        # transformer is frozen, prompts detached). Skipped steps therefore run
        # only the TRANSFORMER under no_grad; scheduler.step always executes
        # with grad enabled so the additive integration path stays connected.
        # The first n_grad_steps (early, high-sigma) get transformer gradients.
        n_steps = len(timesteps)
        n_grad_steps = max(1, int(n_steps * self._gradient_timestep_fraction))

        # H4: TGR hooks disabled — register_backward_hook return value replaces
        # grad_input, not grad_output, corrupting gradients silently.
        if self._tgr_enabled:
            import warnings
            warnings.warn(
                "H4 TGR hooks are disabled due to incorrect grad semantics with "
                "register_backward_hook. Set token_gradient_regularization=False.",
                stacklevel=2,
            )

        # ----- 9. Denoising loop -----
        from torch.utils.checkpoint import checkpoint as grad_checkpoint

        for step_idx, t in enumerate(timesteps):
            # Transformer Jacobian only for the first n_grad_steps; the
            # scheduler integration below always runs with grad enabled so the
            # latent chain from vae.encode to vae.decode is never severed.
            use_grad = step_idx < n_grad_steps
            timestep = t.expand(batch_size).to(dtype)

            if use_grad:
                # Gradient checkpointing: recompute transformer activations
                # during backward instead of retaining them, saving several GB
                # for the single_transformer_blocks stack in FLUX.
                def _transformer_fwd(hs, ts, enc_hs, t_ids, i_ids):
                    return transformer(
                        hidden_states=hs,
                        timestep=ts,
                        guidance=None,
                        encoder_hidden_states=enc_hs,
                        txt_ids=t_ids,
                        img_ids=i_ids,
                        return_dict=False,
                    )[0]
                noise_pred = grad_checkpoint(
                    _transformer_fwd,
                    noisy_latents, timestep / 1000, prompt_embeds,
                    text_ids, latent_ids,
                    use_reentrant=False,
                )
            else:
                # Skipped step: transformer output is detached (no activation
                # memory, no Jacobian), but the scheduler step below still
                # propagates gradient through the latent path.
                with torch.no_grad():
                    noise_pred = transformer(
                        hidden_states=noisy_latents,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_ids,
                        return_dict=False,
                    )[0]

            noisy_latents = scheduler.step(
                noise_pred, t, noisy_latents, return_dict=False
            )[0]

        # TGR hooks were not registered (disabled); nothing to remove.

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
