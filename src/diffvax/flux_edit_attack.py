"""Differentiable FLUX.2 Klein *edit-mode* surrogate.

Why this exists
---------------
``FluxAttack`` (flux_attack.py) runs img2img: it VAE-encodes the protected
image, mixes it with noise at ``strength`` and denoises. That is NOT how the
editors we want to defend against work. ``Flux2KleinPipeline`` (and FLUX.1
Kontext, Qwen-Image-Edit, and by public evidence other instruction editors)
treat the reference image as *context tokens*:

  1. the reference image is VAE-encoded -> patchified -> BN-normalised ->
     packed into tokens, given time coordinate T=10 (generation tokens T=0);
  2. the generation latents start from PURE NOISE;
  3. at EVERY denoising step the transformer sees
     ``cat([generation_tokens, reference_tokens])``; only the generation
     tokens are updated (the prediction is sliced to them). The reference
     tokens are never noised and never change.

The perturbation therefore has a direct, clean path into every block of every
step -- the same mechanism that makes the original DiffVax work against the
SD 1.5 inpainting UNet (which sees the clean masked-image latent at every
step). This class reproduces that path differentiably, mirroring
``diffusers.Flux2KleinPipeline.__call__`` (verified by a parity test against
the real pipeline class in tests/test_flux_edit_attack.py).

Gradient path
-------------
The generation latents start from noise, so the input image reaches the
output ONLY through transformer calls that run with autograd enabled. Steps
beyond ``gradient_timestep_fraction`` run the transformer under no_grad
(their output is a constant w.r.t. the image) while the scheduler update
still executes with grad so earlier steps' influence propagates forward.

Resolution
----------
The real pipeline downsizes any reference larger than 1024*1024 pixels to
that area before encoding (``_resize_to_target_area``), then floors to a
multiple of 16. A 1088px reference is therefore *shrunk* by the editor. This
class does the same (differentiable bicubic, antialiased; the pipeline uses
PIL LANCZOS) and resizes the result back to the input size, so callers can
treat it as size-preserving and the perturbation is trained to survive the
editor's own resize.
"""

import math
from typing import List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from diffvax.attack_base import BaseAttack
from diffvax.flux_attack import FluxAttack
from diffvax.utils import empty_cache, resolve_device, resolve_dtype

_REF_T_COORD = 10  # Flux2 _prepare_image_ids: scale + scale * i, i = 0 for one reference


class FluxKleinEditAttack(BaseAttack):
    """Differentiable reference-token edit pass through FLUX.2 Klein.

    Args:
        model_link: HF repo id of a Klein checkpoint (pin the 4B distilled
            ``black-forest-labs/FLUX.2-klein-4B`` unless you know you need
            another; non-distilled "base" variants use CFG).
        gradient_timestep_fraction: fraction of denoising steps whose
            transformer call keeps autograd (first steps first).
        use_gradient_checkpointing: recompute transformer activations in
            backward (non-reentrant).
        guidance_scale: only used by NON-distilled checkpoints (CFG).
        max_reference_area: references larger than this many pixels are
            shrunk to it, exactly like the pipeline (None disables).
        offload_text_encoder: move the Qwen3 encoder to CPU after encoding
            (CUDA only). Off by default: not needed on 80 GB cards.
    """

    def __init__(
        self,
        model_link: str,
        gradient_timestep_fraction: float = 1.0,
        use_gradient_checkpointing: bool = True,
        guidance_scale: float = 4.0,
        max_reference_area: Optional[int] = 1024 * 1024,
        offload_text_encoder: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        try:
            from diffusers import Flux2KleinPipeline
        except ImportError as exc:  # pragma: no cover - env specific
            raise ImportError(
                "FLUX.2 Klein needs a diffusers build with Flux2KleinPipeline "
                "(pip install -U diffusers)."
            ) from exc

        dtype = dtype or resolve_dtype(resolve_device())
        self.pipe = Flux2KleinPipeline.from_pretrained(model_link, torch_dtype=dtype)
        self.model_link = model_link
        self._init_common(
            gradient_timestep_fraction, use_gradient_checkpointing,
            guidance_scale, max_reference_area, offload_text_encoder,
        )
        self.pipe.vae.requires_grad_(False)
        self.pipe.transformer.requires_grad_(False)
        if getattr(self.pipe, "text_encoder", None) is not None:
            self.pipe.text_encoder.requires_grad_(False)

    def _init_common(self, gtf, ckpt, guidance_scale, max_area, offload):
        self._gradient_timestep_fraction = float(gtf)
        self._use_grad_ckpt = bool(ckpt)
        self._guidance_scale = float(guidance_scale)
        self._max_reference_area = max_area
        self._offload_text_encoder = bool(offload)
        self._capturing = False
        self._captured: list = []

    # ------------------------------------------------------------------
    # BaseAttack interface
    # ------------------------------------------------------------------

    def to_device(self, device: str):
        self.pipe = self.pipe.to(device)

    def to_cpu(self):
        self.pipe.to("cpu")
        empty_cache()

    @property
    def loss_uses_mask_weighting(self) -> bool:
        return False

    @property
    def is_inpainting(self) -> bool:
        return False

    @property
    def vae_channels(self) -> int:
        return 16

    @property
    def native_resolution(self) -> int:
        # Large enough that the training loop never pre-downsizes: the
        # pipeline's own area cap (max_reference_area) is applied inside
        # attack(), differentiably.
        return 1088

    def get_vae(self):
        return self.pipe.vae

    # ------------------------------------------------------------------
    # Helpers mirroring Flux2KleinPipeline
    # ------------------------------------------------------------------

    @property
    def _vae_scale_factor(self) -> int:
        return int(getattr(self.pipe, "vae_scale_factor", 8))

    @property
    def _is_distilled(self) -> bool:
        return bool(getattr(self.pipe.config, "is_distilled", False))

    @property
    def _use_cfg(self) -> bool:
        return self._guidance_scale > 1.0 and not self._is_distilled

    def _reference_size(self, h: int, w: int):
        """Replicates the pipeline's size logic (area cap, floor to x16)."""
        if self._max_reference_area is not None and h * w > self._max_reference_area:
            scale = math.sqrt(self._max_reference_area / (w * h))
            w, h = int(w * scale), int(h * scale)
        mult = self._vae_scale_factor * 2
        return (h // mult) * mult, (w // mult) * mult

    def _prepare_reference_image(self, image: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) in [-1,1] -> reference-resolution image, differentiable."""
        h, w = image.shape[-2:]
        rh, rw = self._reference_size(h, w)
        if (rh, rw) == (h, w):
            return image
        # Resize to the (pre-floor) target area, then center-crop to x16,
        # matching the pipeline's resize + resize_mode="crop".
        if self._max_reference_area is not None and h * w > self._max_reference_area:
            scale = math.sqrt(self._max_reference_area / (w * h))
            th, tw = int(h * scale), int(w * scale)
            image = F.interpolate(
                image.float(), (th, tw), mode="bicubic",
                align_corners=False, antialias=True,
            ).clamp(-1, 1).to(image.dtype)
            h, w = th, tw
        top, left = (h - rh) // 2, (w - rw) // 2
        return image[..., top:top + rh, left:left + rw]

    def _encode_reference(self, image: torch.Tensor):
        """Reference image -> (packed tokens (B,N,C*4), ids (B,N,4), grid (gh,gw)).

        Per-sample (each batch element is its own single-reference sample),
        not the pipeline's multi-reference concatenation.
        """
        vae = self.pipe.vae
        dtype = next(vae.parameters()).dtype
        device = next(vae.parameters()).device
        latents = vae.encode(image.to(device=device, dtype=dtype)).latent_dist.mode()
        latents = FluxAttack._patchify_latents(latents)
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=dtype)
        bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        ).to(device=device, dtype=dtype)
        latents = (latents - bn_mean) / bn_std
        _, _, gh, gw = latents.shape
        ids = torch.cartesian_prod(
            torch.tensor([_REF_T_COORD]), torch.arange(gh), torch.arange(gw), torch.arange(1)
        )
        ids = ids.unsqueeze(0).expand(latents.shape[0], -1, -1).to(device)
        return FluxAttack._pack_latents(latents), ids, (gh, gw), bn_mean, bn_std

    def _encode_text(self, prompt, batch_size, device):
        if isinstance(prompt, (tuple, list)):
            prompts = list(prompt)
        else:
            prompts = [str(prompt)] * batch_size
        with torch.no_grad():
            embeds, text_ids = self.pipe.encode_prompt(prompt=prompts, device=device)
            neg = neg_ids = None
            if self._use_cfg:
                neg, neg_ids = self.pipe.encode_prompt(
                    prompt=[""] * len(prompts), device=device
                )
        return embeds.detach(), text_ids.detach(), neg, neg_ids

    def _transformer_call(self, hs, ts, enc, txt_ids, img_ids):
        return self.pipe.transformer(
            hidden_states=hs,
            timestep=ts / 1000,
            guidance=None,
            encoder_hidden_states=enc,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]

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
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Differentiable edit of ``image`` (B,3,H,W) in [-1,1].

        ``mask`` and ``strength`` are accepted for interface compatibility and
        ignored: Klein edit mode always generates from pure noise conditioned
        on the clean reference. ``latents`` optionally supplies the initial
        noise as (B, C*4, h/16, w/16) (parity testing); default is randn.
        Returns (B,3,H,W) at the INPUT resolution.
        """
        from diffusers.pipelines.flux2.pipeline_flux2_klein import (
            compute_empirical_mu, retrieve_timesteps,
        )

        vae = self.pipe.vae
        transformer = self.pipe.transformer
        scheduler = self.pipe.scheduler
        dtype = next(vae.parameters()).dtype
        device = next(vae.parameters()).device  # NOT pipe.device (stale after offload)
        out_h, out_w = image.shape[-2:]
        batch_size = image.shape[0]

        prompt_embeds, text_ids, neg_embeds, neg_text_ids = self._encode_text(
            prompt, batch_size, device
        )
        if (
            self._offload_text_encoder and device.type == "cuda"
            and getattr(self.pipe, "text_encoder", None) is not None
        ):
            self.pipe.text_encoder.to("cpu")
            empty_cache(device)

        ref_img = self._prepare_reference_image(image)
        ref_tokens, ref_ids, (gh, gw), bn_mean, bn_std = self._encode_reference(ref_img)
        rh, rw = ref_img.shape[-2:]

        # Generation latents: pure noise on the same grid, T=0 ids.
        c4 = transformer.config.in_channels
        if latents is None:
            latents = torch.randn(batch_size, c4, gh, gw, device=device, dtype=dtype)
        gen = FluxAttack._pack_latents(latents.to(device=device, dtype=dtype))
        gen_ids = FluxAttack._prepare_latent_ids(latents, device=device)
        n_gen = gen.shape[1]
        img_ids = torch.cat([gen_ids, ref_ids], dim=1)

        sigmas = torch.linspace(1.0, 1 / num_inference_steps, num_inference_steps).numpy()
        if getattr(scheduler.config, "use_flow_sigmas", False):
            sigmas = None
        mu = compute_empirical_mu(image_seq_len=n_gen, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
        )
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(0)

        n_steps = len(timesteps)
        n_grad_steps = max(1, int(n_steps * self._gradient_timestep_fraction))

        from torch.utils.checkpoint import checkpoint as grad_checkpoint

        for step_idx, t in enumerate(timesteps):
            ts = t.expand(gen.shape[0]).to(gen.dtype)
            model_in = torch.cat([gen, ref_tokens], dim=1).to(transformer.dtype)

            def _pred(model_in, ts, enc, txt_ids):
                return self._transformer_call(model_in, ts, enc, txt_ids, img_ids)[:, :n_gen]

            if step_idx < n_grad_steps:
                if self._use_grad_ckpt:
                    cond = grad_checkpoint(
                        _pred, model_in, ts, prompt_embeds, text_ids, use_reentrant=False
                    )
                    if self._use_cfg:
                        uncond = grad_checkpoint(
                            _pred, model_in, ts, neg_embeds, neg_text_ids, use_reentrant=False
                        )
                else:
                    cond = _pred(model_in, ts, prompt_embeds, text_ids)
                    if self._use_cfg:
                        uncond = _pred(model_in, ts, neg_embeds, neg_text_ids)
            else:
                with torch.no_grad():
                    cond = _pred(model_in, ts, prompt_embeds, text_ids)
                    if self._use_cfg:
                        uncond = _pred(model_in, ts, neg_embeds, neg_text_ids)

            noise_pred = (
                uncond + self._guidance_scale * (cond - uncond) if self._use_cfg else cond
            )
            gen_dtype = gen.dtype
            gen = scheduler.step(noise_pred, t, gen, return_dict=False)[0]
            if gen.dtype != gen_dtype:
                gen = gen.to(gen_dtype)

        out = gen.permute(0, 2, 1).reshape(batch_size, c4, gh, gw)
        out = out * bn_std + bn_mean
        out = FluxAttack._unpatchify_latents(out)
        img = vae.decode(out.to(dtype), return_dict=False)[0].to(dtype)
        if img.shape[-2:] != (out_h, out_w):
            img = F.interpolate(
                img.float(), (out_h, out_w), mode="bicubic", align_corners=False
            ).to(dtype)
        return img

    # ------------------------------------------------------------------
    # Hidden-state disruption (replaces the saturating attention-entropy loss)
    # ------------------------------------------------------------------

    def _block_list(self, group: str, target: str, num_hooks: int):
        """Pick blocks: ('single'|'double', index) tuples."""
        out = []
        for g, blocks in (
            ("double", getattr(self.pipe.transformer, "transformer_blocks", [])),
            ("single", getattr(self.pipe.transformer, "single_transformer_blocks", [])),
        ):
            if group not in (g, "both"):
                continue
            n = len(blocks)
            if n == 0:
                continue
            if target == "middle":
                lo, hi = n // 3, max(n // 3 + 1, 2 * n // 3)
                idx = list(range(lo, hi))
            elif target == "all":
                idx = list(range(n))
            else:
                idx = list(target)
            step = max(1, len(idx) // max(1, num_hooks))
            out += [(g, i) for i in idx[::step][:num_hooks]]
        return out

    def hidden_state_loss(
        self,
        img_orig: torch.Tensor,
        img_adv: torch.Tensor,
        prompt,
        sigma: float = 1.0,
        group: str = "single",
        target_blocks="middle",
        num_hooks: int = 4,
    ) -> torch.Tensor:
        """Mean cosine similarity between clean and protected REFERENCE-token
        hidden states at the chosen transformer blocks (one forward each).

        Bounded in [-1, 1]; MINIMISING it pushes the protected reference's
        internal representation away from the clean one -- the quantity a
        DeContext-style attack targets -- without relying on attention maps.
        Generation tokens are identical noise in both passes, so any
        difference is caused by the reference alone.
        """
        transformer = self.pipe.transformer
        device = next(self.pipe.vae.parameters()).device
        dtype = next(self.pipe.vae.parameters()).dtype
        batch = img_adv.shape[0]
        prompt_embeds, text_ids, _, _ = self._encode_text(prompt, batch, device)
        blocks = self._block_list(group, target_blocks, num_hooks)
        if not blocks:
            return torch.zeros((), device=device)

        def run(image, grad: bool):
            ref_img = self._prepare_reference_image(image)
            ref_tokens, ref_ids, (gh, gw), _, _ = self._encode_reference(ref_img)
            g = torch.Generator(device="cpu").manual_seed(0)
            noise = torch.randn(
                batch, transformer.config.in_channels, gh, gw, generator=g
            ).to(device=device, dtype=dtype)
            gen = FluxAttack._pack_latents(noise)
            ids = torch.cat([FluxAttack._prepare_latent_ids(noise, device=device), ref_ids], 1)
            n_gen = gen.shape[1]
            n_txt = prompt_embeds.shape[1]
            model_in = torch.cat([gen, ref_tokens], dim=1).to(transformer.dtype)
            ts = torch.full((batch,), float(sigma) * 1000.0, device=device, dtype=gen.dtype)

            caps, handles = {}, []
            for grp, i in blocks:
                blk = (transformer.transformer_blocks if grp == "double"
                       else transformer.single_transformer_blocks)[i]

                def hook(m, a, out, key=(grp, i)):
                    if not self._capturing:
                        return
                    if key[0] == "double":
                        h = out[1][:, n_gen:]                    # image stream: [gen, ref]
                    else:
                        h = out[:, n_txt + n_gen:]               # [text, gen, ref]
                    caps[key] = h
                handles.append(blk.register_forward_hook(hook))

            def fwd(model_in, ts):
                return self._transformer_call(model_in, ts, prompt_embeds, text_ids, ids)

            self._capturing = True
            try:
                if grad and self._use_grad_ckpt:
                    from torch.utils.checkpoint import checkpoint as gc
                    gc(fwd, model_in, ts, use_reentrant=False)
                elif grad:
                    fwd(model_in, ts)
                else:
                    with torch.no_grad():
                        fwd(model_in, ts)
            finally:
                self._capturing = False   # recompute in backward must not re-capture
                for h in handles:
                    h.remove()
            return caps

        clean = run(img_orig.detach(), grad=False)
        adv = run(img_adv, grad=True)
        sims = [
            F.cosine_similarity(adv[k].float(), clean[k].detach().float(), dim=-1).mean()
            for k in adv
        ]
        return torch.stack(sims).mean()
