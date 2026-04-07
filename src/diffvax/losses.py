"""Additional loss terms for DiffVax training.

Implements optional auxiliary losses motivated by the literature:

  - vae_feature_loss: maximise ||VAE(x+δ) - VAE(x)||₂ in latent space.
    Improves cross-architecture transfer (H4, motivated by arXiv:2603.13028).

  - attention_entropy_loss: maximise entropy of cross-attention maps captured
    during the attack model's forward pass. Disrupts semantic grounding of the
    editing model, improving transfer to DiT architectures (H7, motivated by
    arXiv:2602.14679 — Universal Immunization via Semantic Injection).

Usage:
    from diffvax.losses import AttentionHookManager, vae_feature_loss

    # VAE feature loss
    loss_vae = vae_feature_loss(shared_vae, img_clean, img_immunized)

    # Attention entropy loss (during attack model forward pass)
    with AttentionHookManager(attack_model.model.unet) as hooks:
        _ = attack_model.attack(...)  # forward pass captures attention maps
    loss_attn = hooks.attention_entropy_loss()
"""

import torch
import torch.nn.functional as F
from contextlib import contextmanager
from typing import List, Optional


# ---------------------------------------------------------------------------
# VAE feature-space loss (H4)
# ---------------------------------------------------------------------------

def vae_feature_loss(
    vae,
    img_clean: torch.Tensor,
    img_immunized: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute negative squared L2 distance in VAE latent space.

    Minimising this loss (i.e. gradient descent on it) MAXIMISES the distance
    between clean and immunized latent representations, making perturbations
    that corrupt the VAE encoding and thus disrupt any downstream model that
    uses the same (or similar) VAE.

    Args:
        vae: frozen AutoencoderKL (half precision on CUDA).
        img_clean: clean image batch, shape (B, 3, H, W), in [-1, 1].
        img_immunized: immunized image batch, same shape.
        normalize: if True, divide by number of latent elements.

    Returns:
        Scalar loss (negative = we want to minimise, which maximises distance).
    """
    with torch.no_grad():
        orig_latents = vae.encode(img_clean.half()).latent_dist.mean.float()
    imm_latents = vae.encode(img_immunized.half()).latent_dist.mean.float()

    sq_dist = (imm_latents - orig_latents.detach()).pow(2)
    if normalize:
        return -sq_dist.mean()
    return -sq_dist.sum()


# ---------------------------------------------------------------------------
# Attention entropy loss (H7 — future experiment)
# ---------------------------------------------------------------------------

class AttentionHookManager:
    """Context manager that hooks into a UNet's cross-attention layers and
    captures attention weight tensors during a forward pass.

    Used to compute an attention entropy loss that disrupts the model's
    semantic grounding of the editing prompt (inspired by arXiv:2602.14679).

    Currently supports SD 1.5 UNet (Transformer2DModel blocks). For FLUX/SD3
    transformers, use FluxAttentionHookManager.

    Usage::

        with AttentionHookManager(pipe.unet) as hooks:
            edited = attack.attack(prompt, masked_image, mask, ...)
        loss = hooks.attention_entropy_loss()
        loss.backward()
    """

    def __init__(self, unet):
        self.unet = unet
        self._hooks: list = []
        self._attn_maps: List[torch.Tensor] = []

    def __enter__(self):
        self._attn_maps.clear()
        self._hooks.clear()
        self._register_hooks()
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _register_hooks(self):
        """Register forward hooks on all Attention layers inside the UNet."""
        for name, module in self.unet.named_modules():
            if _is_attention_module(module):
                h = module.register_forward_hook(self._attention_hook)
                self._hooks.append(h)

    def _attention_hook(self, module, inputs, output):
        """Capture attention weights from the module's last forward pass."""
        # SD 1.5 attention modules store the last attn weights in .attn_weights
        # if `return_attention_scores` was True. Otherwise, we compute them here.
        if hasattr(output, "attentions") and output.attentions is not None:
            for attn in output.attentions:
                if attn is not None:
                    self._attn_maps.append(attn.detach())

    def attention_entropy_loss(self, target: str = "maximize") -> torch.Tensor:
        """Compute attention entropy loss from captured maps.

        Maximising entropy of cross-attention maps distributes attention
        uniformly across tokens — disrupting the model's focus on key prompt
        tokens and degrading edit quality.

        Args:
            target: "maximize" to spread attention (disrupt edits) or
                    "minimize" to concentrate attention (alternative strategy).

        Returns:
            Scalar loss. Use loss.backward() after calling attack().
        """
        if not self._attn_maps:
            return torch.tensor(0.0, requires_grad=True)

        stacked = torch.cat([m.reshape(-1, m.shape[-1]) for m in self._attn_maps])
        # Softmax to get probability distribution (if not already)
        probs = F.softmax(stacked.float(), dim=-1)
        # Shannon entropy: H = -sum(p * log(p))
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()

        if target == "maximize":
            return -entropy  # gradient descent → maximise entropy
        return entropy


class FluxAttentionHookManager:
    """Attention hook manager for FLUX transformer blocks.

    FLUX uses double-stream and single-stream transformer blocks with
    different attention module layouts than SD 1.5.
    """

    def __init__(self, transformer):
        self.transformer = transformer
        self._hooks: list = []
        self._attn_maps: List[torch.Tensor] = []

    def __enter__(self):
        self._attn_maps.clear()
        self._hooks.clear()
        self._register_hooks()
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _register_hooks(self):
        for name, module in self.transformer.named_modules():
            cls_name = type(module).__name__
            if "Attention" in cls_name and hasattr(module, "to_q"):
                h = module.register_forward_hook(self._attn_hook)
                self._hooks.append(h)

    def _attn_hook(self, module, inputs, output):
        if isinstance(output, tuple) and len(output) >= 2:
            maybe_attn = output[1]
            if isinstance(maybe_attn, torch.Tensor) and maybe_attn.dim() >= 2:
                self._attn_maps.append(maybe_attn.detach())

    def attention_entropy_loss(self) -> torch.Tensor:
        if not self._attn_maps:
            return torch.tensor(0.0, requires_grad=True)
        stacked = torch.cat([m.reshape(-1, m.shape[-1]) for m in self._attn_maps])
        probs = F.softmax(stacked.float(), dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
        return -entropy  # minimising this maximises entropy


def _is_attention_module(module) -> bool:
    """Check if a module is a cross-attention layer in a diffusion UNet."""
    cls = type(module).__name__
    # HuggingFace diffusers naming conventions
    return cls in (
        "Attention",
        "CrossAttention",
        "SelfAttention",
        "BasicTransformerBlock",
    ) or ("Attn" in cls and hasattr(module, "to_q"))
