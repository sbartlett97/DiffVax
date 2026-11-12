"""H8: VAE latent-space disruption loss.

Pushes ``vae.encode(img_adv)`` AWAY from ``vae.encode(img_orig)`` (PhotoGuard
encoder-attack objective, Salman 2023). The returned term is the cosine
similarity between the two latents, so MINIMIZING it maximizes the
latent-space distance. Bounded in [-1, 1]: 1 = identical latents (no
disruption), -1 = anti-aligned latents (maximal disruption).

Sign convention matters: an earlier implementation returned
``1 - cosine_similarity`` and added it to the minimized total loss, which
rewarded keeping the adversarial latents IDENTICAL to the originals — the
exact opposite of the intended objective. tests/test_attack_gradient_flow.py
carries the regression test.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def latent_disruption_loss(
    vae, img_orig: Tensor, img_adv: Tensor, mode: str = "cosine"
) -> Tensor:
    """Cosine similarity between VAE latents of the original and adversarial image.

    Args:
        vae:      Frozen diffusers ``AutoencoderKL``-style module exposing
                  ``encode(x).latent_dist.mode()``.
        img_orig: Clean image batch (B, 3, H, W) in [-1, 1]. No gradient is
                  propagated through this branch.
        img_adv:  Adversarial image batch, same shape. Gradient flows through
                  this branch back to the perturbation network.

    mode: "cosine" (legacy default; returns mean cosine similarity) or "l2"
        (returns -d/(1+d), d = normalised squared latent distance).

    Returns:
        Scalar tensor: mean cosine similarity over the batch (mode="cosine"). Add it to the
        minimized training loss (optionally weighted) to push adversarial
        latents away from the clean latents.
    """
    dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        lat_orig = vae.encode(img_orig.to(dtype=dtype)).latent_dist.mode().detach()
    lat_adv = vae.encode(img_adv.to(dtype=dtype)).latent_dist.mode()
    lat_orig, lat_adv = lat_orig.float(), lat_adv.float()
    if mode == "l2":
        # Normalised squared distance d = ||z_adv - z||^2 / ||z||^2 mapped to
        # d/(1+d) in [0, 1) and NEGATED so minimising the loss maximises the
        # distance. Cosine on ~300k-dim latents barely moves under a small
        # pixel budget (observed 0.97-0.99); a relative-distance term has a
        # usable gradient scale and saturates smoothly instead of unbounded.
        num = (lat_adv - lat_orig).flatten(1).pow(2).sum(dim=1)
        den = lat_orig.flatten(1).pow(2).sum(dim=1).clamp(min=1e-8)
        d = num / den
        return -(d / (1.0 + d)).mean()
    if mode != "cosine":
        raise ValueError(f"latent_loss.mode must be 'cosine' or 'l2', got {mode!r}")
    return F.cosine_similarity(
        lat_orig.flatten(1), lat_adv.flatten(1), dim=1
    ).mean()
