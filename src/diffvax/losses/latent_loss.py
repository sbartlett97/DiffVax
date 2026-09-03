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


def latent_disruption_loss(vae, img_orig: Tensor, img_adv: Tensor) -> Tensor:
    """Cosine similarity between VAE latents of the original and adversarial image.

    Args:
        vae:      Frozen diffusers ``AutoencoderKL``-style module exposing
                  ``encode(x).latent_dist.mode()``.
        img_orig: Clean image batch (B, 3, H, W) in [-1, 1]. No gradient is
                  propagated through this branch.
        img_adv:  Adversarial image batch, same shape. Gradient flows through
                  this branch back to the perturbation network.

    Returns:
        Scalar tensor: mean cosine similarity over the batch. Add it to the
        minimized training loss (optionally weighted) to push adversarial
        latents away from the clean latents.
    """
    dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        lat_orig = vae.encode(img_orig.to(dtype=dtype)).latent_dist.mode().detach()
    lat_adv = vae.encode(img_adv.to(dtype=dtype)).latent_dist.mode()
    return F.cosine_similarity(
        lat_orig.flatten(1), lat_adv.flatten(1), dim=1
    ).mean()
