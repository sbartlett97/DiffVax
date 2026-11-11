"""Per-image PGD against an edit-mode surrogate: the upper bound for any
amortized (one-shot) generator trained on the same objective.

If no 16/255 (or 8/255) perturbation found by direct per-image optimisation can
break FLUX.2 Klein edits, a feed-forward network trained on that objective will
not either. Two objectives are supported:

  * ``encoder``: PhotoGuard-style encoder attack. Push the VAE latent of the
    protected image toward the latent of a flat mid-gray image. The editor reads
    the reference ONLY through this latent, so a corrupted latent corrupts every
    denoising step. Cheap (no denoising loop).
  * ``e2e``: end-to-end. Push the EDIT OUTPUT of the differentiable editor toward
    a flat mid-gray image (an edit that "fails" to a blank result), over random
    generation noise each step.

Optional EoT (JPEG + resize, differentiable) makes the perturbation target what
survives social-media recompression.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


def _encoder_loss_fn(atk):
    vae = atk.pipe.vae
    dtype = next(vae.parameters()).dtype
    device = next(vae.parameters()).device
    cache = {}

    def loss(x_adv: torch.Tensor, prompt) -> torch.Tensor:
        key = tuple(x_adv.shape)
        if key not in cache:
            with torch.no_grad():
                gray = torch.zeros(x_adv.shape, device=device, dtype=dtype)
                cache[key] = vae.encode(gray).latent_dist.mode().float()
        z = vae.encode(x_adv.to(device=device, dtype=dtype)).latent_dist.mode().float()
        return F.mse_loss(z, cache[key])

    return loss


def _e2e_loss_fn(atk, num_inference_steps: int):
    def loss(x_adv: torch.Tensor, prompt) -> torch.Tensor:
        out = atk.attack(prompt=prompt, image=x_adv, num_inference_steps=num_inference_steps)
        return F.mse_loss(out.float(), torch.zeros_like(out, dtype=torch.float32))

    return loss


def make_objective(atk, name: str, num_inference_steps: int = 4) -> Callable:
    if name == "encoder":
        return _encoder_loss_fn(atk)
    if name == "e2e":
        return _e2e_loss_fn(atk, num_inference_steps)
    raise ValueError(f"unknown objective {name!r} (encoder|e2e)")


def pgd(
    x: torch.Tensor,
    objective: Callable[[torch.Tensor, object], torch.Tensor],
    prompt,
    eps_pixels: float,
    steps: int = 100,
    step_frac: float = 0.1,
    eot: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    log: Optional[Callable[[int, float], None]] = None,
) -> torch.Tensor:
    """Minimise ``objective(x + delta)`` s.t. ||delta||_inf <= eps (pixel units).

    ``x`` is (B,3,H,W) in [-1,1]; returns the protected image in [-1,1]. Signed
    gradient steps of size step_frac*eps, projected onto the L-inf ball and the
    valid pixel range every step.
    """
    eps = 2.0 * float(eps_pixels) / 255.0           # tensors live in [-1, 1]
    x = x.detach().float()
    delta = torch.zeros_like(x, requires_grad=True)
    alpha = step_frac * eps
    for i in range(steps):
        x_adv = (x + delta).clamp(-1, 1)
        inp = eot(x_adv) if eot is not None else x_adv
        loss = objective(inp, prompt)
        (g,) = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta -= alpha * g.sign()
            delta.clamp_(-eps, eps)
            delta.copy_((x + delta).clamp(-1, 1) - x)
        if log is not None:
            log(i, float(loss.detach()))
    return (x + delta.detach()).clamp(-1, 1)
