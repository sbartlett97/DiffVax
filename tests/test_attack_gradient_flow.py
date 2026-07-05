"""End-to-end gradient tests through the REAL attack classes.

Unlike tests/test_gradient_flow.py (which exercises stub reimplementations of
the denoising loop), these tests drive the actual ``SD3Attack.attack`` and
``FluxAttack.attack`` code paths with lightweight fake diffusers pipelines, so
regressions in the production loop structure are caught directly.

Covered:
  A1: SD3Attack — gradient reaches img_adv for gradient_timestep_fraction
      0.25 / 0.5 / 1.0 with a multi-step schedule (regression for the
      whole-step-no_grad truncation bug that silently zeroed loss1's grad).
  A2: FluxAttack — same property through patchify/BN-normalize/pack path.
  A3: latent_disruption_loss — sign convention: minimizing the term must push
      adversarial latents AWAY from clean latents (regression for the
      inverted 1 - cos_sim implementation).
  A4: Mini learning test — a tiny NestedUNet trained through the real
      SD3Attack loop at gtf=0.5 must reduce loss1, demonstrating the training
      method produces a usable learning signal end to end.
"""

import os
import sys
import types

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffvax.sd3_attack import SD3Attack           # noqa: E402
from diffvax.flux_attack import FluxAttack         # noqa: E402
from diffvax.losses.latent_loss import latent_disruption_loss  # noqa: E402


# ---------------------------------------------------------------------------
# Fake pipeline components
# ---------------------------------------------------------------------------

class _LatentDist:
    def __init__(self, latents):
        self._latents = latents

    def mode(self):
        return self._latents

    def sample(self):
        return self._latents


class _EncodeResult:
    def __init__(self, latents):
        self.latent_dist = _LatentDist(latents)


class FakeVAE(nn.Module):
    """16-channel VAE stand-in: conv encode (stride 8), conv-transpose decode.

    Bias-free so encode is an odd function (encode(-x) == -encode(x)), which
    the latent-loss sign test relies on.
    """

    def __init__(self):
        super().__init__()
        self.enc = nn.Conv2d(3, 16, kernel_size=8, stride=8, bias=False)
        self.dec = nn.ConvTranspose2d(16, 3, kernel_size=8, stride=8, bias=False)
        # FLUX-specific attributes (harmless for SD3 use)
        self.bn = nn.BatchNorm2d(64)
        self.config = types.SimpleNamespace(
            scaling_factor=1.5,
            shift_factor=0.06,
            batch_norm_eps=1e-5,
            block_out_channels=[32, 64, 128, 256],  # len 4 → scale factor 8
        )
        for p in self.parameters():
            p.requires_grad_(False)

    def encode(self, x):
        return _EncodeResult(self.enc(x))

    def decode(self, z, return_dict=True):
        out = self.dec(z)
        if return_dict:
            return types.SimpleNamespace(sample=out)
        return (out,)


class FakeSD3Transformer(nn.Module):
    """MM-DiT stand-in: content-dependent, shape-preserving map."""

    def forward(self, hidden_states, timestep=None, encoder_hidden_states=None,
                pooled_projections=None, return_dict=False):
        return (torch.tanh(hidden_states) * 0.1,)


class FakeFluxTransformer(nn.Module):
    """FLUX DiT stand-in: packed-sequence (B, seq, C) shape-preserving map."""

    def forward(self, hidden_states, timestep=None, guidance=None,
                encoder_hidden_states=None, txt_ids=None, img_ids=None,
                return_dict=False):
        return (torch.tanh(hidden_states) * 0.1,)


class FakeFlowScheduler:
    """FlowMatch-Euler-style scheduler: prev = x + (sigma_next - sigma) * v."""

    def set_timesteps(self, num_inference_steps, device=None, mu=None):
        n = num_inference_steps
        self.timesteps = torch.linspace(1000.0, 1000.0 / n, n)
        self.sigmas = torch.linspace(1.0, 0.0, n + 1)

    def step(self, model_output, t, sample, return_dict=True):
        idx = int((self.timesteps == t).nonzero()[0].item())
        sigma = self.sigmas[idx]
        sigma_next = self.sigmas[idx + 1]
        prev = sample + (sigma_next - sigma) * model_output
        if return_dict:
            return types.SimpleNamespace(prev_sample=prev)
        return (prev,)


class FakeSD3Pipe:
    def __init__(self):
        self.device = torch.device("cpu")
        self.vae = FakeVAE()
        self.transformer = FakeSD3Transformer()
        self.scheduler = FakeFlowScheduler()

    def encode_prompt(self, prompt, prompt_2=None, prompt_3=None, device=None,
                      num_images_per_prompt=1, do_classifier_free_guidance=True,
                      negative_prompt=None):
        bs = len(prompt)
        embeds = torch.zeros(bs, 4, 8)
        pooled = torch.zeros(bs, 8)
        return embeds, embeds.clone(), pooled, pooled.clone()


class FakeFluxPipe:
    def __init__(self):
        self.device = torch.device("cpu")
        self.vae = FakeVAE()
        self.transformer = FakeFluxTransformer()
        self.scheduler = FakeFlowScheduler()

    def encode_prompt(self, prompt, device=None):
        bs = len(prompt)
        embeds = torch.zeros(bs, 4, 8)
        text_ids = torch.zeros(bs, 4, 4)
        return embeds, text_ids


def make_sd3_attack(gtf: float) -> SD3Attack:
    atk = SD3Attack.__new__(SD3Attack)
    atk.pipe = FakeSD3Pipe()
    atk.model_link = "fake"
    atk.strength = 0.75
    atk._gradient_timestep_fraction = gtf
    atk._tgr_enabled = False
    atk._tgr_hooks = []
    return atk


def make_flux_attack(gtf: float) -> FluxAttack:
    atk = FluxAttack.__new__(FluxAttack)
    atk.pipe = FakeFluxPipe()
    atk.model_link = "fake"
    atk.strength = 0.75
    atk._gradient_timestep_fraction = gtf
    atk._tgr_enabled = False
    atk._tgr_hooks = []
    atk.vae_scale_factor = 8
    return atk


# ---------------------------------------------------------------------------
# A1 / A2: gradient reaches img_adv through the real attack loops
# ---------------------------------------------------------------------------

@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-path test")
@pytest.mark.parametrize("gtf", [0.25, 0.5, 1.0])
def test_a1_sd3_attack_gradient_reaches_input(gtf):
    torch.manual_seed(0)
    img_adv = torch.randn(1, 3, 64, 64, requires_grad=True)

    atk = make_sd3_attack(gtf)
    out = atk.attack(
        prompt=["edit"], image=img_adv, height=64, width=64,
        num_inference_steps=4, batch_size=1, strength=0.9,
    )
    loss1 = out.float().abs().mean()

    assert loss1.requires_grad, (
        f"SD3Attack severed the gradient chain at gtf={gtf} — loss1 has no "
        f"grad_fn. Whole-step no_grad truncation regression."
    )
    (grad,) = torch.autograd.grad(loss1, img_adv)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0, f"Zero gradient through SD3Attack at gtf={gtf}"


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-path test")
@pytest.mark.parametrize("gtf", [0.25, 0.5, 1.0])
def test_a2_flux_attack_gradient_reaches_input(gtf):
    torch.manual_seed(1)
    img_adv = torch.randn(1, 3, 64, 64, requires_grad=True)

    atk = make_flux_attack(gtf)
    out = atk.attack(
        prompt=["edit"], image=img_adv, height=64, width=64,
        num_inference_steps=4, batch_size=1, strength=0.9,
    )
    loss1 = out.float().abs().mean()

    assert loss1.requires_grad, (
        f"FluxAttack severed the gradient chain at gtf={gtf}"
    )
    (grad,) = torch.autograd.grad(loss1, img_adv)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0, f"Zero gradient through FluxAttack at gtf={gtf}"


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-path test")
def test_a1_sd3_attack_partial_strength():
    """Partial-strength img2img (t_start > 0) must also keep the chain."""
    torch.manual_seed(2)
    img_adv = torch.randn(1, 3, 64, 64, requires_grad=True)

    atk = make_sd3_attack(0.5)
    out = atk.attack(
        prompt=["edit"], image=img_adv, height=64, width=64,
        num_inference_steps=4, batch_size=1, strength=0.75,
    )
    loss1 = out.float().abs().mean()
    (grad,) = torch.autograd.grad(loss1, img_adv)
    assert grad.abs().sum() > 0


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-path test")
def test_a1_full_strength_carries_no_image_signal():
    """Documentation: at strength=1.0 with a flow-matching schedule, t_start=0
    and sigma_0=1.0, so the init mix (1-sigma)*latents + sigma*noise contains
    ZERO image contribution — the generation is unconditional and the image
    gradient is mathematically zero. Protection training must sample
    strength < 1.0 (the training loop's strength_range upper bound of 1.0 is
    only safe because int(n*strength) < n for strength just below 1.0).
    """
    torch.manual_seed(7)
    img_adv = torch.randn(1, 3, 64, 64, requires_grad=True)

    atk = make_sd3_attack(1.0)
    out = atk.attack(
        prompt=["edit"], image=img_adv, height=64, width=64,
        num_inference_steps=4, batch_size=1, strength=1.0,
    )
    loss1 = out.float().abs().mean()
    (grad,) = torch.autograd.grad(loss1, img_adv)
    assert grad.abs().sum() == 0, (
        "Expected exactly zero image gradient at strength=1.0 (sigma_0=1.0). "
        "If this fails the schedule no longer starts at pure noise."
    )


# ---------------------------------------------------------------------------
# A3: latent disruption loss sign convention
# ---------------------------------------------------------------------------

def test_a3_latent_loss_identical_images_max():
    """Identical images → cosine similarity 1.0 (worst case for disruption)."""
    torch.manual_seed(3)
    vae = FakeVAE()
    img = torch.randn(2, 3, 64, 64)
    val = latent_disruption_loss(vae, img, img.clone())
    assert val.item() == pytest.approx(1.0, abs=1e-4)


def test_a3_latent_loss_decreases_with_disruption():
    """The term must DECREASE as adversarial latents move away from clean
    latents — i.e. minimizing it maximizes disruption. The inverted
    (1 - cos_sim) implementation fails this ordering.
    """
    torch.manual_seed(4)
    vae = FakeVAE()  # bias-free → encode(-x) = -encode(x)
    img = torch.randn(1, 3, 64, 64)

    val_same = latent_disruption_loss(vae, img, img.clone())
    val_opposite = latent_disruption_loss(vae, img, -img)

    assert val_opposite.item() < val_same.item(), (
        "latent_disruption_loss must be lower for MORE disrupted latents "
        f"(got same={val_same.item():.4f}, opposite={val_opposite.item():.4f}) "
        "— sign inversion regression (H8/C7)."
    )
    assert val_opposite.item() == pytest.approx(-1.0, abs=1e-4)


def test_a3_latent_loss_gradient_flows_to_adv_only():
    torch.manual_seed(5)
    vae = FakeVAE()
    img_orig = torch.randn(1, 3, 64, 64, requires_grad=True)
    img_adv = (img_orig.detach() + 0.01 * torch.randn(1, 3, 64, 64)).requires_grad_(True)

    val = latent_disruption_loss(vae, img_orig, img_adv)
    val.backward()

    assert img_adv.grad is not None and img_adv.grad.abs().sum() > 0
    assert img_orig.grad is None, "No gradient may flow through the clean branch"


# ---------------------------------------------------------------------------
# A4: mini learning test — the training signal actually reduces loss1
# ---------------------------------------------------------------------------

@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-path test")
def test_a4_tiny_nested_unet_learns_through_sd3_attack():
    """Train a tiny NestedUNet through the REAL SD3Attack loop (gtf=0.5) and
    assert loss1 (push attack output toward a fixed target) decreases.

    This is the end-to-end sanity check that the DiffVax training method —
    perturbation network → clamp → attack surrogate → pixel loss → backprop —
    provides a usable learning signal with partial-timestep gradient enabled.
    """
    from diffvax.model import NestedUNet

    torch.manual_seed(6)
    net = NestedUNet(num_classes=3, nb_filter=[4, 8, 16, 32, 64])
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)

    img = torch.rand(1, 3, 64, 64) * 2 - 1  # fixed "training image"
    atk = make_sd3_attack(0.5)
    target = torch.zeros(1, 3, 64, 64)

    losses = []
    for step in range(30):
        torch.manual_seed(123)  # freeze attack noise so the signal is clean
        unet_out = net(img)
        img_adv = torch.clamp(img + unet_out, -1.0, 1.0)
        out = atk.attack(
            prompt=["edit"], image=img_adv, height=64, width=64,
            num_inference_steps=4, batch_size=1, strength=0.9,
        )
        loss1 = (out.float() - target).abs().mean()
        opt.zero_grad()
        loss1.backward()
        opt.step()
        losses.append(loss1.item())

    first = sum(losses[:5]) / 5
    last = sum(losses[-5:]) / 5
    assert last < first, (
        f"loss1 did not decrease training through the real SD3Attack loop "
        f"(first5={first:.5f}, last5={last:.5f}) — no usable learning signal."
    )
