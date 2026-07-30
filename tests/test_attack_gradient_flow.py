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


class _FakeAttn(nn.Module):
    def forward(self, x):
        return torch.tanh(x) * 0.5


class _FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _FakeAttn()

    def forward(self, x):
        return x + self.attn(x)


class FakeSD3Transformer(nn.Module):
    """MM-DiT stand-in: content-dependent, shape-preserving map with real
    ``transformer_blocks`` (each exposing ``.attn``) so Phase 7 attention
    hooks can attach exactly as they do on the production transformer."""

    def __init__(self, n_blocks: int = 6):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            _FakeBlock() for _ in range(n_blocks)
        )

    def forward(self, hidden_states, timestep=None, encoder_hidden_states=None,
                pooled_projections=None, return_dict=False):
        x = hidden_states
        for block in self.transformer_blocks:
            x = block(x)
        return (x * 0.1,)


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
        # Real nn.Modules (not just attributes) so a stray .to("cpu") call
        # is actually observable via .parameters().
        self.text_encoder = nn.Linear(4, 4)
        self.text_encoder_2 = nn.Linear(4, 4)
        self.text_encoder_3 = nn.Linear(4, 4)

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
        # Real nn.Module (not just an attribute) so a stray .to("cpu") call
        # is actually observable via .parameters().
        self.text_encoder = nn.Linear(4, 4)

    def encode_prompt(self, prompt, device=None):
        bs = len(prompt)
        embeds = torch.zeros(bs, 4, 8)
        text_ids = torch.zeros(bs, 4, 4)
        return embeds, text_ids


def make_sd3_attack(gtf: float, use_grad_ckpt: bool = True,
                    tgr: bool = False) -> SD3Attack:
    atk = SD3Attack.__new__(SD3Attack)
    atk.pipe = FakeSD3Pipe()
    atk.model_link = "fake"
    atk.strength = 0.75
    atk._gradient_timestep_fraction = gtf
    atk._tgr_enabled = tgr
    atk._tgr_hooks = []
    atk._use_grad_ckpt = use_grad_ckpt
    return atk


def make_flux_attack(gtf: float, use_grad_ckpt: bool = True) -> FluxAttack:
    atk = FluxAttack.__new__(FluxAttack)
    atk.pipe = FakeFluxPipe()
    atk.model_link = "fake"
    atk.strength = 0.75
    atk._gradient_timestep_fraction = gtf
    atk._tgr_enabled = False
    atk._tgr_hooks = []
    atk.vae_scale_factor = 8
    atk._use_grad_ckpt = use_grad_ckpt
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
# A8: text-encoder CPU offload must be CUDA-only (not MPS/CPU)
# ---------------------------------------------------------------------------
#
# On unified-memory backends (MPS), moving a submodule to "cpu" frees no
# memory (there's no separate VRAM pool) and has been observed in practice to
# leave HF's lazily/meta-loaded modules with unmaterialized ("placeholder")
# storage on the next call — a real crash
# ("RuntimeError: Placeholder storage has not been allocated on MPS device!").
# attack() must never call .to("cpu") on the text encoder(s) except when the
# resolved device is literally "cuda". These tests run with device.type ==
# "cpu" (no accelerator in this sandbox), so the offload branch must be
# skipped entirely — verified with a call-spy since a real .to("cpu") on an
# already-CPU module is otherwise unobservable by final device alone.

class _ToCallSpy:
    """Wraps nn.Module.to to record every device it was asked to move to,
    without changing behavior — needed because calling .to("cpu") on a
    module that's already on CPU is a silent no-op we couldn't otherwise
    detect just by checking the module's device afterward."""

    def __init__(self, module: nn.Module):
        self.calls = []
        self._orig_to = module.to

        def spy_to(*args, **kwargs):
            self.calls.append((args, kwargs))
            return self._orig_to(*args, **kwargs)

        module.to = spy_to


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-path test")
def test_a8_sd3_attack_does_not_offload_text_encoders_off_cuda():
    torch.manual_seed(12)
    img_adv = torch.randn(1, 3, 64, 64, requires_grad=True)

    atk = make_sd3_attack(1.0)
    spies = [
        _ToCallSpy(atk.pipe.text_encoder),
        _ToCallSpy(atk.pipe.text_encoder_2),
        _ToCallSpy(atk.pipe.text_encoder_3),
    ]

    atk.attack(
        prompt=["edit"], image=img_adv, height=64, width=64,
        num_inference_steps=4, batch_size=1, strength=0.9,
    )

    for spy in spies:
        assert not any("cpu" in str(a) + str(kw) for a, kw in spy.calls), (
            f"Text encoder was moved to cpu off-CUDA: {spy.calls}"
        )


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-path test")
def test_a8_flux_attack_does_not_offload_text_encoder_off_cuda():
    torch.manual_seed(13)
    img_adv = torch.randn(1, 3, 64, 64, requires_grad=True)

    atk = make_flux_attack(1.0)
    spy = _ToCallSpy(atk.pipe.text_encoder)

    atk.attack(
        prompt=["edit"], image=img_adv, height=64, width=64,
        num_inference_steps=4, batch_size=1, strength=0.9,
    )

    assert not any("cpu" in str(a) + str(kw) for a, kw in spy.calls), (
        f"Text encoder was moved to cpu off-CUDA: {spy.calls}"
    )


# ---------------------------------------------------------------------------
# A7: device resolution must ignore pipe.device (text-encoder-offload drift)
# ---------------------------------------------------------------------------
#
# diffusers' DiffusionPipeline.device property returns whichever component
# it finds first in the pipeline's constructor signature — in practice this
# is often a text encoder. Both SD3Attack and FluxAttack move their text
# encoder(s) to CPU at the end of every attack() call (to save RAM) and never
# move them back, so pipe.device silently starts reporting "cpu" from the
# second attack() call onward even though vae/transformer stay on the real
# accelerator. Regression: attack() must derive its working device from
# vae's own parameters, never from self.pipe.device. Simulated here with
# torch.device("meta") as an unmistakable wrong-device sentinel — if attack()
# ever reads self.pipe.device again, real tensors get an operand on "meta"
# and the call raises or misbehaves instead of quietly succeeding.

@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-path test")
def test_a7_sd3_attack_ignores_stale_pipe_device():
    torch.manual_seed(10)
    img_adv = torch.randn(1, 3, 64, 64, requires_grad=True)

    atk = make_sd3_attack(0.5)
    atk.pipe.device = torch.device("meta")  # simulate post-offload drift

    out = atk.attack(
        prompt=["edit"], image=img_adv, height=64, width=64,
        num_inference_steps=4, batch_size=1, strength=0.9,
    )
    assert out.device.type == "cpu"
    loss1 = out.float().abs().mean()
    (grad,) = torch.autograd.grad(loss1, img_adv)
    assert grad.abs().sum() > 0


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-path test")
def test_a7_flux_attack_ignores_stale_pipe_device():
    torch.manual_seed(11)
    img_adv = torch.randn(1, 3, 64, 64, requires_grad=True)

    atk = make_flux_attack(0.5)
    atk.pipe.device = torch.device("meta")  # simulate post-offload drift

    out = atk.attack(
        prompt=["edit"], image=img_adv, height=64, width=64,
        num_inference_steps=4, batch_size=1, strength=0.9,
    )
    assert out.device.type == "cpu"
    loss1 = out.float().abs().mean()
    (grad,) = torch.autograd.grad(loss1, img_adv)
    assert grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# A5: Phase 7 attention loss carries gradient through the checkpointed attack
# ---------------------------------------------------------------------------

@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-path test")
def test_a5_attention_loss_gradient_through_checkpointed_attack():
    """Hook-captured activations under NON-REENTRANT gradient checkpointing
    stay connected to the graph (verified property), so the Phase 7 attention
    entropy loss must produce a nonzero gradient on img_adv through the real
    SD3Attack loop with checkpointing enabled and gtf=0.5.
    """
    from diffvax.losses.attention_loss import AttentionDisruptionLoss

    torch.manual_seed(8)
    img_adv = torch.randn(1, 3, 64, 64, requires_grad=True)

    atk = make_sd3_attack(0.5)  # _use_grad_ckpt defaults True below
    attn_loss = AttentionDisruptionLoss(
        {"attention_loss": {"target_blocks": "middle", "num_hooks": 2}}
    )
    attn_loss.register_hooks(atk.pipe.transformer)
    try:
        _ = atk.attack(
            prompt=["edit"], image=img_adv, height=64, width=64,
            num_inference_steps=4, batch_size=1, strength=0.9,
        )
        loss_attn = attn_loss.compute()
    finally:
        attn_loss.remove_hooks()

    assert loss_attn.requires_grad, (
        "Attention loss lost its grad path through the checkpointed attack"
    )
    (grad,) = torch.autograd.grad(loss_attn, img_adv)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0, "Zero gradient from Phase 7 attention loss"


def test_a5_attention_loss_all_detached_warns_and_returns_zero():
    """If every captured activation is detached (e.g. reentrant checkpointing
    or a fully no_grad forward), compute() must warn once and return 0 rather
    than silently adding a constant to the loss.
    """
    from diffvax.losses.attention_loss import AttentionDisruptionLoss

    attn_loss = AttentionDisruptionLoss(
        {"attention_loss": {"target_blocks": "early", "num_hooks": 2}}
    )
    transformer = FakeSD3Transformer(n_blocks=4)
    attn_loss.register_hooks(transformer)
    try:
        with torch.no_grad():
            transformer(torch.randn(1, 16, 8, 8))
        with pytest.warns(UserWarning, match="detached"):
            val = attn_loss.compute()
    finally:
        attn_loss.remove_hooks()

    assert not val.requires_grad
    assert val.item() == 0.0

    # Second call must not warn again (one-time warning)
    attn_loss.register_hooks(transformer)
    try:
        with torch.no_grad():
            transformer(torch.randn(1, 16, 8, 8))
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            val2 = attn_loss.compute()
    finally:
        attn_loss.remove_hooks()
    assert val2.item() == 0.0


# ---------------------------------------------------------------------------
# A6: TGR (H4) — full backward PRE-hook semantics
# ---------------------------------------------------------------------------

def test_a6_tgr_pre_hook_equalizes_token_gradients():
    """The TGR pre-hook must equalize per-token gradient norms while
    preserving the overall gradient scale (mean token norm).
    """

    class SeqBlock(nn.Module):
        def forward(self, x):  # (B, seq, dim)
            return x * 2.0

    block = SeqBlock()
    handle = block.register_full_backward_pre_hook(
        SD3Attack._tgr_backward_pre_hook
    )
    try:
        x = torch.randn(1, 4, 8, requires_grad=True)
        out = block(x)
        # Token 0 dominates the loss by 100x
        loss = out[:, 0].sum() * 10.0 + out[:, 1:].sum() * 0.1
        loss.backward()
        tok_norms = x.grad.norm(dim=-1)[0]
    finally:
        handle.remove()

    assert torch.allclose(tok_norms, tok_norms[0].expand_as(tok_norms), rtol=1e-4), (
        f"TGR did not equalize token gradient norms: {tok_norms.tolist()}"
    )
    assert tok_norms[0].item() > 0


def test_a6_tgr_pre_hook_fires_through_checkpoint():
    """TGR hooks must also apply when the block runs under non-reentrant
    gradient checkpointing (the attacks' default execution mode)."""
    from torch.utils.checkpoint import checkpoint

    class SeqBlock(nn.Module):
        def forward(self, x):
            return x * 2.0

    block = SeqBlock()
    handle = block.register_full_backward_pre_hook(
        SD3Attack._tgr_backward_pre_hook
    )
    try:
        x = torch.randn(1, 4, 8, requires_grad=True)
        out = checkpoint(block, x, use_reentrant=False)
        loss = out[:, 0].sum() * 10.0 + out[:, 1:].sum() * 0.1
        loss.backward()
        tok_norms = x.grad.norm(dim=-1)[0]
    finally:
        handle.remove()

    assert torch.allclose(tok_norms, tok_norms[0].expand_as(tok_norms), rtol=1e-4), (
        f"TGR inactive under checkpointing: {tok_norms.tolist()}"
    )


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-path test")
def test_a6_tgr_registration_suppresses_kwarg_backward_hook_warning():
    """Real DiT blocks are invoked with all-keyword arguments, which makes
    PyTorch's full-backward-hook input-tracking (incorrectly) conclude no
    inputs require grad and warn on every backward. _register_tgr_hooks must
    install a filter for this specific message so enabling TGR doesn't spam
    a warning on every training batch. The hook's actual gradient-output
    correctness (unaffected by this PyTorch bookkeeping quirk) is covered by
    test_a6_tgr_pre_hook_equalizes_token_gradients and
    test_a6_tgr_enabled_attack_gradients_stay_finite_nonzero.
    """
    import warnings

    class KwargOnlyBlock(nn.Module):
        def forward(self, hidden_states, encoder_hidden_states=None):
            return hidden_states + 0.0 * encoder_hidden_states.sum()

    block = KwargOnlyBlock()
    atk = make_sd3_attack(1.0, tgr=True)
    atk._register_tgr_hooks(
        types.SimpleNamespace(transformer_blocks=[block])
    )
    try:
        hs = torch.randn(1, 4, 8, requires_grad=True)
        enc = torch.randn(1, 4, 8).detach()
        # Do NOT call simplefilter("always") here — that would reset the
        # filter list and defeat the very suppression under test. Recording
        # (record=True) alone preserves whatever filters are already active
        # (including the one _register_tgr_hooks just installed) and simply
        # diverts anything that gets through to `w`.
        with warnings.catch_warnings(record=True) as w:
            out = block(hidden_states=hs, encoder_hidden_states=enc)
            out.sum().backward()
            matched = [
                x for x in w
                if "Full backward hook is firing" in str(x.message)
            ]
        assert not matched, (
            "Expected the kwarg-only backward-hook warning to be filtered "
            "after _register_tgr_hooks; got: "
            f"{[str(x.message) for x in matched]}"
        )
    finally:
        atk._remove_tgr_hooks()


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-path test")
def test_a6_tgr_enabled_attack_gradients_stay_finite_nonzero():
    """Regression for the old register_backward_hook implementation, which
    corrupted gradients (replaced grad_input): with TGR enabled the real
    attack loop must still deliver finite, nonzero gradient to img_adv."""
    torch.manual_seed(9)
    img_adv = torch.randn(1, 3, 64, 64, requires_grad=True)

    atk = make_sd3_attack(0.5, tgr=True)
    out = atk.attack(
        prompt=["edit"], image=img_adv, height=64, width=64,
        num_inference_steps=4, batch_size=1, strength=0.9,
    )
    assert len(atk._tgr_hooks) > 0, "TGR hooks were not registered"
    loss1 = out.float().abs().mean()
    (grad,) = torch.autograd.grad(loss1, img_adv)
    assert torch.isfinite(grad).all(), "TGR corrupted gradients (NaN/Inf)"
    assert grad.abs().sum() > 0, "TGR zeroed the gradient"


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
