"""Gradient-flow and loss-signal regression tests.

T1: Gradient-flow test — asserts that with gtf=1.0 (no truncation) loss1
    carries a nonzero, finite gradient all the way back to img_adv, and that
    the ORIGINAL bug (early-steps-with-grad, final-steps-no-grad) breaks the
    chain in the way the plan documents.

T2: Loss-signal test — asserts that the fixed noise target (H7) produces
    deterministic, nonzero loss1 gradient, while a per-batch random target
    produces mean gradient near zero (C2 bug documented).

T3: Scaler-hygiene assertion — verifies that reading .grad before unscale_
    yields inflated values, and after unscale_ yields correct values (C3 bug
    documented, requires CUDA).

S2: GroupNorm consistency — NestedUNet output must be identical for a given
    sample regardless of batch size (BatchNorm would fail this, GroupNorm passes).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Minimal stub: differentiable denoising loop with trainable transformer proxy
# ---------------------------------------------------------------------------
# Unlike the real attack's frozen transformer, the stub must have at least one
# path through which requires_grad propagates.  We achieve this with a simple
# linear layer that we *don't* freeze — but its inputs (noisy_latents derived
# from img_adv via a differentiable VAE) carry requires_grad, so the chain is
# maintained even without trainable weights as long as the input requires grad.

class _StubSchedulerStep(torch.autograd.Function):
    """Euler step: out = x - sigma * noise_pred.  Both inputs contribute."""
    @staticmethod
    def forward(ctx, x, noise_pred, sigma):
        return x - sigma * noise_pred

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, -grad_output, None


def _euler_step(noisy, noise_pred, sigma):
    """Differentiable Euler denoising step."""
    return noisy - float(sigma) * noise_pred


def _run_denoising_loop_patched(
    img_adv: torch.Tensor,
    gtf: float,
    strength: float,
    n_inference_steps: int = 4,
) -> torch.Tensor:
    """Patched loop: straight-through latent path with partial transformer grad.

    For skipped steps only the TRANSFORMER call runs under no_grad (detached
    noise_pred); the Euler integration step always executes with grad enabled,
    so the latent chain stays connected end to end. The first n_grad_steps
    (early, high-sigma) get transformer gradients.

    The stub VAE is the identity / scale-by-0.5 encode, scale-by-2 decode.
    No transformer weights (frozen) — the chain runs through the LATENT PATH:
        img_adv → encode → latents → noise_mix → noisy_latents_0 → ... → decode → output
    """
    # VAE encode (differentiable): latents = img_adv * 0.5
    latents = img_adv * 0.5

    # Timestep schedule
    sigmas = torch.linspace(0.99, 0.01, n_inference_steps + 1)
    all_timesteps = list(range(n_inference_steps - 1, -1, -1))

    init_timestep = min(int(n_inference_steps * strength), n_inference_steps)
    t_start = max(n_inference_steps - init_timestep, 0)
    timesteps = all_timesteps[t_start:]

    # Add noise
    noise = torch.randn_like(latents).detach()
    sigma0 = float(sigmas[t_start]) if t_start < len(sigmas) else 0.01
    noisy_latents = (1.0 - sigma0) * latents + sigma0 * noise

    # PATCHED: transformer grad on the FIRST n_grad_steps; scheduler step
    # always differentiable (straight-through latent path).
    n_steps = len(timesteps)
    n_grad_steps = max(1, int(n_steps * gtf))

    for step_idx, t in enumerate(timesteps):
        use_grad = step_idx < n_grad_steps
        sigma_t = float(sigmas[t]) if t < len(sigmas) else 0.01
        # Stub "transformer": just scale by a constant (no trainable weights,
        # but the input carries requires_grad so output inherits it)
        if use_grad:
            noise_pred = noisy_latents * 0.1
        else:
            with torch.no_grad():
                noise_pred = noisy_latents * 0.1
        # Integration step OUTSIDE any no_grad context — keeps the chain.
        noisy_latents = _euler_step(noisy_latents, noise_pred, sigma_t)

    # VAE decode: output = noisy_latents * 2
    output = noisy_latents * 2.0
    return output


def _run_denoising_loop_whole_step_no_grad(
    img_adv: torch.Tensor,
    gtf: float,
    strength: float,
    n_inference_steps: int = 4,
) -> torch.Tensor:
    """C1-era loop: keeps the LAST n_grad_steps, but runs each skipped step
    WHOLLY under no_grad — including the integration step. The first skipped
    step detaches noisy_latents, and since the transformer is frozen the later
    "grad" steps operate on a tensor with requires_grad=False, so the output
    silently carries no gradient for any gtf < 1.0 with multiple steps.
    """
    latents = img_adv * 0.5
    sigmas = torch.linspace(0.99, 0.01, n_inference_steps + 1)
    all_timesteps = list(range(n_inference_steps - 1, -1, -1))

    init_timestep = min(int(n_inference_steps * strength), n_inference_steps)
    t_start = max(n_inference_steps - init_timestep, 0)
    timesteps = all_timesteps[t_start:]

    noise = torch.randn_like(latents).detach()
    sigma0 = float(sigmas[t_start]) if t_start < len(sigmas) else 0.01
    noisy_latents = (1.0 - sigma0) * latents + sigma0 * noise

    n_steps = len(timesteps)
    n_grad_steps = max(1, int(n_steps * gtf))
    first_grad_step = n_steps - n_grad_steps

    for step_idx, t in enumerate(timesteps):
        use_grad = step_idx >= first_grad_step
        ctx = torch.enable_grad() if use_grad else torch.no_grad()
        sigma_t = float(sigmas[t]) if t < len(sigmas) else 0.01
        with ctx:
            noise_pred = noisy_latents * 0.1
            noisy_latents = _euler_step(noisy_latents, noise_pred, sigma_t)

    output = noisy_latents * 2.0
    return output


def _run_denoising_loop_broken(
    img_adv: torch.Tensor,
    gtf: float,
    strength: float,
    n_inference_steps: int = 4,
) -> torch.Tensor:
    """ORIGINAL (broken) loop: gradients on FIRST n_grad_steps only.

    The final steps run under no_grad, detaching noisy_latents from the graph
    before vae.decode — so the loss has no grad_fn and autograd.grad raises.
    """
    latents = img_adv * 0.5
    sigmas = torch.linspace(0.99, 0.01, n_inference_steps + 1)
    all_timesteps = list(range(n_inference_steps - 1, -1, -1))

    init_timestep = min(int(n_inference_steps * strength), n_inference_steps)
    t_start = max(n_inference_steps - init_timestep, 0)
    timesteps = all_timesteps[t_start:]

    noise = torch.randn_like(latents).detach()
    sigma0 = float(sigmas[t_start]) if t_start < len(sigmas) else 0.01
    noisy_latents = (1.0 - sigma0) * latents + sigma0 * noise

    n_steps = len(timesteps)
    n_grad_steps = max(1, int(n_steps * gtf))

    for step_idx, t in enumerate(timesteps):
        # BUG: gradients on EARLY steps; FINAL steps run under no_grad
        use_grad = step_idx < n_grad_steps
        ctx = torch.enable_grad() if use_grad else torch.no_grad()
        sigma_t = float(sigmas[t]) if t < len(sigmas) else 0.01
        with ctx:
            noise_pred = noisy_latents * 0.1
            noisy_latents = _euler_step(noisy_latents, noise_pred, sigma_t)

    output = noisy_latents * 2.0
    return output


# ---------------------------------------------------------------------------
# T1: Gradient-flow tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strength", [0.5, 0.75, 0.99])
def test_t1_gradient_flows_gtf_one(strength):
    """T1 (gtf=1.0): With full gradient tracking, loss1 must reach img_adv.

    gtf=1.0 means no truncation — all denoising steps contribute to the graph.
    This is the baseline correctness check.
    """
    torch.manual_seed(0)
    img_adv = torch.randn(1, 3, 8, 8, requires_grad=True)
    fixed_target = torch.full_like(img_adv.detach(), -1.0)

    img_out = _run_denoising_loop_patched(img_adv, gtf=1.0, strength=strength, n_inference_steps=4)
    loss1 = (img_out - fixed_target).abs().mean()

    assert loss1.requires_grad, "loss1 has no grad_fn — chain severed (C1 bug)"

    (grad,) = torch.autograd.grad(loss1, img_adv)

    assert grad is not None, f"Gradient is None for strength={strength}"
    assert torch.isfinite(grad).all(), f"Gradient contains NaN/Inf for strength={strength}"
    assert grad.abs().sum() > 0, f"Gradient is zero for strength={strength} — chain severed"


@pytest.mark.parametrize("gtf", [0.25, 0.5, 1.0])
def test_t1_single_step_all_gtf(gtf):
    """T1 (n_steps=1): With 1 denoising step, any gtf always enables that step.

    With n_inference_steps=1 and strength=1.0, n_steps=1 and n_grad_steps=1,
    so first_grad_step=0 regardless of gtf.  All three gtf values must pass.
    """
    torch.manual_seed(1)
    img_adv = torch.randn(1, 3, 8, 8, requires_grad=True)
    fixed_target = torch.full_like(img_adv.detach(), -1.0)

    img_out = _run_denoising_loop_patched(img_adv, gtf=gtf, strength=1.0, n_inference_steps=1)
    loss1 = (img_out - fixed_target).abs().mean()

    assert loss1.requires_grad, (
        f"loss1 has no grad_fn for gtf={gtf}, n_steps=1 — unexpected chain break"
    )
    (grad,) = torch.autograd.grad(loss1, img_adv)
    assert grad is not None and grad.abs().sum() > 0


def test_t1_original_bug_breaks_chain():
    """T1 (regression): The original code (early-steps-with-grad) severs the chain.

    With n_steps=2 and gtf=0.5, the original code keeps step 0 with grad
    and runs step 1 under no_grad.  The final noisy_latents is detached, so
    vae.decode → img_out has no grad_fn, and loss1.requires_grad is False.
    """
    torch.manual_seed(2)
    img_adv = torch.randn(1, 3, 8, 8, requires_grad=True)
    fixed_target = torch.full_like(img_adv.detach(), -1.0)

    # n_inference_steps=2, strength=1.0 → n_steps=2
    img_out = _run_denoising_loop_broken(img_adv, gtf=0.5, strength=1.0, n_inference_steps=2)
    loss1 = (img_out - fixed_target).abs().mean()

    # The broken loop's final latent is detached → loss has no grad_fn
    assert not loss1.requires_grad, (
        "Expected broken loop to produce loss1 with no grad_fn (chain severed). "
        "If this fails, the 'broken' function no longer models the original bug."
    )


def test_t1_patched_loop_fixes_chain_when_no_truncation():
    """T1 (fix verification): Patched loop with gtf=1.0 keeps the chain intact.

    This test is the counterpart to test_t1_original_bug_breaks_chain.
    Same setup (n_steps=2, strength=1.0) but with the patched loop at gtf=1.0
    — all steps have gradients, chain is intact, loss reaches img_adv.
    """
    torch.manual_seed(2)
    img_adv = torch.randn(1, 3, 8, 8, requires_grad=True)
    fixed_target = torch.full_like(img_adv.detach(), -1.0)

    img_out = _run_denoising_loop_patched(img_adv, gtf=1.0, strength=1.0, n_inference_steps=2)
    loss1 = (img_out - fixed_target).abs().mean()

    assert loss1.requires_grad, "Patched loop with gtf=1.0 severed the chain — regression"

    (grad,) = torch.autograd.grad(loss1, img_adv)
    assert grad is not None and grad.abs().sum() > 0, "Zero gradient with patched loop, gtf=1.0"


@pytest.mark.parametrize("gtf", [0.25, 0.5])
def test_t1_truncated_multi_step_keeps_chain(gtf):
    """T1 (straight-through): gtf < 1.0 with multiple steps must keep gradient.

    Regression for the C1-era fix, which ran skipped steps wholly under
    no_grad: with any truncation the first skipped step detached the latents
    and loss1's gradient silently became zero (loss2 still had gradient, so
    training would collapse the perturbation to zero instead of crashing).
    """
    torch.manual_seed(3)
    img_adv = torch.randn(1, 3, 8, 8, requires_grad=True)
    fixed_target = torch.full_like(img_adv.detach(), -1.0)

    img_out = _run_denoising_loop_patched(img_adv, gtf=gtf, strength=1.0, n_inference_steps=4)
    loss1 = (img_out - fixed_target).abs().mean()

    assert loss1.requires_grad, f"Chain severed at gtf={gtf} with 4 steps"
    (grad,) = torch.autograd.grad(loss1, img_adv)
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0, f"Zero gradient at gtf={gtf} — chain severed"


def test_t1_whole_step_no_grad_severs_chain():
    """T1 (documentation): running whole skipped steps under no_grad severs
    the chain for gtf < 1.0 — this is why the straight-through pattern exists.
    """
    torch.manual_seed(4)
    img_adv = torch.randn(1, 3, 8, 8, requires_grad=True)
    fixed_target = torch.full_like(img_adv.detach(), -1.0)

    img_out = _run_denoising_loop_whole_step_no_grad(
        img_adv, gtf=0.5, strength=1.0, n_inference_steps=4
    )
    loss1 = (img_out - fixed_target).abs().mean()

    assert not loss1.requires_grad, (
        "Expected whole-step no_grad loop to sever the chain. If this fails, "
        "the stub no longer models the C1-era behaviour."
    )


# ---------------------------------------------------------------------------
# T2: Loss-signal tests
# ---------------------------------------------------------------------------

def test_t2_fixed_target_gradient_is_deterministic_and_nonzero():
    """T2: Fixed ±1 target (H7 patched) must produce identical, nonzero gradients
    when img_out is held constant across calls.  This verifies that the optimizer
    sees a consistent direction rather than zero-mean noise.
    """
    torch.manual_seed(42)
    g = torch.Generator().manual_seed(1234)
    shape = (1, 3, 8, 8)

    fixed_target = (torch.randint(0, 2, shape, generator=g).float() * 2 - 1)
    # Hold img_out constant — check that gradient is stable across multiple calls
    img_out_val = torch.randn(shape)

    gradients = []
    for _ in range(5):
        img_out = img_out_val.clone().requires_grad_(True)
        loss1 = (img_out - fixed_target).abs().mean()
        (grad,) = torch.autograd.grad(loss1, img_out)
        gradients.append(grad.detach().flatten())

    for g_vec in gradients[1:]:
        assert torch.allclose(g_vec, gradients[0], atol=1e-6), (
            "Fixed target produced different gradients across calls — "
            "suggests target is being re-sampled (C2 bug)"
        )

    assert gradients[0].abs().sum() > 0, "Fixed target produced zero gradient"


def test_t2_random_target_gradient_mean_collapses():
    """T2: Per-batch random ±1 target (C2 bug) has zero-mean gradient.

    For a neutral output (zeros), the per-sign expectation E[sign(0-t)] is
    zero-mean when t is random ±1.  Averaging over 200 batches should give
    mean gradient close to zero.
    """
    torch.manual_seed(99)
    shape = (1, 3, 32, 32)

    gradients = []
    for _ in range(200):
        random_target = (torch.randint(0, 2, shape).float() * 2 - 1)
        img_out = torch.zeros(shape, requires_grad=True)
        loss1 = (img_out - random_target).abs().mean()
        (grad,) = torch.autograd.grad(loss1, img_out)
        gradients.append(grad.detach())

    mean_grad = torch.stack(gradients).mean(dim=0)
    assert mean_grad.abs().mean() < 0.05, (
        f"Expected mean gradient near zero for random target, "
        f"got {mean_grad.abs().mean():.4f} — C2 bug not reproduced"
    )


def test_t2_random_target_gradient_changes_each_call():
    """T2: Per-batch random target (C2 bug) must give inconsistent gradient direction."""
    torch.manual_seed(0)
    shape = (1, 3, 8, 8)
    img_out_val = torch.randn(shape)

    gradients = []
    for _ in range(10):
        random_target = (torch.randint(0, 2, shape).float() * 2 - 1)
        img_out = img_out_val.clone().requires_grad_(True)
        loss1 = (img_out - random_target).abs().mean()
        (grad,) = torch.autograd.grad(loss1, img_out)
        gradients.append(grad.detach().flatten())

    # At least some gradients should differ (random target → random sign)
    any_differ = any(
        not torch.allclose(gradients[0], g, atol=1e-6) for g in gradients[1:]
    )
    assert any_differ, (
        "All random-target gradients are identical — target may not be random (unexpected)"
    )


# ---------------------------------------------------------------------------
# T3: Scaler-hygiene test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GradScaler requires CUDA")
def test_t3_unscale_corrects_inflated_gradients():
    """T3: scaler.scale(loss).backward() inflates .grad by ~scale_factor.
    unscale_(optimizer) must divide them back to true magnitude.

    Verifies that reading .grad BEFORE unscale_ gives wrong (inflated) values
    and AFTER unscale_ gives the correct values — motivating the C3 fix of
    always calling unscale_ before flat_minima.apply() and record_gradient().
    """
    model = nn.Linear(4, 4, bias=False).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    init_scale = 1024.0
    scaler = torch.amp.GradScaler("cuda", init_scale=init_scale, enabled=True)

    x = torch.randn(2, 4, device="cuda")
    loss = model(x).sum()

    scaler.scale(loss).backward()

    # Before unscale_: .grad is inflated by init_scale
    grad_before = model.weight.grad.clone()

    scaler.unscale_(optimizer)

    # After unscale_: .grad should be the true gradient
    grad_after = model.weight.grad.clone()

    ratio = (grad_before.abs().mean() / (grad_after.abs().mean() + 1e-10)).item()

    # The ratio should be close to init_scale (allow 20% tolerance)
    assert abs(ratio - init_scale) / init_scale < 0.2, (
        f"Expected grad_before/grad_after ratio ≈ {init_scale}, got {ratio:.1f}. "
        "GradScaler may not be scaling as expected."
    )

    assert torch.isfinite(grad_after).all(), "Unscaled gradient has NaN/Inf"

    scaler.step(optimizer)
    scaler.update()


# ---------------------------------------------------------------------------
# S2: GroupNorm consistency (BatchNorm vs GroupNorm batch-size independence)
# ---------------------------------------------------------------------------

def test_s2_groupnorm_consistent_across_batch_sizes():
    """S2: GroupNorm output for sample[0] must equal output when processed alone.

    BatchNorm would fail this test (batch statistics change with batch size),
    GroupNorm passes because it normalizes within each sample independently.
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from diffvax.model import NestedUNet

    torch.manual_seed(7)
    net = NestedUNet(num_classes=3).eval()

    single_input = torch.randn(1, 3, 64, 64)
    other_input = torch.randn(1, 3, 64, 64)
    batch_input = torch.cat([single_input, other_input], dim=0)

    with torch.no_grad():
        out_single = net(single_input)
        out_batch = net(batch_input)

    assert torch.allclose(out_single, out_batch[:1], atol=1e-4), (
        "NestedUNet output for sample[0] differs between bs=1 and bs=2. "
        "This indicates BatchNorm is still present (S2 regression)."
    )


def test_s2_groupnorm_consistent_in_train_mode():
    """S2: GroupNorm output must be identical in train and eval mode for same input.

    BatchNorm uses running stats in eval and batch stats in train — they diverge
    especially at low batch sizes or after curriculum stage changes.
    GroupNorm has no running stats, so train/eval output is identical.
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from diffvax.model import NestedUNet

    torch.manual_seed(8)
    net = NestedUNet(num_classes=3)

    x = torch.randn(2, 3, 32, 32)

    net.eval()
    with torch.no_grad():
        out_eval = net(x)

    net.train()
    with torch.no_grad():
        out_train = net(x)

    assert torch.allclose(out_eval, out_train, atol=1e-4), (
        "NestedUNet produces different output in train vs eval mode. "
        "This indicates BatchNorm is still present (S2 regression)."
    )
