"""FluxKleinEditAttack: parity with the REAL Flux2KleinPipeline + gradient tests.

The parity test builds the real ``diffusers.Flux2KleinPipeline`` from tiny
random-init components on CPU (no text encoder: prompt embeddings are injected)
and asserts our differentiable forward reproduces its output given identical
initial noise. That is the check that "mirrors the real edit path" is true.
"""
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffusers import (  # noqa: E402
    AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from diffvax.flux_edit_attack import FluxKleinEditAttack  # noqa: E402
from diffvax.model import NestedUNet                       # noqa: E402

TXT_LEN, JOINT = 5, 12


def _tiny_pipe(distilled: bool, seed: int = 0):
    torch.manual_seed(seed)
    vae = AutoencoderKLFlux2(
        block_out_channels=(8, 8, 8, 8), decoder_block_out_channels=(8, 8, 8, 8),
        layers_per_block=1, latent_channels=4, norm_num_groups=2, sample_size=64,
    )
    # non-trivial BN stats so a missing/incorrect (de)normalisation shows up
    vae.bn.running_mean.copy_(torch.randn_like(vae.bn.running_mean) * 0.3)
    vae.bn.running_var.copy_(torch.rand_like(vae.bn.running_var) + 0.5)
    tr = Flux2Transformer2DModel(
        patch_size=1, in_channels=16, num_layers=1, num_single_layers=4,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=JOINT,
        timestep_guidance_channels=16, mlp_ratio=2.0, axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=False,
    )
    pipe = Flux2KleinPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae, text_encoder=None,
        tokenizer=None, transformer=tr, is_distilled=distilled,
    )
    pipe.vae.eval(); pipe.transformer.eval()
    pipe.vae.requires_grad_(False); pipe.transformer.requires_grad_(False)
    cond = torch.randn(1, TXT_LEN, JOINT, generator=torch.Generator().manual_seed(1))
    unc = torch.randn(1, TXT_LEN, JOINT, generator=torch.Generator().manual_seed(2))
    orig = pipe.encode_prompt

    def fake_encode(prompt=None, device=None, **kw):
        n = len(prompt) if isinstance(prompt, list) else 1
        # The real pipeline passes the negative prompt as a bare "" string.
        is_neg = prompt == "" or (isinstance(prompt, list) and prompt[0] == "")
        base = unc if is_neg else cond
        return orig(prompt=None, prompt_embeds=base.repeat(n, 1, 1), device=device)

    pipe.encode_prompt = fake_encode
    return pipe, cond, unc


def _attack(pipe, gtf=1.0, ckpt=False, gs=4.0, max_area=None):
    atk = FluxKleinEditAttack.__new__(FluxKleinEditAttack)
    atk.pipe = pipe
    atk.model_link = "tiny"
    atk._init_common(gtf, ckpt, gs, max_area, False)
    return atk


def _pil_and_tensor(size=64, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1)[None].float() / 127.5 - 1.0
    return Image.fromarray(arr), t


@pytest.mark.parametrize("distilled", [True, False])
def test_parity_with_real_klein_pipeline(distilled):
    pipe, cond, unc = _tiny_pipe(distilled)
    pil, img = _pil_and_tensor(64)
    steps = 4
    noise = torch.randn(1, 16, 4, 4, generator=torch.Generator().manual_seed(7))

    with torch.no_grad():
        ref = pipe(
            image=pil, prompt_embeds=cond, num_inference_steps=steps,
            guidance_scale=1.0 if distilled else 4.0, latents=noise.clone(),
            output_type="pt", max_sequence_length=TXT_LEN,
        ).images
    atk = _attack(pipe, gs=1.0 if distilled else 4.0)
    with torch.no_grad():
        out = atk.attack(
            prompt=["x"], image=img, height=64, width=64,
            num_inference_steps=steps, latents=noise.clone(),
        )
    mine = (out / 2 + 0.5).clamp(0, 1)
    assert mine.shape == ref.shape
    assert torch.allclose(mine, ref, atol=1e-4), (mine - ref).abs().max()
    # sanity: the output genuinely depends on the reference image
    _, other = _pil_and_tensor(64, seed=9)
    with torch.no_grad():
        out2 = atk.attack(prompt=["x"], image=other, height=64, width=64,
                          num_inference_steps=steps, latents=noise.clone())
    assert (out - out2).abs().max() > 1e-4


@pytest.mark.parametrize("gtf", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("ckpt", [False, True])
def test_gradient_reaches_input(gtf, ckpt):
    pipe, _, _ = _tiny_pipe(True)
    atk = _attack(pipe, gtf=gtf, ckpt=ckpt, gs=1.0)
    _, img = _pil_and_tensor(64)
    img.requires_grad_(True)
    out = atk.attack(prompt=["x"], image=img, height=64, width=64, num_inference_steps=4)
    assert out.requires_grad
    (g,) = torch.autograd.grad(out.abs().mean(), img)
    assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_reference_tokens_identical_at_every_step():
    pipe, _, _ = _tiny_pipe(True)
    atk = _attack(pipe, gs=1.0)
    seen = []
    pipe.transformer.register_forward_pre_hook(
        lambda m, a, kw: seen.append(kw["hidden_states"].detach().clone()), with_kwargs=True
    )
    _, img = _pil_and_tensor(64)
    with torch.no_grad():
        atk.attack(prompt=["x"], image=img, height=64, width=64, num_inference_steps=4)
    n_gen = 16                     # (64/16)^2 tokens
    assert len(seen) == 4
    refs = [s[:, n_gen:] for s in seen]
    gens = [s[:, :n_gen] for s in seen]
    for r in refs[1:]:
        assert torch.equal(r, refs[0]), "reference tokens must never be noised/updated"
    assert not torch.equal(gens[0], gens[1]), "generation tokens must evolve"


def test_area_cap_shrinks_reference_and_restores_size():
    pipe, _, _ = _tiny_pipe(True)
    atk = _attack(pipe, gs=1.0, max_area=48 * 48)
    assert atk._reference_size(64, 64) == (48, 48)
    _, img = _pil_and_tensor(64)
    img.requires_grad_(True)
    out = atk.attack(prompt=["x"], image=img, height=64, width=64, num_inference_steps=2)
    assert out.shape[-2:] == (64, 64)
    (g,) = torch.autograd.grad(out.abs().mean(), img)
    assert g.abs().sum() > 0


def test_reference_size_matches_pipeline_arithmetic_at_1088():
    atk = _attack(_tiny_pipe(True)[0], max_area=1024 * 1024)
    # int() truncation then floor-to-16 exactly as Flux2KleinPipeline does
    import math
    s = math.sqrt(1024 * 1024 / (1088 * 1088))
    w = int(1088 * s); h = int(1088 * s)
    assert atk._reference_size(1088, 1088) == ((h // 16) * 16, (w // 16) * 16)


def test_hidden_state_loss_bounded_and_carries_gradient():
    pipe, _, _ = _tiny_pipe(True)
    atk = _attack(pipe, gs=1.0)
    _, img = _pil_and_tensor(64)
    same = atk.hidden_state_loss(img, img.clone().requires_grad_(True), ["x"], num_hooks=2)
    assert abs(same.item() - 1.0) < 1e-4          # identical -> cosine 1
    adv = (img + 0.2 * torch.randn_like(img)).clamp(-1, 1).requires_grad_(True)
    loss = atk.hidden_state_loss(img, adv, ["x"], num_hooks=2)
    assert -1.0 <= loss.item() < 1.0
    (g,) = torch.autograd.grad(loss, adv)
    assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_hidden_state_loss_lowers_with_checkpointing_and_double_blocks():
    pipe, _, _ = _tiny_pipe(True)
    atk = _attack(pipe, gs=1.0, ckpt=True)
    _, img = _pil_and_tensor(64)
    adv = (img + 0.2 * torch.randn_like(img)).clamp(-1, 1).requires_grad_(True)
    loss = atk.hidden_state_loss(img, adv, ["x"], group="both", num_hooks=2)
    (g,) = torch.autograd.grad(loss, adv)
    assert g.abs().sum() > 0


def test_tiny_unet_learns_through_edit_path():
    """End to end: a tiny NestedUNet lowers loss1 (edit output -> zeros)."""
    torch.manual_seed(0)
    pipe, _, _ = _tiny_pipe(True)
    atk = _attack(pipe, gtf=0.5, gs=1.0)
    net = NestedUNet(nb_filter=[4, 8, 16, 32, 64], epsilon=16)
    with torch.no_grad():
        net.final.weight.mul_(0.01)
    opt = torch.optim.Adam(net.parameters(), lr=3e-2)
    _, img = _pil_and_tensor(64)
    noise = torch.randn(1, 16, 4, 4, generator=torch.Generator().manual_seed(3))
    first = last = None
    for i in range(12):
        adv = (img + net(img)).clamp(-1, 1)
        out = atk.attack(prompt=["x"], image=adv, height=64, width=64,
                         num_inference_steps=2, latents=noise.clone())
        loss = out.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    assert last < first, (first, last)


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-path test")
def test_real_training_loop_with_edit_surrogate_epsilon_hs_and_l2(tmp_path, monkeypatch):
    """The REAL trainer + REAL FluxKleinEditAttack (tiny) + epsilon bound +
    hidden-state loss + normalised-L2 latent loss: completes, loss is finite,
    and the saved net's output respects the 16/255 pixel budget."""
    from test_training_smoke import _write_dataset
    from diffvax.attack_manager import AttackModelManager
    from diffvax.immunization.diffvax_immunization import DiffVaxImmunization
    import diffvax.attack_manager as _am
    import diffvax.immunization.diffvax_immunization as _di
    # The tiny pipeline lives on CPU; keep trainer and manager there too
    # (otherwise an MPS host moves the pipe/inputs to different devices).
    monkeypatch.setattr(_am, "resolve_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(_di, "resolve_device", lambda: torch.device("cpu"))

    torch.manual_seed(0)
    data_dir, out_dir = str(tmp_path / "d"), str(tmp_path / "o")
    os.makedirs(data_dir); os.makedirs(out_dir)
    entries = _write_dataset(data_dir)

    pipe, _, _ = _tiny_pipe(True)
    atk = _attack(pipe, gtf=0.5, gs=1.0)
    config = {
        "learning_rate": 5e-3, "resolution": 64, "batch_size": 1,
        "num_inference_steps": 2, "nb_filter": [4, 8, 16, 32, 64],
        "epsilon": 16, "dataloader": {"num_workers": 0},
        "latent_loss": {"enabled": True, "weight": 1.0, "mode": "l2"},
        "hidden_state_loss": {"enabled": True, "weight": 0.5, "num_hooks": 2},
    }
    manager = AttackModelManager(models={"flux": atk}, probabilities={"flux": 1.0})
    imm = DiffVaxImmunization(attack_manager=manager, config=config, output_dir=out_dir)
    result = imm.train_immunization_all_images_batch(
        entries, data_dir, "images", "masks", size=(64, 64), iter_num=4,
        SEED=5, batch_size=1, alpha=1, strength_range=[0.6, 0.9],
    )
    assert result is not None, "training aborted (NaN/Inf)"
    _, final_path = result
    net = NestedUNet(nb_filter=[4, 8, 16, 32, 64], epsilon=16)
    net.load_state_dict(torch.load(final_path, weights_only=True))
    _, img = _pil_and_tensor(64)
    with torch.no_grad():
        assert net(img).abs().max().item() <= 2 * 16 / 255 + 1e-6


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-path test")
def test_hidden_state_loss_gets_unaugmented_protected_image_with_eot(tmp_path, monkeypatch):
    """With EoT on, the clean image is NOT augmented, so the hidden-state term
    must be fed the UN-augmented protected image; otherwise JPEG/blur/noise
    alone would create 'separation' and be rewarded."""
    from test_training_smoke import _write_dataset, StubAttack
    from diffvax.attack_manager import AttackModelManager
    from diffvax.immunization.diffvax_immunization import DiffVaxImmunization
    import diffvax.attack_manager as _am
    import diffvax.immunization.diffvax_immunization as _di
    monkeypatch.setattr(_am, "resolve_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(_di, "resolve_device", lambda: torch.device("cpu"))

    seen = {}

    class Rec(StubAttack):
        def hidden_state_loss(self, img_orig, img_adv, prompt, **kw):
            seen["delta"] = (img_adv - img_orig).abs().max().item()
            return (img_adv - img_orig).abs().mean() * 0.0 + img_adv.mean() * 0.0

    data_dir, out_dir = str(tmp_path / "d"), str(tmp_path / "o")
    os.makedirs(data_dir); os.makedirs(out_dir)
    entries = _write_dataset(data_dir, n=1)
    config = {
        "learning_rate": 1e-3, "resolution": 64, "batch_size": 1,
        "num_inference_steps": 2, "nb_filter": [4, 8, 16, 32, 64],
        "epsilon": 16, "dataloader": {"num_workers": 0},
        # very strong additive noise: if it leaked into img_adv, delta >> eps
        "eot": {"enabled": True, "p_jpeg": 0.0, "p_resize": 0.0, "p_blur": 0.0,
                "p_noise": 1.0, "noise_std_range": [0.5, 0.5]},
        "hidden_state_loss": {"enabled": True, "weight": 0.5},
    }
    mgr = AttackModelManager(models={"flux": Rec()}, probabilities={"flux": 1.0})
    imm = DiffVaxImmunization(attack_manager=mgr, config=config, output_dir=out_dir)
    imm.train_immunization_all_images_batch(
        entries, data_dir, "images", "masks", size=(64, 64), iter_num=1,
        SEED=5, batch_size=1, alpha=1, strength_range=[0.6, 0.9])
    assert "delta" in seen, "hidden_state_loss was never called"
    assert seen["delta"] <= 2 * 16 / 255 + 1e-5, seen["delta"]


def test_reference_resize_matches_real_flux2_image_processor():
    """_prepare_reference_image vs the REAL Flux2ImageProcessor path on a 1088px
    image: same output size, and close pixels (bicubic-antialias vs LANCZOS)."""
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    proc = Flux2ImageProcessor(vae_scale_factor=16)
    yy, xx = np.mgrid[0:1088, 0:1088]
    arr = np.stack([
        (127 + 100 * np.sin(xx / 90.0)), (127 + 100 * np.cos(yy / 70.0)),
        (127 + 60 * np.sin((xx + yy) / 150.0))], -1).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    w, h = pil.size
    if w * h > 1024 * 1024:
        pil = proc._resize_to_target_area(pil, 1024 * 1024)
        w, h = pil.size
    ref = proc.preprocess(pil, height=(h // 16) * 16, width=(w // 16) * 16, resize_mode="crop")

    atk = _attack(_tiny_pipe(True)[0], max_area=1024 * 1024)
    t = torch.from_numpy(arr).permute(2, 0, 1)[None].float() / 127.5 - 1.0
    mine = atk._prepare_reference_image(t)
    assert mine.shape == ref.shape, (mine.shape, ref.shape)
    assert (mine - ref).abs().mean() < 0.01


@pytest.mark.parametrize("objective", ["encoder", "e2e"])
def test_pgd_respects_budget_and_lowers_objective(objective):
    from diffvax.pgd import make_objective, pgd
    pipe, _, _ = _tiny_pipe(True)
    atk = _attack(pipe, gtf=0.5, gs=1.0)
    _, img = _pil_and_tensor(64)
    obj = make_objective(atk, objective, num_inference_steps=2)
    torch.manual_seed(0)
    with torch.no_grad():
        before = obj(img, ["x"]).item() if objective == "encoder" else None
    hist = []
    adv = pgd(img, obj, ["x"], eps_pixels=16, steps=8, log=lambda i, l: hist.append(l))
    assert (adv - img).abs().max().item() <= 2 * 16 / 255 + 1e-6
    assert adv.min() >= -1 and adv.max() <= 1
    if objective == "encoder":
        with torch.no_grad():
            assert obj(adv, ["x"]).item() < before        # deterministic: must strictly improve
    else:
        assert min(hist[-3:]) <= hist[0] + 1e-6           # stochastic (random noise each step)
