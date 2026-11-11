# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DiffVax is an optimization-free image immunization framework (ICLR 2026): a small trained NestedUNet
generates an additive perturbation that disrupts diffusion-based image editing in a single forward pass
(no per-image optimization, unlike PhotoGuard/DiffusionGuard baselines also implemented here).

This fork extends the original SD 1.5-only release with a research track targeting modern DiT-based
editors (SD3/SD3.5, FLUX.1/FLUX.2) via multi-surrogate training, additional loss terms, EoT augmentation,
and a resolution curriculum. **Read the "Project Status" section before trusting any protection-rate
claims for SD3/FLUX/closed-source targets** — that track is verified end-to-end by the CPU test suite
(gradient-path integrity, loss signs, real learning) but **not yet validated on GPU**. The original SD 1.5
pipeline (`app.py`, `scripts/demo.py`, `configs/train.yml`, `checkpoints/diffvax_trained.pth`) is the
published, evaluated paper result and is untouched by this research track.
Two shared-code changes slightly alter its numerics, both strictly more accurate: on CUDA the
compute dtype is now bf16 (was fp16) where supported, and `immunize_image_pil` runs in
float32 and rounds (rather than truncates) to uint8, a difference of at most one pixel level.
The perturbation `epsilon` bound is opt-in and absent for that checkpoint.

`research/findings.md`, `research/research-log.md`, and `research/research-state.yaml` are the live record
of what's been verified vs. what's still hypothesis — check them before assuming a research phase works as
described, and update them when you change verification status.

## Commands

```bash
# Install
pip install -r requirements.txt        # or: pip install -e .
pip install pytest

# Test (CPU-only; all but one GradScaler test run without a GPU)
python -m pytest tests/
python -m pytest tests/test_attack_gradient_flow.py::test_a1_sd3_attack_gradient_reaches_input
python -m pytest tests/ -k "gtf"       # by keyword, e.g. gradient_timestep_fraction cases

# Train (single-process; device auto-selected CUDA > MPS > CPU)
python scripts/train.py --config configs/train.yml --data-dir data --output-dir outputs
python scripts/train.py --config configs/research_v3.yml   # start here for the v3 hypothesis bundle
python scripts/train.py --config configs/train_1088_v3.yml --checkpoint <ckpt>.pth

# Train (multi-GPU DDP — one frozen surrogate pinned per rank, no config changes needed)
torchrun --nproc_per_node=3 scripts/train.py --config configs/full_v2.yml

# Demo / eval
python app.py                                              # Gradio web demo, SD 1.5 only
python scripts/demo.py                                     # single-image before/after comparison
python scripts/demo.py --run-baselines --no-display        # + PhotoGuard/DiffusionGuard comparison
python scripts/evaluate.py --checkpoint <ckpt>.pth
python scripts/eval_multimodel.py --checkpoint <ckpt>.pth  # protection metrics across a model zoo
python scripts/jpeg_robustness.py --checkpoint <ckpt>.pth --images data/validation/images
python scripts/download_dataset.py                         # also auto-runs on first train/demo
```

FLUX support needs a `diffusers` build with `Flux2KleinPipeline` (install from source if the released
version predates it). SD3.5 and FLUX.2 Klein are gated HF models — accept their license and set `HF_TOKEN`.
`kornia` is required only when a config sets `eot.p_jpeg > 0` (fails loudly if missing).

## Architecture

**Perturbation network**: `src/diffvax/model.py::NestedUNet` (UNet++), ~9M params by default
(`[32,64,128,256,512]` filters; `[64,128,256,512,1024]` variant is ~37M). GroupNorm throughout so it's
safe at `batch_size=1`. Output is a 3-channel perturbation added to the input image, then clamped to a
valid pixel range.

**Attack surrogates** implement `src/diffvax/attack_base.py::BaseAttack.attack()` — a fully
differentiable forward pass (VAE encode → denoising loop → VAE decode) so gradients flow from the loss
back through the frozen diffusion model to the perturbation network:

| File | Surrogate | VAE | Backbone |
|---|---|---|---|
| `attack.py` | SD 1.5 inpainting | 4-ch | UNet |
| `sd3_attack.py` | SD3 / SD3.5 | 16-ch | MM-DiT (joint bidirectional attention) |
| `flux_attack.py` | FLUX.2 Klein (img2img proxy; `flux_attack.mode: img2img`) | 16-ch | single-stream DiT |
| `flux_edit_attack.py` | FLUX.2 Klein **edit mode** (`flux_attack.mode: edit`): clean reference tokens at every step, generation from noise; parity-tested vs the real pipeline | 16-ch | single-stream DiT |

`gradient_timestep_fraction` runs only a fraction of denoising steps with the transformer's gradients
enabled (rest run under `no_grad`), trading VRAM for late-timestep signal without severing the gradient
chain — this is the mechanism `test_t1_*` in `tests/test_gradient_flow.py` exists to protect.

**`attack_manager.py::AttackModelManager`** selects a surrogate per training step (random by configured
probability, or gradient-aware adaptive reweighting — AdaEA-inspired — when `adaptive_ensemble` is
enabled). Only one surrogate is resident on GPU at a time; each surrogate's text encoder(s) are further
offloaded to CPU during backward, so peak VRAM is bounded by whichever surrogate is currently active, not
the sum of all configured surrogates. Under DDP, `adaptive_ensemble` and the sampling probabilities are
ignored — the effective mixture is determined by how many ranks are assigned to each surrogate instead
(see below).

**Multi-GPU (`distributed.py`)**: ensemble sharding, not model sharding. Each rank pins **one** frozen
surrogate for the whole run (round-robin by rank) and holds its own DDP replica of the ~9M-param
NestedUNet; only its gradients (~36 MB fp32) are all-reduced, so this needs no NVLink and is fine over
PCIe. The perturbation network learns from the whole ensemble every step rather than one sampled surrogate
per batch. The one hard requirement is that the *largest single surrogate* fits on one card. Tensor/
pipeline sharding of a surrogate is deliberately not implemented — putting the gradient path across device
boundaries is "the one thing in this codebase that has repeatedly broken silently" (see module docstring).
Any per-rank early exit (OOM skip, NaN abort) must go through `any_rank_true` to stay collective-safe.
Rank 0 alone writes checkpoints, Hub uploads, and the JSON training log.

**Perturbation budget**: `epsilon` (config, PIXEL units, e.g. 16 for 16/255) makes
`NestedUNet` output `eps_t * tanh(x / eps_t)` with `eps_t = 2*eps/255` (tensors are in
[-1, 1]). Unset = unbounded (legacy checkpoints). A raw `.pth` carries no epsilon: pass it
to `load_perturbation_net(..., epsilon=)` or the bound is silently absent at inference.
`resolve_dtype` is bf16 on CUDA when supported (GradScaler then disabled).

**Evaluation**: `scripts/eval_protocol.py` + `src/diffvax/protocol.py` (social-media
transform, matched-PSNR and same-L-inf noise controls, LPIPS edit divergence, image-level
bootstrap CIs). Use it, not `clip_delta` means, to judge protection.

**Loss terms** (`src/diffvax/losses/`), each opt-in via `enabled: true` in its config section — leaving
all of them `false` reproduces original v1 behavior exactly:
- `clip_loss.py` — CLIP feature/prompt-alignment disruption (architecture-agnostic; the one thing shared
  across closed-source editors too)
- `spectral_loss.py` — penalizes low-frequency perturbation energy
- `latent_loss.py` — pushes the adversarial image's VAE latent away from the clean latent (targets 16-ch
  VAE / SD3/FLUX)
- `hidden_state_loss` (method on `FluxKleinEditAttack`) — cosine of clean vs protected reference-token hidden states at middle blocks; replaces the saturating attention-entropy term
- `attention_loss.py` — (saturates at ln N; off by default) maximizes cross-attention entropy in DiT blocks (`target_blocks: "middle"`, per
  DeContext)
- `flat_minima.py` — SAM-style sharpness regularization for cross-model transfer

Other opt-in phases: `eot.py` (differentiable JPEG/resize/blur/noise augmentation), `curriculum.py`
(multi-stage resolution schedule, e.g. 512→768→1024→1088px).

**Training loops** (`src/diffvax/immunization/`): `diffvax_immunization.py` is the main multi-surrogate
loop wiring all the above together; `photoguard_immunization.py` and `diffusionguard_immunization.py` are
PGD-optimization baselines used only for comparison figures, not trained models.

**Device handling** (`utils.py`): every entry point auto-selects CUDA > MPS > CPU via
`resolve_device()`, with no code changes needed cross-platform. MPS-specific quirks worth knowing before
touching device code: compute dtype defaults to bf16 on MPS (not fp16 — MPS fp16 kernel coverage is
historically unreliable) via `resolve_dtype()`; `GradScaler` is a no-op on MPS/CPU since bf16 doesn't need
fp16-underflow mitigation; seeded generation on MPS always uses a CPU `torch.Generator`
(`make_generator()`) since `torch.Generator(device="mps")` doesn't reproduce seeds reliably. kornia's
JPEG codec, `torch.fft` (spectral_loss), and DiT attention-backward under MPS's Metal SDPA have not been
verified on real Apple Silicon — treat MPS as "runs" not "performance-validated."

## Config system

Every file in `configs/` is a complete standalone training run (see each file's header comment for
intended use case / VRAM estimate) rather than an override layer — pick the closest one and edit it, or
diff against `configs/train.yml` to see what a research config changes. `sd_probability` /
`sd3_probability` / `flux_probability` select which surrogates train and must sum to 1.0 across whichever
are non-null. Config-to-code correspondence: each `configs/*.yml` top-level section (`eot`, `clip_loss`,
`spectral_loss`, `latent_loss`, `attention_loss`, `flat_minima`, `curriculum`, `adaptive_ensemble`,
`sd3_attack`/`flux_attack`) maps directly to the identically-named module in `src/diffvax/`.

## Testing conventions

`tests/test_gradient_flow.py` and `tests/test_attack_gradient_flow.py` are the load-bearing suites for
this research track — they drive the *real* attack/loss code (not mocks) through lightweight fake
diffusers pipelines to assert gradient-path integrity and loss-sign correctness end-to-end. When adding a
new loss term or modifying a surrogate's forward pass, add a case here proving gradients still reach the
adversarial input, not just that the function runs. `test_distributed.py` uses real 2-rank gloo process
groups (no GPU) to test DDP wiring. `test_device_resolution.py` covers the CUDA/MPS/CPU + dtype selection
logic in `utils.py`. Test IDs (`A1`–`A8`, `T1`, `H1`–`H8`, `C7`/`C8`/`C10`/`C11`...) correspond to
hypotheses/cycles tracked in `research/research-state.yaml` and `research/findings.md` — check there for
why a given test exists before changing its expected behavior.
