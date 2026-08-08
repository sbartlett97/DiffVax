# DiffVax High-Resolution Research Log

## 2026-03-26 — Bootstrap: Codebase Analysis

### Codebase Summary

**Perturbation Network (`src/diffvax/model.py`)**
- `NestedUNet` (UNet++) with dense skip connections
- Filter sizes: [32, 64, 128, 256, 512] → ~1.8M parameters
- Fully convolutional, resolution-agnostic
- 5 max-pool levels → minimum input must be multiple of 16 (32 for safety)
- At 1088px, bottleneck is 1088/16=68px — functional, not a hard limit
- No positional encodings — pure local receptive field

**Attack Surrogates**
- `Attack` (SD 1.5 inpaint): 4-ch VAE, UNet, 512px native — differentiable via straight-through VAE encode
- `FluxAttack` (FLUX.2 Klein): 16-ch VAE, DiT, 1024px native — differentiable via mode()
- `SD3Attack` (SD3/SD3.5): 16-ch VAE, MM-DiT, 1024px native — differentiable via mode()

**v2 Features (all config-gated)**
- Phase 1: EoT (JPEG, resize, blur, noise augmentation)
- Phase 2: CLIP feature + semantic disruption loss
- Phase 3: SD3/FLUX 16-ch VAE support
- Phase 4: Multi-resolution curriculum (512→768→1024→1088)
- Phase 5: Adaptive ensemble weighting (gradient similarity)
- Phase 6: Flat-minima regularization (grad_norm or SAM)
- Phase 7: Cross-attention disruption (entropy maximization in DiT blocks)

**High-Resolution Configuration (`configs/full_v2.yml`)**
- Target: 1088×1088 final stage (batch_size=1)
- Curriculum: 512(500k) → 768(700k) → 1024(900k) → 1088(1000k epochs)
- VRAM estimates: SD3.5 needs ~26-28GB (40GB recommended)
- Protection targets: >75% SD1.5, >65% SDXL/SD3.5/FLUX

### Identified Limitations and Research Gaps

1. **NestedUNet may be undercapacitated at 1088px**: [32,64,128,256,512] filters is shallow. At high resolution, the perturbation needs longer-range context for coherent patterns.

2. **Gradient signal quality degrades with resolution**: The backprop path through SD3.5 at 1088px passes through 18,496-token joint attention. Gradient magnitude at the UNet++ input is very small after long chains. May cause vanishing gradients.

3. **Loss1 pushes to zero (black) in pixel space**: Semantically arbitrary for transformer models. A black image is still a valid image that transformers may handle gracefully. Latent-space disruption may be more effective.

4. **DALL-E 3 not covered**: No surrogate exists. CLIP-H proxy is the only feasible white-box approximation.

5. **Multi-scale robustness**: Curriculum trains at one resolution per batch — perturbation optimized at 1088 may not generalize down to 512 (which is what users might actually upload).

6. **EoT resize_range [0.5, 2.0]** could cause the image to become 2x larger during augmentation (2176px), which is very expensive at 1088px base.

### Direction Decided

Start with H3 (latent-space loss) as it is the most mechanistically well-motivated change for improving effectiveness against transformer-based models without requiring architecture changes. This can be validated quickly by measuring protection rate on FLUX/SD3 before and after.

Secondary priority: H2 (scaled-up NestedUNet) and H7 (multi-scale gradient aggregation) for the high-resolution scaling problem.

---

## 2026-03-26 — Literature Survey Queued

Searching for:
1. Image immunization / adversarial perturbation against diffusion models
2. Adversarial transferability to vision transformers
3. Latent-space adversarial attacks
4. High-resolution adversarial examples
5. DALL-E 3 vulnerability research

---

## 2026-03-27 — Inner Loop Cycle 1: Bundle H1+H2+H4+H7

### Decisions

Prioritised the 4-hypothesis bundle over individual validation runs because:
- H2 (VRAM reduction) is a hard dependency for all 1088px training — must ship first
- H1 (attention fix) corrects a production bug — early blocks wrong per DeContext
- H7 (noise target) is a one-line change with high expected impact
- H4 (TGR) is 40 lines of backward-hook code that pairs naturally with H2

**Critical bug found**: `attention_loss.target_blocks: "early"` in `full_v2.yml`.
DeContext (arXiv:2512.16625) is unambiguous — middle blocks carry the primary
context signal in MM-DiT.  Early-block targeting was providing essentially no
gradient disruption signal for SD3.5/FLUX.  Fixed in all v3 configs and corrected
in `full_v2.yml`.

### Implementation Notes

- H2: `gradient_timestep_fraction` added to `SD3Attack` and `FluxAttack`
  constructors.  Denoising loop conditionally runs with `torch.no_grad()` after
  first `fraction * len(timesteps)` steps.  Verified that `vae.encode()` path
  still receives gradients (it runs before the denoising loop).
- H4 TGR: backward hooks normalise per-token gradient magnitude.  Avoids the
  18,496-token joint attention variance that causes gradient explosion at 1088px.
  Hook registered on every transformer_block, removed after backward.
- H7: `torch.randint(0, 2, shape) * 2 - 1` gives ±1 uniform random noise.
  Config-gated so v1/v2 behaviour (zeros target) is preserved when disabled.
- H1: Added `"middle"` case to `AttentionDisruptionLoss._should_hook()`.
  Samples `num_hooks=8` evenly from the central third of total block depth.

### Configs Created

- `configs/research_v3.yml`: 50k iter research run at 512px, all 5 v3 hypotheses
- `configs/train_1088_v3.yml`: 500k iter production, stage-2 from v3-512 checkpoint

---

## 2026-03-27 — Inner Loop Cycle 2: H5 + H6

### H5: SpectralLoss

New `src/diffvax/losses/spectral_loss.py` module.  Uses `torch.fft.rfft2` with
`norm="ortho"` to compute the 2D frequency spectrum of `δ = img_adv − img_orig`.
Penalises the mean magnitude of frequency components with normalised distance
from DC < `low_freq_radius` (default 0.1).  Resolution-adaptive mask cached by
(H, W, device) to avoid rebuilding every forward pass.

Wired into `LossComposer` under `"spectral"` term.  Prediction: SSIM +0.02–0.05,
PSNR +1–3 dB at same epsilon budget.

### H6: Configurable NestedUNet

Added `nb_filter: list | None = None` to `NestedUNet.__init__`.  Stored as
`self.nb_filter` so `PyTorchModelHubMixin` serialises it to `config.json`.
Default unchanged: `[32, 64, 128, 256, 512]`.  Larger variant: `[64, 128, 256, 512, 1024]`.

**Surprise**: Actual parameter counts are ~9M (small) and ~37M (large), not
~1.8M/~7M as estimated.  The UNet++ dense skip connections at each level create
many more parameters than a plain U-Net of equal depth.  Both variants are
negligible VRAM relative to attack surrogates (<150 MB fp32).

### Bugs Fixed in Same Session

1. `_loss_composer` was only initialised when `clip_loss.enabled: true`.
   Spectral loss would silently not run if only `spectral_loss.enabled: true`.
   Fixed: initialise when EITHER term is enabled.
2. `spectral_loss` and `nb_filter` were missing from `immunization_config` in
   `scripts/train.py`.  Both config keys were silently ignored when launching
   via the train script.

### Research State: Outer Loop 1

All 7 hypotheses implemented.  No training validation yet.  Next priority:
run `research_v3.yml` for 5k iterations to validate loss convergence and
check VRAM at 1024px with partial-timestep gradient.

---

## 2026-07-05 — Inner Loop Cycle 3: Training-Signal Audit (C7, C8) + Real-Code Gradient Tests

### C7 (critical): H8 latent loss sign was inverted

`diffvax_immunization.py` computed `loss_latent = (1 - cos_sim(lat_orig, lat_adv))`
and ADDED it to the minimized total loss. Minimizing `1 - cos_sim` maximizes
similarity — i.e. it rewarded keeping the adversarial latents IDENTICAL to the
clean latents, the exact opposite of the H8 disruption objective, and actively
pushed the perturbation toward zero. Both flagship configs
(`research_v3.yml`, `train_1088_v3.yml`) shipped with `latent_loss.enabled: true`,
so every planned training run carried this anti-protective term at weight 1.0.

Fix: extracted `src/diffvax/losses/latent_loss.py::latent_disruption_loss`,
which returns the cosine SIMILARITY (minimize → push apart, bounded [-1, 1]).
Regression tests in `tests/test_attack_gradient_flow.py` (A3).

### C8 (critical): C1-era partial-timestep truncation still severed the chain

The C1 fix kept the LAST `n_grad_steps` under `enable_grad` and ran earlier
steps wholly under `no_grad`. But the latent chain
`vae.encode → step_0 → … → step_N → vae.decode` is the ONLY gradient route back
to `img_adv` (transformer frozen, prompts detached): the first whole-step
`no_grad` detaches `noisy_latents`, and every later step operates on a tensor
with `requires_grad=False`. Result: for ANY `gradient_timestep_fraction < 1.0`
with a multi-step schedule, loss1's gradient was silently zero. Because loss2
(perturbation magnitude) still had gradient, training would not crash — it
would converge to a null perturbation. Both flagship configs use `gtf=0.5`.

Fix (straight-through latent path) in `sd3_attack.py` and `flux_attack.py`:
skipped steps run only the TRANSFORMER under `no_grad` (detached noise_pred —
this is where the activation VRAM lives), while `scheduler.step` always
executes with grad enabled so the additive Euler integration path stays
connected end to end. This also restores the literature-grounded intent
(Distraction CVPR 2024): the FIRST `n_grad_steps` (early, high-sigma) now get
transformer Jacobians, not the last.

### New test harness: real attack classes, not stubs

`tests/test_attack_gradient_flow.py` drives the REAL `SD3Attack.attack` and
`FluxAttack.attack` with lightweight fake diffusers pipelines (fake 16-ch VAE,
fake MM-DiT/DiT, FlowMatch-style scheduler):
- A1/A2: nonzero finite gradient reaches `img_adv` at gtf ∈ {0.25, 0.5, 1.0}
  through both attacks (fails on the pre-fix code).
- A3: latent-loss sign regression (identical → 1.0, anti-aligned → −1.0).
- A4: mini learning test — a tiny NestedUNet trained through the real
  SD3Attack loop at gtf=0.5 measurably reduces loss1 in 30 steps. First
  end-to-end evidence on CPU that the training method produces a usable
  learning signal with partial-timestep gradient enabled.

### New property discovered (documented in test)

At `strength=1.0` with a flow-matching schedule, `t_start=0` and `sigma_0=1.0`,
so the init mix `(1-sigma)*latents + sigma*noise` contains ZERO image
contribution: the generation is unconditional and the image gradient is
mathematically zero. `strength_range` upper bound 1.0 is safe only because
`int(n*strength) < n` for strength just below 1.0 (t_start ≥ 1). Eval and
training should treat strength≈1.0 batches as protection-irrelevant.

### Status

29 tests pass on CPU torch. Remaining known gaps for full confidence:
1. `DiffVaxImmunization.train_immunization_all_images_batch` is CUDA-hard-coded;
   a device-agnostic refactor would allow the FULL training loop (dataset,
   EoT, LossComposer, scaler, curriculum) to run as a CPU smoke test.
2. No quantitative training validation yet (needs GPU).
3. TGR (H4) remains disabled pending register_full_backward_hook rewrite.

### Addendum (same session): device-agnostic training loop + full-loop CPU smoke test

Made `DiffVaxImmunization` device-agnostic (`self.device` = cuda when
available, else cpu): GradScaler becomes a passthrough on CPU, the dataset
streams fp32 instead of fp16 on CPU, DataLoader pin_memory follows CUDA
availability, and all hard-coded `"cuda"` literals in the loop/LossComposer/
AttackModelManager now resolve at runtime. GPU behaviour is unchanged.

New `tests/test_training_smoke.py` runs the REAL
`train_immunization_all_images_batch` end to end on CPU (disk dataset →
NestedUNet → clamp → stub attack → loss1/loss2 → scaler → optimizer →
checkpoints → JSON reporter) for 8 epochs with a tiny NestedUNet and asserts:
(S1) final checkpoint saved, (S2) epoch-average loss decreases, (S3) model
parameters changed (no silent step-skipping). This is the first automated
proof that the training method's own plumbing learns.

Also made `diffvax.immunization` package imports lazy (PEP 562) so the
DiffusionGuard baseline's cv2 dependency is no longer required for core
training. Suite: 30 tests, 29 pass on CPU + 1 CUDA-only skip.

---

## 2026-07-05 — Inner Loop Cycle 4: Loss-Path Audit (EoT, CLIP, Attention, Spectral, Flat-Minima, Eval)

### Audit results

| Component | Verdict |
|-----------|---------|
| `eot.py` | Sound. All four transforms differentiable; kornia absence fails loudly (S4); final clamp correct. |
| `clip_loss.py` | Sign conventions CORRECT (minimize orig↔adv feature cos-sim; minimize out↔prompt alignment). Still hard-codes CUDA — acceptable (GPU-only dependency), noted. |
| `spectral_loss.py` | Sound. rfft2 low-freq penalty on δ, resolution-normalized, correct mask folding. |
| `attention_loss.py` | See C9 investigation below — NOT dead, hardened anyway. |
| `flat_minima.py` | Mathematically inert under Adam: a uniform scalar on all grads cancels in Adam's per-parameter normalization (m/√v is scale-invariant). Harmless but a placebo — documented, left as-is (off by default). |
| `eval_multimodel.py` | Metric direction correct: `clip_delta = clip_no_defense − clip_with_defense`, positive = protection. C5 strengths already fixed. |

### C9 investigated — hypothesis DISPROVEN, code hardened

Hypothesis: Phase 7 forward hooks capture detached activations under gradient
checkpointing, making the attention loss a silent no-op on DiT models.

Empirical test (torch checkpoint, both modes): with `use_reentrant=False`
(what both attacks use) hook-captured tensors KEEP `grad_fn` — non-reentrant
checkpointing builds the full graph and only drops saved tensors for
recompute-on-backward. Gradient through a loss built from the captured tensor
flows correctly. Only `use_reentrant=True` detaches. So Phase 7 is live in
the current code. (Initial config edits based on the wrong hypothesis were
reverted within the session.)

Hardening shipped anyway:
- `AttentionDisruptionLoss.compute()` now filters to gradient-carrying maps
  (maps captured during no_grad-skipped timesteps at gtf<1.0 ARE detached and
  previously added constant entropy terms into the average), warns once and
  returns 0 if every capture is detached (reentrant ckpt / full no_grad), and
  no longer hard-codes CUDA.
- `use_gradient_checkpointing` knob added to SD3Attack/FluxAttack and wired
  through train.py (profiling/debugging).
- New tests: A5a proves the attention loss produces nonzero grad on img_adv
  through the REAL checkpointed SD3 attack at gtf=0.5; A5b covers the
  all-detached warning path. Suite: 32 tests, 31 pass + 1 CUDA-only skip.

### Remaining confidence gaps (unchanged priorities)

1. GPU training validation of research_v3.yml (loss curves, protection rate).
2. TGR (H4) rewrite with register_full_backward_hook + measured baseline.
3. Closed-model (nano-banana/DALL-E-class) protection is only claimable via
   CLIP-H proxy transfer — needs empirical eval, cannot be proven from code.

---

## 2026-07-05 — Inner Loop Cycle 5: H4 TGR Rewrite + Metrics Audit + Confidence Assessment

### H4 TGR re-enabled with correct hook semantics

Rewrote TGR in both attacks using `register_full_backward_pre_hook`, whose
return value replaces grad_output (the old `register_backward_hook` return
replaced grad_input, corrupting gradients — the reason C4 force-disabled it).
Normalization is now scale-preserving: each token gradient is rescaled to the
mean token norm, removing token-to-token variance without changing global
magnitude (relevant under Adam). Hooks are persistent (registered lazily at
first attack call) so they fire during the actual training backward — and,
verified empirically, through non-reentrant checkpoint recomputation.
Tests A6a-c: equalization, checkpoint interaction, and a corruption
regression (finite nonzero grads with TGR active through the real attack).

### Metrics audit

skimage SSIM/PSNR, reference FSIM (phase congruency + Scharr GM), and
OpenCLIP score (normalized cosine ×100) — all standard and correct.

### Confidence assessment written to findings.md

Three tiers: (1) proven by tests — gradient integrity at all gtf, loss
signs, end-to-end learning, Phase 7 viability, TGR; (2) needs GPU — actual
protection rates and ablations; (3) unprovable from code — closed-model
(nano-banana/DALL-E 3) transfer, only claimable via black-box evals.

Suite: 35 tests, 34 pass + 1 CUDA-only skip.

---

## 2026-07-05 — Apple Silicon (MPS) Streamlining Pass

### Motivation

User asked whether further streamlining was possible for running the
multi-surrogate training pipeline on Apple Silicon MPS backends. An earlier
audit (same session, prior turn) found ~20 hardcoded `"cuda"`/`.cuda()`/
`.half()` call sites scattered beyond the `diffvax_immunization.py` device
refactor already done for CPU testing (C7-C9 cycles): every attack surrogate,
the attack manager, CLIP loss, both PGD baselines, the CLIP-score metric, and
every user-facing script/app.

### Shared resolution utilities (src/diffvax/utils.py)

Added `resolve_device()` (CUDA > MPS > CPU), `resolve_dtype(device)` (fp16
CUDA / bf16 MPS / fp32 CPU — MPS fp16 has historically incomplete/unreliable
kernel coverage; bf16 shares fp32's exponent range so needs no loss scaling),
`empty_cache(device=None)` (cuda/mps/no-op dispatch), and `make_generator(device, seed)`.

`make_generator` encodes a documented diffusers/PyTorch limitation:
`torch.Generator(device="mps")` does not reproduce seeded results reliably,
so diffusers recommends a CPU generator even when the pipeline itself runs
on MPS. This is NOT the same as the "device-matched generator" pattern used
for direct tensor sampling (e.g. the H7 fixed-noise-target cache in
`diffvax_immunization.py`), which still requires generator/output device
parity — for that one call site, MPS now samples on CPU and moves the
result rather than attempting `Generator(device="mps")` at all.

### Applied everywhere

- `attack.py`, `sd3_attack.py`, `flux_attack.py`: constructors accept an
  optional `dtype` override, default `resolve_dtype(resolve_device())`, used
  for `torch_dtype=` at `from_pretrained` time. Final `output.half()` casts
  replaced with `output.to(dtype)` (dtype already read from the live VAE
  params). `to_cpu()` cache-clearing and the FLUX "still on CPU" guard
  generalized to any accelerator, not just CUDA.
- `attack_manager.py`: `select_and_load()` now targets `resolve_device()`
  instead of a CUDA-or-CPU ternary.
- `losses/clip_loss.py`, `metrics/clip_score.py`: model/tensors moved via
  `resolve_device()`/`resolve_dtype()` instead of hardcoded `.cuda().half()`;
  `clip_score.py` previously ran entirely on CPU tensors while wrapped in a
  CUDA-only `autocast` context (harmless on torch ≥2.x, which silently
  disables autocast without a CUDA context, but fragile and left the model
  off any available GPU).
- `photoguard_immunization.py`, `diffusionguard_immunization.py`: VAE/UNet
  dtype casts now read from the live model (`next(vae.parameters()).dtype`)
  instead of hardcoded `.half()`; generators and cache-clearing
  device-agnostic.
- `diffvax_immunization.py`: `self.device` now goes through
  `resolve_device()`; OOM-recovery path's error-type string and log message
  generalized from "cuda_oom" (the substring match already covered "MPS
  backend out of memory" — text-only fix, not a functional gap).
- `app.py`, `scripts/demo.py`, `scripts/compare_baselines.py`,
  `scripts/eval_multimodel.py`: replaced hardcoded `.half().cuda()` input-tensor
  prep with `resolve_device()`/`resolve_dtype()`, and — a real pre-existing
  gap independent of MPS — added an explicit `attack_model.to_device(...)`
  call after `Attack(...)` construction. These scripts never moved the SD
  pipeline off its `from_pretrained` default location before calling
  `edit_image()`; the hardcoded `.cuda()` on the INPUT tensors masked this on
  a CUDA machine only in the sense that the immunization step (perturbation
  network only) worked, while the actual diffusion edit comparison may have
  been running on CPU. `scripts/train.py` needed no changes — it never
  touches device/dtype directly, all of it flows through the now-generic
  constructors.

### Verification

Added `tests/test_device_resolution.py`: monkeypatches
`torch.cuda.is_available`/`torch.backends.mps.is_available` to test the
CUDA>MPS>CPU priority logic, dtype-per-backend mapping, and the
`make_generator` CPU-fallback-for-MPS behavior — all without needing real
accelerator hardware. Full suite: 46 tests, 45 pass + 1 CUDA-only skip.

### Known limits (documented in README "Apple Silicon (MPS)")

No physical Apple Silicon hardware is available in this environment. What's
verified is the *selection logic* (device/dtype priority, generator
fallback) and that nothing regressed on CPU. What remains unverified:
kornia JPEG codec MPS kernel coverage (`eot.py`), `torch.fft` coverage for
`spectral_loss.py`, and MPS Metal SDPA backward correctness/performance for
the SD3/FLUX transformer blocks. This is a "runs without crashing"
streamlining pass, not an MPS performance/correctness validation — that
needs a real machine, same caveat structure as the GPU validation runbook
for CUDA.

---

## 2026-07-05 — C10 (critical): stale pipe.device after text-encoder offload

### Bug report (real MPS training run)

User hit a live crash after the MPS streamlining pass:
```
RuntimeError: slow_conv2d_forward_mps: input(device='cpu') and weight(device='mps:0')
must be on the same device
```
at `vae.encode(image_input)` in `SD3Attack.attack()`.

### Root cause

`SD3Attack.attack()` and `FluxAttack.attack()` both captured
`device = self.pipe.device` at the top of the call, then later in the SAME
call moved the text encoder(s) to CPU permanently (`enc.to("cpu")`) to save
RAM during the VAE/transformer-heavy backward pass — nothing ever moves them
back. `diffusers.DiffusionPipeline.device` (verified against the actual
diffusers source) returns the device of whichever component appears first
in the pipeline's constructor signature — for both pipelines that's a text
encoder, not the VAE or transformer. Once text encoders are parked on CPU
after the first `attack()` call, every subsequent call to the SAME
(unswitched) surrogate reads `pipe.device == "cpu"` while `vae`/`transformer`
are still on the real accelerator — silently, since `AttackModelManager`
only re-runs `to_device()` when the SELECTED surrogate changes, not on every
batch. Consecutive batches choosing the same surrogate (a normal outcome of
`random.choices`) hit this immediately.

`FluxAttack.attack()`'s own "still on CPU" guard (added this session for the
MPS pass) had the identical bug: it checked `self.pipe.device` too, and
would have raised its OWN false-positive RuntimeError on the same drift
before ever reaching the conv2d crash.

### Fix

Both `attack()` methods now derive `device` from `next(vae.parameters()).device`
(the same pattern already used for `dtype`), computed once `vae` is fetched
and never touched by the text-encoder offload. `FluxAttack`'s CPU guard was
reordered to run after this corrected device is known.

### Regression tests (A7)

`tests/test_attack_gradient_flow.py`: sets `atk.pipe.device = torch.device("meta")`
after construction (an unmistakable wrong-device sentinel — meta tensors
carry no storage, so any code path that still reads `pipe.device` errors
immediately instead of quietly misbehaving) and asserts `attack()` still
succeeds and produces a correctly gradient-carrying output for both
SD3Attack and FluxAttack. Suite: 47 tests, 46 pass + 1 CUDA-only skip.

### Lesson

This is the same bug *shape* as C1/C7/C8/C4 this session: a plausible-looking
API call (`pipe.device`) that is quietly wrong under a specific stateful
sequence (repeated selection of the same surrogate) that unit tests with a
single `attack()` call per test never exercised. Worth generalizing: any
future per-call "convenience" reads of aggregate pipeline state should be
checked against what the code itself mutates elsewhere in the same class.

---

## 2026-07-05 — TGR full-backward-hook warning explained and silenced

### Investigation

User reported (from a real MPS training run with `token_gradient_regularization: true`):
```
UserWarning: Full backward hook is firing when gradients are computed with
respect to module outputs since no inputs require gradients.
```

Confirmed the precise mechanism empirically (not from memory/speculation):
`register_full_backward_pre_hook`'s input-tracking cannot see a
gradient-requiring tensor when the module is called with **all-keyword
arguments and zero positional arguments** — it defensively concludes "no
inputs require gradients" and warns, even though the hook still receives the
*correct* `grad_output` and gradients still flow correctly. Verified with a
minimal repro: identical block/hook, called positionally → no warning;
called as `block(hidden_states=hs, encoder_hidden_states=enc)` (kwargs
only) → warning fires every time. Further verified the hook's captured
`grad_output` matches the analytically-expected value in the kwargs-only
case — i.e. the warning has zero bearing on correctness.

Cross-checked against the real `diffusers` source (downloaded and unpacked
the actual wheel): `SD3Transformer2DModel.forward()` calls each block as
`block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
temb=temb, joint_attention_kwargs=joint_attention_kwargs)` — exactly the
all-keyword pattern that triggers this. TGR's hooks (added this session,
H4 rewrite) are attached to these blocks, so this fires on every backward
whenever `token_gradient_regularization: true`.

### Fix

New `attack_base.suppress_full_backward_hook_kwarg_warning()` installs a
message-scoped `warnings.filterwarnings("ignore", ...)` (not a blanket
ignore-all), called once from both `SD3Attack._register_tgr_hooks` and
`FluxAttack._register_tgr_hooks`. New regression test constructs a
kwargs-only block + hook matching the real pattern and asserts the warning
is filtered — verified non-vacuous by confirming the warning fires without
the filter using the identical setup. Suite: 48 tests, 47 pass + 1 CUDA-only
skip.

---

## 2026-07-05 — C11 (critical): text-encoder CPU offload made CUDA-only per user directive

### Bug report (real MPS training run, third crash in this sequence)

```
RuntimeError: Placeholder storage has not been allocated on MPS device!
```
in `self.pipe.encode_prompt(...)` → CLIP text encoder's token embedding
lookup, on a LATER call to `SD3Attack.attack()` after at least one prior
call had already run.

### Root cause

Both `SD3Attack.attack()` and `FluxAttack.attack()` move their text
encoder(s) to CPU at the end of every call (`enc.to("cpu")`) to save VRAM
during the VAE/transformer-heavy backward pass, and never move them back —
relying on `AttackModelManager.select_and_load()`'s whole-pipeline
`to_device()` to restore them only when the SELECTED SURROGATE CHANGES
(consistent with the C10 fix earlier this session). HuggingFace pipelines
loaded with the modern default (`low_cpu_mem_usage=True`) instantiate
parameters lazily via a meta-device placeholder, materialized in place when
first dispatched to a real device. Manually yanking an individual submodule
back and forth between an accelerator and CPU via raw `.to(device)` calls —
bypassing `accelerate`'s dispatch/hook machinery — was observed in practice
to leave the text encoder's parameters as unmaterialized placeholder storage
on a subsequent call, crashing on the very first op that reads a parameter
(the token embedding lookup).

This is architecturally moot on Apple Silicon regardless of the crash: MPS
and CPU share the same physical (unified) memory, so moving a submodule to
"cpu" frees no memory the way it does on CUDA's separate VRAM pool — the
offload was providing zero benefit on MPS while being an active liability.
User's explicit directive: "New plan for MPS - DO NOT OFFLOAD ANYTHING TO
CPU."

### Fix

Both text-encoder-offload blocks are now gated behind `device.type == "cuda"`
(`device` already correctly resolved from `vae.parameters()` per the C10
fix). CUDA behavior is unchanged; MPS and CPU never touch the text
encoder(s) after the initial whole-pipeline placement.

### Regression tests (A8)

`tests/test_attack_gradient_flow.py`: `FakeSD3Pipe`/`FakeFluxPipe` gained
real `nn.Linear` text-encoder stand-ins; a `_ToCallSpy` wraps their `.to`
method (necessary because calling `.to("cpu")` on an already-CPU module is a
silent no-op — final-device inspection alone can't detect whether the call
happened) and asserts no "cpu" move occurs when `attack()` runs off-CUDA.
Verified non-vacuous by confirming the spy does catch an unconditional
`.to("cpu")` call. Suite: 50 tests, 49 pass + 1 CUDA-only skip.

---

## 2026-07-05 — Multi-GPU: ensemble sharding via DDP (one surrogate per rank)

### Question

User asked whether the pipeline could be distributed across multiple GPUs
with pipeline/tensor parallelism, motivated by cost: renting one
large-memory card is expensive relative to a box of smaller cards.

### Analysis — why the standard answers don't apply

Checked the actual diffusers 0.39 source rather than assuming:
- `SD3Transformer2DModel` has **neither** `_tp_plan` nor `_cp_plan` — diffusers'
  built-in parallelism does nothing for SD3.5.
- FLUX2 has a `_cp_plan`, but that is *context* parallelism (shards the
  sequence to cut activations); every rank still holds full weights, so it
  does not enable smaller cards for a model whose weights don't fit.
- `_modeling_parallel.py`'s own error strings say "context parallel
  **inference**" — nothing there is validated for backward.

FSDP / DeepSpeed-ZeRO are also the wrong tool: they shard optimizer state,
gradients and parameters of the model being *trained*. Here the trained model
is the ~9M-param NestedUNet with trivial optimizer state; the memory is spent
on a **frozen** surrogate's weights plus activations retained for a backward
pass that only produces a gradient w.r.t. its *input*. ZeRO has no lever on
either. This is an unusual enough shape that the reflexive answer is wrong.

### Implemented: shard the ensemble, not the model

`src/diffvax/distributed.py` + DDP integration. Each rank pins ONE frozen
surrogate for the whole run (round-robin by rank) and holds a replica of the
NestedUNet under DDP. Only the NestedUNet's gradients are all-reduced (~36 MB
fp32/step) — fine over plain PCIe, no NVLink needed. Requirement reduces to:
the *largest single surrogate* must fit on *one* card.

Bonus: this is a research improvement, not just a cost dodge. The
single-process path samples ONE surrogate per batch (high-variance gradient);
under DDP the perturbation net receives the true ensemble gradient every step.

### Collective-safety — the real hazard

DDP synchronises every step, so any per-rank early exit deadlocks the job.
Three paths needed fixing:
1. **OOM skip**: `continue` on one rank left peers blocked forever in the
   gradient all-reduce. Now `any_rank_true()` agrees the skip globally first.
2. **NaN abort**: `return` on one rank, same hazard. Also collective now.
3. **Epoch loss / best-model**: ranks comparing *local* losses would disagree
   about whether to checkpoint. Now `all_reduce_mean()`d so the branch is
   entered in lockstep; only rank 0 writes.

Also: `DistributedSampler(drop_last=True)` so every rank runs an identical
number of batches (an uneven count is the same deadlock), `set_epoch()` per
epoch, seed offset by rank for augmentation diversity (does not perturb the
sampler, which carries its own seed, nor the H7 fixed target), and
`self.unetmodel(...)` instead of `.forward()` — the latter bypasses DDP's
sync hooks entirely.

`scripts/train.py` now builds surrogates lazily so each rank constructs ONLY
its own; previously it built all enabled surrogates, which under DDP would
have instantiated the full ensemble on every card and OOM'd immediately.

### Tests

`tests/test_distributed.py` — 6 tests, gloo/CPU, no GPU required:
- helpers degrade to safe identities single-process;
- `init_distributed()` declines rather than hanging without rendezvous env;
- real 2-rank collectives: `all_reduce_mean` averages across ranks, and a
  flag raised on rank 1 alone is observed by rank 0 (the property the
  OOM/NaN paths depend on);
- **load-bearing**: 2 ranks run the REAL training loop and must end with
  bit-identical weights. Verified non-vacuous — swapping
  `self.unetmodel(...)` back to `self._unet_module(...)` (bypassing DDP)
  makes it fail exactly as intended, then restored.
- rank 0 alone writes checkpoints.

Suite: 57 tests, 56 pass + 1 CUDA-only skip.

### Not verified

gloo/CPU only — no multi-GPU hardware available. NCCL behaviour, real
per-card memory, and straggler cost (the slowest surrogate gates every step,
so an SD 1.5 rank will idle waiting for SD3.5) are all unmeasured.
Deliberately not implemented: tensor/pipeline parallelism *inside* a
surrogate — that puts the gradient path across device boundaries, the exact
failure mode that produced C1/C7/C8/C10 this session.
