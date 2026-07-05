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
