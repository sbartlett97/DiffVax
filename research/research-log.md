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
