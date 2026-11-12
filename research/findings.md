# DiffVax Research Findings

**Last updated:** 2026-11-11 (see the section below; earlier sections predate it)
**Status:** Inner loop cycle 2 complete — all 7 hypotheses implemented, awaiting training validation

---

## Current Understanding

DiffVax trains a NestedUNet (UNet++) to generate imperceptible pixel-space perturbations
that disrupt downstream diffusion model editing in a single feedforward pass (250,000× faster
than iterative PGD baselines). The v2 implementation includes multi-surrogate training
(SD1.5, FLUX.2, SD3.5), EoT augmentation, CLIP loss, and cross-attention disruption.
Training at 512px against SD 1.5 is proven. The research challenge is:
1. **Scaling to 1088×1088** (high-res social media protection)
2. **Disrupting SOTA transformer-based DiT models** (FLUX.1/2, SD3, SD3.5, DALL-E 3)

### Key Architecture Facts
- Perturbation network: NestedUNet [32,64,128,256,512] ~1.8M params, resolution-agnostic
- Attack surrogates: SD 1.5 (4-ch VAE, UNet), FLUX.2 Klein (16-ch VAE, DiT), SD3.5 (16-ch VAE, MM-DiT)
- Epsilon budget: 32/255 in pixel space
- Gradient path: UNet++ → pixel clamp → (EoT) → attack model → loss
- Current loss: L1 push-to-black (loss1) + L1 perturbation magnitude (loss2) + CLIP + attention

---

## Patterns and Insights (from Literature Survey)

### P1: DiT models use a fundamentally different protection mechanism
Literature (DeContext arXiv:2512.16625, Distraction CVPR2024, Immunizing via Cross-Attention ACM MM2025):
- **Encoder-space attacks lose effectiveness on DiT models** — DiT does not rely on the same VAE bottleneck
- Context flows through **multimodal attention at early-to-mid timesteps (t > 0.7)**
- **MIDDLE transformer blocks** are the critical disruption targets, not early blocks
- DiffVax Phase 7 currently targets "early" blocks — this is **wrong for DiT protection**
- Fix: change `attention_loss.target_blocks` to "middle" and restrict gradient computation to t > 0.7

### P2: Token gradient variance causes gradient instability in DiT backpropagation
Literature (TGR CVPR2023 arXiv:2303.15754):
- High token-to-token gradient variance in ViT/DiT blocks causes poor adversarial transfer
- At 1088px, SD3.5's joint attention has ~18,496 tokens — variance extremely high
- Token-wise gradient normalization during backprop → 8.8% transfer improvement
- This is the most likely cause of poor gradient signal quality at 1088px

### P3: Partial timestep backpropagation matches full backprop at 50% VRAM
Literature (Distraction Is All You Need CVPR2024):
- Selecting the K most informative timesteps for gradient computation is equivalent to full backprop
- 50% VRAM reduction directly addresses the 26-28GB bottleneck for SD3.5 at 1088px
- Brings SD3.5 into feasibility range for 24GB GPUs

### P4: CLIP embedding disruption transfers to DALL-E 3 and closed-source models
Literature (Glaze arXiv:2302.04222, DTIA Springer2024, NL Adversarial arXiv:2410.08620):
- CLIP is the shared semantic interface for DALL-E 3, SD, FLUX, SDXL, and Midjourney
- Disrupting CLIP-H/14 embeddings provides model-agnostic protection
- DiffVax v2 CLIP loss uses ViT-B/32 — upgrading to ViT-H/14 covers DALL-E 3

### P5: High-frequency perturbation concentration improves imperceptibility
Literature (DDAP arXiv:2407.20141, AdvAD NeurIPS2024):
- High-frequency perturbations are less perceptible at same L-inf norm
- DCT/wavelet frequency domain penalties push perturbation energy into imperceptible bands
- At 1088px, L-inf ε=32/255 in high-frequency domain looks essentially invisible

### P6: Loss target selection matters — sharp/noisy target > smooth black
Literature (Mist arXiv:2305.12683):
- Textural disruption loss is sensitive to target image selection
- Sharp-edges / high-contrast / noise patterns significantly outperform smooth targets (black image)
- Current DiffVax loss1 target = zeros (black image) — suboptimal, especially for DiT models

### P7: Content-conditioned perturbations are more resistant to averaging attacks
Literature (Hönig et al. ICLR2025):
- Pixel-space UAP perturbations defeated by averaging 50+ samples
- DiffVax's content-conditioned NestedUNet produces image-specific perturbations
- These are harder to average out than fixed UAPs (Glaze, Mist)
- But still vulnerable — scope protection claims to < 10 images per subject

---

## Lessons and Constraints

- Resolution must be multiple of 16 (hard requirement for NestedUNet 5-level pooling)
- 1088 = 17×64, chosen as next multiple of 64 above 1024 (also multiple of 16)
- EoT resize_range [0.5, 2.0] is OOM risk at 1088px — should cap at [0.7, 1.5] for 1088px stage
- `enable_mem_efficient_sdp(False)` required for backward pass stability
- SD3.5 joint attention: 1024px = ~16k tokens, 1088px = ~18.5k tokens — O(n²) memory
- Phase 7 hooks "early" transformer blocks — MUST change to "middle" per DeContext paper
- NestedUNet bottleneck at 1088px is 68×68 — spatially adequate but filter count [32-512] may be tight
- DALL-E 3 requires no direct model access — CLIP-H/14 is the appropriate proxy
- DiffJPEG requires separate installation; graceful fallback exists if missing
- Gradient via GradScaler + AMP requires flash attention (not mem_efficient SDPA)

---

## Open Questions

1. **[Experiment H3-revised]** Does changing Phase 7 attention hooks from "early" to "middle" blocks improve DiT protection rate?
2. **[Experiment H-TGR]** Does TGR token-wise gradient normalization in SD3.5 backward pass stabilize 1088px training?
3. **[Experiment H-VRAM]** Does restricting backward pass to t > 0.7 timesteps in SD3.5 achieve equivalent protection at 50% VRAM?
4. **[Experiment H-CLIP-H]** Does CLIP ViT-H/14 loss provide measurable protection against DALL-E 3 proxy (CLIP cosine distance)?
5. **[Experiment H-FREQ]** Does spectral/frequency-domain perturbation penalty improve SSIM at same epsilon budget?
6. **[Experiment H2]** Does scaling NestedUNet to [64,128,256,512,1024] improve protection rate at 1088px?
7. **[Threat model]** What is the minimum number of images needed to break DiffVax with robust mimicry?
8. **[C-SCALER, from real runs' logs — UPDATED]** A live GPU run
   (research_v3.yml-class mixture, `[sd3]` selected, ~180 epochs / ~10,440
   steps in) showed `[Scaler] scale=0.000e+00` holding for 400+ consecutive
   batches while loss values stayed superficially normal (4.03-4.31, no
   trend). A subsequent abort-guard-equipped run collapsed within 3 epochs
   instead (see research-log.md 2026-09-03), giving much earlier evidence:
   `found_inf` fired on ~85% of steps by batch 100, on a freshly-initialized,
   in-distribution first batch — ruling out training dynamics or perturbation
   magnitude as the driver.
   - **Confirmed by ablation**: a loss1+loss2-only run (every optional term
     — CLIP/spectral/latent/attention — disabled) reproduced the identical
     ~85% found_inf rate and collapse timing. Since loss2 never touches the
     surrogate, this rules out all four optional terms and narrows the cause
     to loss1's gradient path through SD3Attack itself.
   - **img_adv fp32 fix (first fix)**: removed a real, redundant fp16
     downcast that forced all six loss terms' backward gradients through one
     shared fp16 buffer. Correct to keep, but the ablation above shows it
     was NOT the primary collapse driver (loss1+loss2 alone, which never
     touches that shared buffer meaningfully, collapsed identically) — demote
     this from "the funnel" to "one redundant accumulation point removed."
   - **TGR fp16 norm overflow (leading candidate, second fix)**: found by
     code trace + CPU reproduction (not GPU) that
     `_tgr_backward_pre_hook` (sd3_attack.py / flux_attack.py) computes each
     token's gradient norm in fp16. Proven on CPU: an entirely ordinary raw
     gradient, once multiplied by GradScaler's scale and reduced over
     MM-DiT's wide hidden dim (1536), can produce a norm that legitimately
     exceeds fp16's 65504 max, which poisons `mean_norm` (shared across
     every token) into inf/NaN. Fixed with the same float32-upcast pattern
     as `_weighted_l1`. **Not yet confirmed as THE cause**: TGR was enabled
     (`token_gradient_regularization: true`) in every config file inspected,
     but the actual config path used by the real runs was never confirmed,
     and no `token_gradient_regularization: false` run has been done. That
     run is the decisive test — if disabling TGR alone drops the skip rate
     to ~0, this is confirmed; if skips persist, something else is also
     contributing and this investigation continues.
9. **[Loss-term saturation, same log window]** In that same run, `Attn`
   held at -6.9277±0.0005 (≈ -ln(1024), the max-entropy ceiling for a
   32x32-token map) across batches where `Loss1` swung 0.55-1.08 — i.e.
   near-zero sensitivity to input/attack content, possibly because
   `attention_loss.py`'s token-importance proxy (`attn_out.abs().mean(dim=-1)`)
   is close to uniform by construction after LayerNorm, pinning the softmax
   near-uniform regardless of any attack. `Lat` (latent cosine similarity)
   sat at 0.97-0.99, i.e. adversarial ≈ clean latents — the term is not
   achieving separation either. Both observations are from a window where
   the scaler had already collapsed (see #8) — every weight update in that
   window was a no-op, so this could just as easily be measuring a frozen
   model as measuring a genuinely saturated loss term. Re-check after a
   clean (non-collapsed) run before treating either as a real finding.

---

## Implementation Status (as of 2026-03-27)

All 7 research hypotheses implemented and config-gated. Awaiting training runs for
quantitative validation. Key implementation decisions and surprises:

| Hypothesis | Status | Key Implementation Note |
|-----------|--------|------------------------|
| H7 noise target | Implemented | `torch.randint(0,2,...)*2-1` in loss computation |
| H1 middle-block attn | Implemented | Changed `target_blocks: "early"→"middle"` (was a bug in v2) |
| H2 partial-timestep | Implemented | `gradient_timestep_fraction=0.5` — backprop only early steps |
| H3 CLIP-H/14 | Implemented | Config change only: `model: ViT-B/32→ViT-H-14` |
| H4 TGR hooks | Implemented | Backward hooks on DiT blocks, normalize per-token gradient |
| H5 spectral loss | Implemented | `rfft2` low-freq penalty, `SpectralLoss` in `LossComposer` |
| H6 larger model | Implemented | `nb_filter` param in `NestedUNet`; actual params 9M→37M (not 7M) |

**Surprise**: NestedUNet++ actual parameter count is ~9M (not ~1.8M as estimated).
Dense skip connections at each resolution create many more connections than a plain
U-Net. The larger variant is ~37M parameters. Both are still negligible VRAM vs
attack surrogates.

## Experiment Trajectory

*(Quantitative results pending first training run)*

| Run | Hypothesis | Key Change | Proxy Metric | Δ vs Baseline |
|-----|-----------|------------|--------------|---------------|
| bundle-01 | H1+H2+H4+H7 | noise target, middle-attn, TGR, partial-timestep | loss1, loss_attn | pending |
| spectral-01 | H5 | spectral_loss rfft2 low-freq penalty weight=0.5 | SSIM, PSNR | pending |
| large-net-01 | H6 | nb_filter [64,128,256,512,1024] | loss1 convergence speed | pending |

---

## Research Hypotheses (Literature-Grounded)

### H1: Middle-block critical-timestep cross-attention disruption [HIGH PRIORITY]
- **Basis**: DeContext (2512.16625), Immunizing via Cross-Attention (2509.10359)
- **Change**: Phase 7 `target_blocks: "middle"`, restrict gradient to timesteps where t > 0.7
- **Prediction**: >10% improvement in protection rate against FLUX/SD3.5 vs current early-block hooks
- **Side benefit**: Can be combined with partial-timestep VRAM optimization

### H2: Partial-timestep gradient for VRAM reduction [HIGH PRIORITY]
- **Basis**: Distraction Is All You Need (CVPR 2024)
- **Change**: Compute gradients only at critical timesteps (top-K by gradient magnitude); skip rest
- **Prediction**: Reduce SD3.5 1088px VRAM from ~26GB to ~13GB with <3% protection degradation
- **Immediate benefit**: Enables 1088px SD3.5 training on 24GB GPUs

### H3: CLIP ViT-H/14 loss for DALL-E 3 coverage [MEDIUM PRIORITY]
- **Basis**: Glaze, DTIA, NL Adversarial (all confirm CLIP-H as DALL-E 3 proxy)
- **Change**: Replace ViT-B/32 with ViT-H/14 in CLIP loss; add cosine-distance loss
- **Prediction**: Measurable CLIP-H embedding disruption (>0.3 cosine distance shift) → DALL-E 3 transfer

### H4: Token gradient regularization (TGR) in DiT backprop [MEDIUM PRIORITY]
- **Basis**: TGR CVPR2023 (2303.15754)
- **Change**: Normalize gradient magnitude token-wise during backprop through SD3.5/FLUX attention
- **Prediction**: More stable training at 1088px; improved perturbation quality under SD3.5/FLUX

### H5: Frequency-domain perturbation concentration [MEDIUM PRIORITY]
- **Basis**: DDAP (2407.20141), AdvAD (NeurIPS2024)
- **Change**: Add DCT high-frequency penalty to loss2 (penalize low-frequency perturbation energy)
- **Prediction**: Improved SSIM/PSNR at same epsilon budget; better imperceptibility at 1088px

### H6: Scaled-up NestedUNet [64,128,256,512,1024] [LOW PRIORITY — ablation ready]
- **Basis**: Capacity argument; TGR suggests gradient quality is the bottleneck, not network capacity
- **Change**: Double filter counts in NestedUNet — actual params 9M→37M (dense skip connections)
- **Prediction**: Modest improvement (+3-5%) at high resolution
- **Status**: Implemented via `nb_filter` config key; uncomment in research_v3.yml to test

### H7: Sharp/noise target image for loss1 [EASY WIN — validate first]
- **Basis**: Mist (2305.12683) target image selection insight
- **Change**: Replace zero (black) target in loss1 with high-frequency noise pattern
- **Prediction**: Improved disruption effect at minimal implementation cost; especially effective for DiT models with semantic priors that gracefully handle black outputs

---

## Verification Status & Confidence Assessment (2026-07-05, cycles 3-4)

The question this loop answers: *can the proposed training method learn a
model that protects images from editing by hybrid diffusion/transformer
models (FLUX, SD3.5, nano-banana-class editors)?*

### Tier 1 — Proven by automated tests (CPU, real code paths)

1. **Gradient path integrity.** Nonzero, finite gradient reaches the
   perturbation from loss1 through the REAL `SD3Attack`/`FluxAttack` loops
   (VAE encode → patchify/pack → flow-matching noise mix → denoising →
   decode) at every `gradient_timestep_fraction` ∈ {0.25, 0.5, 1.0} and at
   partial strengths. Two prior generations of this code silently violated
   this (C1: loss-side severed; C8: input-side severed — loss1 gradient was
   exactly zero in the flagship configs, and training would have collapsed
   the perturbation to zero via loss2).
2. **Loss sign conventions.** All active terms verified: loss1 (push edit
   output to target), loss2 (perturbation magnitude), CLIP feature/semantic,
   spectral low-frequency penalty, H8 latent disruption (was INVERTED —
   rewarding identical latents — until C7), attention entropy (negated
   correctly).
3. **End-to-end learning.** The real `train_immunization_all_images_batch`
   (dataset→UNet++→clamp→attack→losses→GradScaler→Adam→checkpoints) reduces
   its loss and updates weights on CPU with a stub surrogate; a tiny
   NestedUNet also reduces loss1 through the real SD3 attack loop at
   gtf=0.5.
4. **Phase 7 viability.** Attention-hook captures stay graph-connected under
   non-reentrant gradient checkpointing (hypothesis that checkpointing killed
   the loss was tested and DISPROVEN); detached captures from no_grad-skipped
   steps are now filtered out with a loud warning if nothing survives.
5. **H4 TGR.** Rewritten with `register_full_backward_pre_hook` (grad_output
   semantics — the old `register_backward_hook` version corrupted gradients
   and was force-disabled). Scale-preserving per-token norm equalization,
   verified to fire through checkpointing, and to keep attack gradients
   finite and nonzero.
6. **Degenerate strength.** At strength=1.0 the flow-matching init mix
   carries mathematically zero image signal — protection-irrelevant batches;
   training draws from [0.5, 1.0) are safe (t_start ≥ 1).

### Tier 2 — Requires GPU runs (cannot be proven from code)

- Actual protection rates against real FLUX.2 Klein / SD3.5 weights
  (research_v3.yml, then train_1088_v3.yml). All prior "pending training
  validation" claims still pending; the difference is that the training
  signal is now verified to exist, so a run measures the method rather
  than the bugs.
- Whether TGR/spectral/latent/attention terms each improve protection
  (ablations); whether gtf=0.5 matches gtf=1.0 quality at half VRAM.
- JPEG/resize robustness in practice (EoT path is differentiable and
  fail-loud, but robustness magnitude is empirical).
- **GradScaler collapse fix (2026-09-03)**: `img_adv` (and everything
  computed from it — loss1, loss2, CLIP, spectral, latent, attention) is now
  kept in float32 through the whole loss graph instead of being downcast to
  the dataset's native load dtype (fp16 on CUDA) right after the
  perturbation network. That downcast dates to the original single-surrogate
  v1 loop (`unet_out.half().cuda()`, predating every extra loss term) and
  was redundant for every current consumer — sd3_attack.py, flux_attack.py,
  clip_loss.py, and latent_loss.py all already cast their own input to
  their required working dtype internally — except the original SD1.5
  `attack.py` path, which is now given the same defensive cast so
  `sd_probability > 0` configs (train_1088_v3.yml) don't break. Mechanism:
  with img_adv in fp16, all six loss terms' backward gradients summed
  through one shared fp16 buffer, well within reach of fp16's ~65504 max;
  removing that downcast removes one accumulation point. A `scale==0.0`
  abort guard (mirroring the existing NaN/Inf guard) was also added since
  GradScaler has no floor and no log line for found_inf skips — the fix
  removes one fp16 accumulation point but gradients still traverse fp16
  inside each surrogate's own subgraph, so this is a backstop, not a
  guarantee against recurrence. Verified: `test_img_adv_stays_float32_...`
  and `test_scaler_collapse_aborts_and_reports` in test_training_smoke.py
  (CPU/MPS, no GPU). **Not yet verified**: that this actually prevents the
  collapse recurring on a real GPU run of the config that produced it — no
  pre-collapse `[Scaler]` log was available to confirm gradual-bleed vs.
  cliff, so the causal link to *this specific run's* collapse is inference
  from code, not a reproduced-and-fixed bug.

### Tier 3 — Unprovable from this repo (external/closed models)

- **nano-banana (Gemini image editing), DALL-E 3, Midjourney**: no
  gradient access exists or will exist. Protection claims can only rest on
  (a) surrogate transfer from the FLUX/SD3.5 ensemble, and (b) the CLIP-H
  proxy argument (shared semantic interface). Both are empirical transfer
  claims requiring black-box evaluation against the actual services, and
  transfer to closed models is historically much weaker than white-box
  results. Any protection-rate claim for these models without such an eval
  would be unsupported.

### Overall

The training METHOD is now verified sound: every loss term carries correctly
signed, nonzero gradient to the perturbation network through the real DiT
attack paths, and the loop demonstrably learns. Confidence that it "can
accurately learn a protective model" against white-box surrogates (FLUX.2,
SD3.5, SD1.5) is high pending one GPU validation run. Confidence for
closed-model transfer (nano-banana et al.) is necessarily lower and capped
until black-box evals are run — this is a property of the threat model, not
of the code.


---

## 2026-11-11 Update: the img2img surrogates were the wrong mechanism

**Observed (outputs/*/eval_report.json, 1088px, n=1-3):** `clip_delta` within
noise of zero and sign-flipping for SD3.5, FLUX.2 Klein and SDXL, for both Hub
checkpoints. No DiT protection has been demonstrated. Several runs with a low
edit SSIM also had a visible perturbation (PSNR 22-25 dB); with no noise control
that proves nothing.

**Mechanism (verified in the installed diffusers 0.39 source):** the editors
(Flux2Klein, and by construction Kontext / Qwen-Image-Edit) feed the CLEAN
reference as extra tokens at every denoising step and generate from pure noise.
The img2img surrogates noised the protected latent instead. The SD 1.5
inpainting result in the original paper uses the same clean-conditioning path.

**Status of earlier claims:** H1-H8 above were "implemented", never shown to
help. Of the loss terms, attention-entropy (saturates at ln N by construction),
latent cosine (0.97-0.99, no-op at weight 1.0), CLIP-H (224px downsample of a
1088px perturbation) and flat-minima (a gradient rescale) have no evidence of
doing what their docstrings claim; they are off in the new configs.

**New (CPU-verified only):** `FluxKleinEditAttack` (parity with the real
pipeline), reference hidden-state loss, latent L2, hard epsilon, controlled
evaluation protocol. **Tier 2 is still entirely open**: see
`research/gpu-validation-runbook.md`. **Tier 3 (Nano Banana etc.) is
unchanged: black-box measurement only.**
