# DiffVax Research Findings

**Last updated:** 2026-03-27
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
