# Literature Survey: Image Immunization & Adversarial Perturbation Against Diffusion Models
**Date:** 2026-03-26 | **Papers surveyed:** 26

---

## Critical Papers for DiffVax High-Resolution Research

### 1. PhotoGuard — ICML 2023 (arXiv:2302.06588)
**Salman et al. (MIT CSAIL)**
- Encoder attack: drive `VAE.encode(x+delta)` to random target latent → model treats image as uninterpretable
- Diffusion attack: full pipeline backprop with gradient averaging over 10 denoising samples
- **Key insight**: Encoder-space disruption generalizes across editing prompts because VAE bottleneck is shared
- **DiffVax baseline**: Implemented in `/src/diffvax/immunization/photoguard_immunization.py`

### 2. DeContext (arXiv:2512.16625) — December 2024
**Shen, Cui, Yang**
- **Most relevant paper for DiT attack effectiveness**
- Contextual signal in DiT models propagates primarily through **multimodal attention layers**, NOT through VAE encoding
- **Early denoising timesteps (t > 0.7) and MIDDLE transformer blocks** are disproportionately responsible for context propagation
- Prior encoder attacks (PhotoGuard-style) lose significant effectiveness on DiT models
- → **DiffVax's Phase 7 currently hooks "early" blocks — should hook MIDDLE blocks for DiT**

### 3. TGR: Token Gradient Regularization — CVPR 2023 (arXiv:2303.15754)
**Zhang et al. (CUHK)**
- High token-to-token gradient variance within ViT blocks causes poor adversarial transfer
- Token-wise gradient normalization during backprop → smoother, more generalizable perturbation direction
- 8.8% improvement in adversarial transfer to unseen ViT architectures
- **Directly addresses DiffVax's gradient instability at 1088px through SD3.5's 18k-token attention**

### 4. DDAP: Dual-Domain Anti-Personalization (arXiv:2407.20141) — CVPR 2024
**Yang et al. (Fudan)**
- Alternates spatial-domain (encoder attack) and frequency-domain (DCT high-frequency targeting)
- **High-frequency perturbation concentration improves imperceptibility at same L-inf budget**
- SSIM > 0.98 while maintaining strong protection — relevant for 1088px imperceptibility

### 5. Distraction Is All You Need — CVPR 2024
**Lo et al. (NYCU)**
- Memory-efficient gradient: compute only at **subset of critical timesteps** → 50% VRAM reduction with equivalent protection
- Cross-attention spatial targeting is more effective than encoder disruption for SD-based models
- **Critical for 1088px VRAM budget: selective timestep gradient would bring SD3.5 under 24GB**

### 6. Mist (arXiv:2305.12683) — 2023
**Liang & Wu**
- Dual loss: semantic loss (maximize LDM denoising error) + textural loss (maximize VAE encoder distance)
- **Target image selection matters**: sharp-edges/high-contrast targets >> smooth targets for textural loss
- Key insight for DiffVax: current loss1 pushes to zero (smooth black) — not optimal

### 7. Hönig et al. — ICLR 2025 (robust mimicry)
- Pixel-space perturbations defeated by averaging 50+ protected images
- Content-conditioned perturbations (DiffVax's NestedUNet) may be more resistant than UAP methods
- **DiffVax must acknowledge this threat model and scope protection claims accordingly**

### 8. AdvDM — ICML 2023 (arXiv:2302.04578)
**Liang et al.**
- Monte Carlo timestep averaging produces smoother gradient signal than single-timestep
- Equivalent to EoT-over-timesteps used in DiffVax training

### 9. MetaCloak — CVPR 2024 Oral (arXiv:2311.13127)
**Liu et al.**
- MAML meta-learning for perturbation + EoT transforms → robust to JPEG/blur purification
- Current state-of-the-art benchmark for EoT-robust protection DiffVax must meet

### 10. CLIP-based Transfer (Glaze + DTIA + NL Adversarial)
- CLIP embedding disruption is model-agnostic and transfers to DALL-E 3 (CLIP-conditioned)
- **DiffVax v2 has CLIP ViT-B/32 — upgrading to CLIP ViT-H/14 would improve DALL-E 3 coverage**
- Glaze arXiv:2302.04222; DTIA DOI:10.1007/s41019-024-00272-9; NL arXiv:2410.08620

### 11. Immunizing via Adversarial Cross-Attention — ACM MM 2025 (arXiv:2509.10359)
**Trippodo et al.**
- Caption-proxy for prompt-agnostic attacks on cross-attention
- Middle transformer blocks contain most transferable attention disruption signal

### 12. Universal Image Immunization (arXiv:2602.14679) — 2025
**Lee et al. (GIST)**
- Universal (content-agnostic) UAP for diffusion editing protection
- Comparison point: UAP vs DiffVax's content-conditioned approach

---

## Key Synthesis for DiffVax v3 Architecture

### For Scaling to 1088px

| Problem | Literature Solution | Implementation |
|---------|--------------------|--------------:|
| Gradient instability in SD3.5 18k-token attention | TGR token-wise gradient normalization | Implement in `sd3_attack.py` backward hook |
| VRAM bottleneck at 1088px (~26-28GB) | Partial timestep gradient (Distraction CVPR2024) | Select top-K timesteps for backprop |
| Imperceptibility degradation at high-res | High-frequency concentration (DDAP FPL) | Spectral penalty in loss2 |
| EoT OOM at 1088px with resize_range 2.0x | Reduce EoT resize to 1.0-1.5x at 1088px stage | Config change in curriculum |

### For Disrupting Transformer-Based DiT Models

| Problem | Literature Solution | Implementation |
|---------|--------------------|--------------:|
| Phase 7 hooks early blocks (wrong) | DeContext: hook MIDDLE blocks | Change `target_blocks: "middle"` in config |
| DALL-E 3 not covered | CLIP-H/14 feature disruption | Add CLIP-H loss term |
| Prompt-agnosticism weaker in DiT | PAP Laplace distribution + diverse training | Already partially addressed by multi-prompt training |
| Black-image target suboptimal for DiT | High-contrast/noise target (Mist insight) | Change target in loss computation |

---

## Papers by arXiv ID for Full Bibliography

```
2302.06588 PhotoGuard (ICML 2023)
2302.04578 AdvDM (ICML 2023)
2303.15433 Anti-DreamBooth (ICCV 2023)
2302.04222 Glaze (USENIX Sec 2023)
2305.12683 Mist (arXiv 2023)
2311.13127 MetaCloak (CVPR 2024 Oral)
2310.04687 ACE / Targeted Attack (NeurIPS 2024)
2408.10571 PAP: Prompt-Agnostic (NeurIPS 2024)
2407.20141 DDAP: Dual-Domain (arXiv 2024)
2303.15754 TGR: Token Gradient Regularization (CVPR 2023)
2512.16625 DeContext as Defense (arXiv Dec 2024)
2509.10359 Immunizing via Cross-Attention (ACM MM 2025)
2412.11638 IDProtector high-res (arXiv 2024)
2512.14320 Semantic mismatch evaluation (arXiv 2024)
2410.08620 NL Induced Adversarial (arXiv 2024)
1707.07397 EoT: Synthesizing Robust Adversarial Examples (ICML 2018)
2311.18815 IMMA model-side protection (ECCV 2024)
2602.14679 Universal Image Immunization (arXiv 2025)
2304.04386 Latent-space adversarial attacks (arXiv 2023)
```
