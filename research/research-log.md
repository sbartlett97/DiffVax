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
