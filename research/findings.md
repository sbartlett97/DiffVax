# DiffVax Extension — Research Findings

## Research Question
How to extend DiffVax image immunization from SD 1.5 (UNet-based, 512×512) to SOTA DiT/hybrid models (FLUX, SD 3.5, gpt-image-edit) and larger resolutions (1088×1088) for a product protecting social media uploads?

---

## Current Understanding

### The DiffVax System (Baseline)
- **Architecture**: NestedUNet (UNet++), ~9.2M params, fully convolutional
- **Training**: Backpropagates through SD 1.5 inpainting (4-step differentiable pass) to produce perturbation that makes the model output zeros (blank) in the masked region
- **Loss**: `L1(edited_output, zeros) + alpha * L1(perturbation, zeros)` — both normalized by `/512`
- **Resolution**: Fixed at 512×512 (hardcoded in loss normalization, VAE stride = 512/8 = 64)
- **Speed advantage**: Single-pass inference (milliseconds) vs PGD-per-image (hours)
- **Published at ICLR 2026**

### Gap Analysis
1. **Model coverage**: Only trained/evaluated against SD 1.5. FLUX, SD 3.5, gpt-image-edit are untested.
2. **Resolution**: 512×512 only. Social media uploads are typically 1080p+ (e.g., Instagram 1080×1080, Twitter 1200×675).
3. **FLUX attack wrapper**: `train_multimodel.yml` and `train_1024.yml` reference FLUX.2-Klein-4B but `attack.py` only implements `StableDiffusionInpaintPipeline`. A FLUX wrapper is needed.
4. **Loss resolution-dependence**: `/512` hardcoded — would produce wrong gradients at 1088.
5. **gpt-image-edit**: Black-box API. Cannot backpropagate through it. Must use transfer attack strategy.

### Architecture Differences: UNet vs DiT
| Property | SD 1.5 (UNet) | FLUX.2 (DiT) | SD 3.5 (MM-DiT) |
|---|---|---|---|
| VAE | 4-ch, scale=0.18215 | 16-ch, scale varies | 16-ch |
| Latent res (512px) | 64×64×4 | 64×64×16 | 64×64×16 |
| Conditioning | Cross-attn (CLIP) | Dual-stream (CLIP+T5) | Dual-stream |
| Inpainting | img+mask concat (9-ch) | Different conditioning | Different conditioning |
| Differentiable? | Yes, via diffusers | Yes, via diffusers | Yes, via diffusers |

Key insight: **All models share a VAE bottleneck** — the image must pass through `VAE.encode()` before anything architecture-specific happens. Perturbations effective at corrupting VAE-encoded features should transfer across model families.

---

## Patterns and Insights

*(To be filled as experiments run)*

---

## Lessons and Constraints

- **No local GPU**: All training experiments need cloud/cluster compute (Lambda Labs, Modal, etc.)
- **Loss normalization bug**: The `/512` in the loss must be replaced with `/resolution` for higher-res training
- **FLUX differentiable attack**: diffusers supports `FluxInpaintPipeline` — can implement a differentiable wrapper similar to `attack.py`
- **gpt-image-edit black-box constraint**: Transfer-only. Best strategy is to maximize protection against open-source models and rely on transfer

---

## Open Questions

1. Does immunization trained on SD 1.5 + FLUX transfer to SD 3.5 (untested)?
2. Is patch-based inference at 1088 sufficient, or does retraining at native resolution improve results?
3. What is the optimal mix of attack models during training for best cross-model generalization?
4. Can frequency-domain perturbation constraints improve imperceptibility at high resolution?
5. How does gpt-image-edit's resistance to transfer attacks compare to open-source models?
