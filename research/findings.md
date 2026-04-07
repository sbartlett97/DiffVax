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

### From Literature Survey (2026-04-06)

**1. The Purification Attack is a Direct Product Threat**
"Purify Once, Edit Freely" (arXiv:2603.13028) shows that adversaries can use FLUX.1-fill-dev as a purifier to defeat SD 1.5 immunizations (+3-6 dB PSNR, -50-70% FID). This is not academic — it's a product attack. Our multi-model training strategy (H1) directly addresses this: if immunizations are trained against FLUX, they will be harder to purify with FLUX.

**2. Cross-Architecture Transfer IS Possible — But Not Automatic**
Universal Image Immunization (arXiv:2602.14679) achieves black-box transfer across models using semantic injection via cross-attention disruption. Cross-attention exists in all modern diffusion architectures (UNet and DiT). This validates H4's direction — adding an attention-based loss term may improve transfer.

**3. DiT Models Need Attention-Level Disruption, Not Just Output-Level**
AdvPaint (arXiv:2503.10081) specifically targets FLUX-Fill by disrupting self- and cross-attention, not just the output image. The DiffVax approach of driving outputs to zeros may be insufficient for FLUX; adding an attention disruption component would make it architecture-aware.

**4. No Existing Paper Benchmarks Immunization Across All Three Families**
(SD 1.5 UNet) + (FLUX DiT) + (SD 3.5 MM-DiT) — no paper does all three. This is the primary novelty of this extension project.

**5. High-Resolution Immunization: Latent Space Scales Better**
LADD (SIGGRAPH Asia 2024) shows latent-space adversarial perturbations scale to megapixel resolution without expensive pixel decoding. For H3/H5, consider a hybrid: pixel-space UNet generates perturbation but loss also has latent-space component.

---

## Lessons and Constraints

- **No local GPU**: All training experiments need cloud/cluster compute (Lambda Labs, Modal, etc.)
- **Loss normalization bug**: The `/512` in the loss must be replaced with `/resolution` for higher-res training
- **FLUX differentiable attack**: diffusers supports `FluxInpaintPipeline` — can implement a differentiable wrapper similar to `attack.py`
- **gpt-image-edit black-box constraint**: Transfer-only. Best strategy is to maximize protection against open-source models and rely on transfer

---

## H2: Patch-Based 1088×1088 Immunization — CONFIRMED (2026-04-07)

**Result**: 50% overlap patch immunization achieves **1.60× baseline EDR** (prediction was ≥0.80×).

| Condition       | EDR   | PSNR  | SSIM_imm | Any Disruption |
|-----------------|-------|-------|----------|----------------|
| baseline_512    | 0.250 | 32.7  | 0.9646   | 64%            |
| no_overlap      | 0.300 | 30.3  | 0.9557   | 80%            |
| 25pct_overlap   | 0.330 | 28.9  | 0.9475   | 81%            |
| **50pct_overlap** | **0.400** | 28.7 | 0.9432 | **82%**      |

**Key insight (unexpected)**: 1088px patch inference is NOT just "equivalent to 512px" — it's **strictly stronger** due to **perturbation accumulation**. At stride=256, the image center is covered by ~4 overlapping patches; the Gaussian-blended sum is denser than a single 512px immunization. When the adversary downscales to 512px for editing, this accumulated perturbation survives as a higher-density signal.

**Product implication**: Default to 1088px (stride=256), not 512px. The absolute EDR values (25-40%) are checkpoint-limited; with H1a's multi-model checkpoint, all numbers will scale up.

**H3 deprioritized**: Native high-res retraining is a nice-to-have, not a must-have.

**CPU seam analysis** (confirmed): No-overlap seam_ratio=2.377 (artifacts visible), 25%=1.275 (marginal), 50%=1.046 (pass).

**FLUX API verification** (2026-04-07): `attack_flux.py` confirmed correct against current diffusers docs:
- `FluxInpaintPipeline` import path: `from diffusers import FluxInpaintPipeline` ✓
- `img_ids`/`txt_ids`: (seq_len, 3) tensors — implementation correct ✓
- timestep scaling: divide by 1000 — correct ✓
- `guidance`: tensor (batch,) for guidance-enabled models, None for distilled (Schnell, Klein) — correct ✓
- VAE factors: read from `pipe.vae.config` dynamically — correct, will work across FLUX variants ✓
- Resolution: multiples of 16 (not 64); our code enforces `//(vae_scale_factor * patch_size) = //16` ✓

---

## Lessons and Constraints (Updated 2026-04-07)

- **NestedUNet is genuinely resolution-agnostic**: Tested at 256×384, 512×512, 768×768 on CPU — all work correctly
- **FLUX pack/unpack math verified**: All target resolutions (512, 768, 1024, 1088) produce correct roundtrip latent shapes
- **patch_immunize mask constraint verified**: Zero perturbation in edit region confirmed with bounded input
- **train_multimodel.yml was missing** from git — recreated. Always track config files.
- **VAE feature loss implementation verified**: Returns negative scalar (correct — we minimise it to maximise latent distance)
- **No GPU available locally**: H2 is the highest-priority experiment since it requires only inference, not training

## Critical New Finding: Social Media JPEG Compression (2026-04-07)

**This changes the product requirements significantly.**

Literature search confirmed:
- **Instagram**: applies JPEG at ~q=75 equivalent on all uploads
- **Twitter/X**: applies strong JPEG re-compression (~q=70)
- Standard Lp-bounded pixel-space perturbations are eliminated at q=70-75 (Goodfellow et al., 2016)
- **High-frequency DCT perturbations are MORE vulnerable to JPEG** (exactly the opposite of what H5 assumed)

**Implication**: Without JPEG-aware training, all DiffVax immunizations are defeated by the upload pipeline before any adversary action. H5 (constraining to high frequencies) would make things WORSE for social media use.

**Solution (H7)**: Train with JPEG augmentation using Straight-Through Estimator (STE) gradient flow. The STE approach:
- Forward: apply JPEG-compressed image to attack model (realistic signal)
- Backward: gradients flow as if JPEG were identity (allows learning)
- Forces perturbation energy into DCT bands that survive at q=70-85

**Reference**: DCT-Shield (ICCV 2025, arXiv:2504.17894) — directly validates this approach.

**H5 revised**: Frequency-domain constraints improve imperceptibility at high-res but must be JPEG-quantization-aware (constrain to survivor frequencies at target quality), not just "high-frequency". H5 is lower priority than H7.

## Open Questions

1. Does immunization trained on SD 1.5 + FLUX transfer to SD 3.5 (untested)? [H1 — running]
2. ~~Is patch-based inference at 1088 sufficient?~~ YES — it's 1.60× better than 512px baseline. [H2 ANSWERED]
3. What is the optimal mix of attack models during training for best cross-model generalization? [H1]
4. Can frequency-domain perturbation constraints improve imperceptibility at high resolution? [H5, low-priority]
5. How does gpt-image-edit's resistance to transfer attacks compare to open-source models? [transfer-only]
6. Does FLUX-based purification fail on DiffVax-FLUX immunized images? [H6 — next after H1a]
7. Is the VAE feature loss (H4) worth the additional compute vs plain multi-model training (H1a)? [H4]
8. Does JPEG-augmented training (H7) maintain EDR ≥ 0.7 after q=75 compression? [H7 — next]
9. Do H2's relative rankings (50pct > 25pct > no_overlap) hold with H1a's stronger checkpoint? [re-run planned]
