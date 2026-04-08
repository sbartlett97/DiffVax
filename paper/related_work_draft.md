# Paper Draft — Related Work
# Status: draft
# Date: 2026-04-08

---

## 2. Related Work

### 2.1 Image Immunization Against Diffusion Editing

**Optimization-based methods.** PhotoGuard [SALMAN2023] and Glaze [SHAN2023] apply
per-image PGD optimization to produce adversarial perturbations that disrupt style
mimicry and identity-consistent generation, respectively. While effective, these
methods require minutes to hours per image, precluding social media scale deployment.
Anti-DreamBooth [VANLE2023] targets personalization rather than inpainting. All
optimization-based methods are also implicitly single-model: they optimize against a
fixed surrogate model and do not train for transfer to unseen architectures.

**Amortized methods.** DiffVax [OZDENTARIKCAN2025] addresses the speed bottleneck by
training a NestedUNet to produce perturbations via a single forward pass. Trained by
backpropagating through a differentiable 4-step SD 1.5 inpainting pass, DiffVax
achieves millisecond-speed immunization at inference, making platform-scale deployment
feasible. Our work extends DiffVax along the three dimensions described above.

**Cross-model transfer.** Universal Image Immunization [ZHONG2026] achieves black-box
transfer across diffusion models using cross-attention disruption without surrogate
model training. This validates that cross-attention is an effective disruption target
across UNet and DiT architectures. Anti-Inpainting [GUO2025] uses multi-scale
augmentation to improve transfer under varying mask and prompt conditions.
Attention Attack [TRIPPODO2025] disrupts cross-attention using auto-generated image
captions as proxy adversarial targets (ACM MM 2025). PromptFlare [NA2025] injects
adversarial noise into semantically uninformative prompt tokens (ACM MM 2025), claiming
"state-of-the-art" performance — but without specifying which model architectures were
tested or providing concrete numerical metrics. Concurrent work AEGIS [LI2026] confirms
that trajectory-aware latent-space injection across multiple denoising steps is more
robust than single-step injection. Critically, *none* of these methods address JPEG
compression robustness or evaluate on more than one model architecture.

### 2.2 Purification Attacks on Immunized Images

The security of immunization methods is increasingly challenged by purification
attacks, which attempt to remove adversarial perturbations before editing.
"Purify Once, Edit Freely" [ZHAO2026] shows that FLUX.1-fill-dev can reconstruct
SD1.5-immunized images with +3–6 dB PSNR improvement and 50–70% FID reduction,
effectively removing the immunization and restoring full editability. Pleimling et al.
[PLEIMLING2026] extend this to commodity image-to-image tools — standard SR and style
transfer models can strip Lp-bounded perturbations across six defense schemes with no
knowledge of the specific defense. These findings establish that any immunization
method targeting only a single model family, or relying solely on pixel-space Lp
constraints, will be practically ineffective against a capable adversary.

DiffVax++ directly addresses both threats: multi-model training (H1) makes the
perturbation adversarially robust against the specific purifier architecture (FLUX),
while JPEG augmentation (H7) forces energy into frequency bands that commodity tools
cannot easily remove without visible image quality loss.

### 2.3 High-Resolution Adversarial Perturbations

Extending adversarial perturbations to high resolution has been studied in the
recognition domain [XIAO2018,SHARIF2019] but is largely unexplored for diffusion
immunization. LADD [CITE-LADD] shows that latent-space perturbations scale to
megapixel resolution. PixelRush [CITE-PIXELRUSH2026] demonstrates that patch-based
4K inference is now standard for modern diffusion models. Our work is the first to
study high-resolution immunization specifically, and the counter-intuitive finding
that overlapping patch inference *strengthens* protection (via perturbation accumulation)
is, to our knowledge, entirely novel.

### 2.4 Compression-Robust Adversarial Examples

Standard Lp-bounded adversarial perturbations are known to be fragile under JPEG
compression: Goodfellow et al. [GOODFELLOW2016] first noted this; subsequent work
confirmed that high-frequency perturbations are disproportionately destroyed by DCT
quantization. DCT-Shield [ICCV-2025] provides a comprehensive analysis of which
frequency bands survive at various JPEG quality settings.

For immunization specifically, DiffVax [OZDENTARIKCAN2025] evaluates JPEG as an
adversarial counter-attack but does *not* train for compression robustness. IDProtector
[CHEN2024] explicitly avoids STE-based JPEG training ("introduces substantial learning
burden") and uses a Gaussian noise proxy evaluated only at q=85 — far above Instagram
(q≈75) and Twitter (q≈70) quality settings. The Straight-Through Estimator [BENGIO2013]
has been applied to quantization-aware training [HUBARA2016] but to our knowledge has
not been applied to immunization training targeting social media compression. DiffVax++
is the first to close this gap, and H7 is the first immunization method explicitly
designed for the Instagram/Twitter deployment scenario.

### 2.5 Comparison Summary

Table 1 summarizes the coverage of existing methods across the three deployment
dimensions addressed by DiffVax++.

| Method | Multi-arch | High-res | JPEG robust | Numeric EDR? | Venue |
|---|---|---|---|---|---|
| DiffVax [OZDENTARIKCAN2025] | ✗ | ✗ | ✗ | ✓ (EDR=0.25) | ICLR 2025 |
| Anti-Inpainting [GUO2025] | ✗ (claimed, unspecified) | ✗ | ✗ | ✗ (qualitative) | arXiv 2025 |
| Attention Attack [TRIPPODO2025] | ✗ | ✗ | ✗ | ✗ (custom metrics) | ACM MM 2025 |
| PromptFlare [NA2025] | ✗ | ✗ | ✗ | ✗ (SOTA claimed, no #s) | ACM MM 2025 |
| IDProtector [CHEN2024] | ✗ | ✗ | q=85 only | ✗ (SSIM only) | Dec 2024 |
| AEGIS [LI2026] | ✗ | ✗ | ✗ | ✗ (qualitative) | Apr 2026 |
| **DiffVax++ (ours)** | **✓ SD+FLUX+SD3** | **✓ 1088px** | **✓ q=70–75** | **✓ (EDR=0.40, 1.60×)** | — |

*No paper in 2024–2026 other than DiffVax reports a standardized numerical disruption metric on a shared public benchmark. The "SOTA" claim by PromptFlare [NA2025] is qualitative and not reproducible from their paper.*
