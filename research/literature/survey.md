# Literature Survey — DiffVax Extension

## Date: 2026-04-06

---

## Tier 1: Critical Papers (must read before running experiments)

### 1. "Purify Once, Edit Freely: Breaking Image Protections under Model Mismatch"
- **arXiv**: 2603.13028 (March 2025)
- **Key finding**: Protections optimized for surrogate models (SD 1.5) fail when adversaries use FLUX.1-fill-dev as a purifier/reconstructor. Improves PSNR by 3-6 dB and reduces FID by 50-70% versus protected images.
- **Why critical**: This is the ADVERSARY model — an attacker who wants to edit protected images can purify the immunization using FLUX. This directly motivates immunizing against FLUX: if we train against FLUX, the purifier attack no longer works.
- **Insight for H1**: Multi-model training is NOT just about forward-transfer; it's also about making purification harder. Training against FLUX forces our immunization to defeat FLUX-based purifiers.

### 2. "Universal Image Immunization Against Diffusion-based Image Editing via Semantic Injection"
- **arXiv**: 2602.14679 (February 2026)
- **Key finding**: First universal adversarial perturbation (UAP) framework for diffusion editing. Injects semantically misleading signals into cross-attention. Achieves strong black-box transferability across diffusion models WITHOUT requiring training data.
- **Why critical**: Demonstrates cross-model transfer IS achievable. The semantic injection approach targets cross-attention — which exists in both UNet (SD 1.5) and DiT (FLUX, SD3.5) architectures.
- **Insight for H1**: Cross-attention disruption as a perturbation target may generalise better than the current "output → zeros" approach.

### 3. "AdvPaint: Protecting Images from Inpainting Manipulation via Adversarial Attention Disruption"
- **arXiv**: 2503.10081 (March 2025)
- **Key finding**: Targets self- and cross-attention blocks specifically in FLUX-Fill inpainting. Disrupts semantic understanding and prompt interactions.
- **Why critical**: Explicitly addresses FLUX inpainting, confirming that FLUX requires dedicated attention-based attack strategies rather than pixel-space perturbations.
- **Insight for H4**: The VAE feature loss idea is good, but adding an attention disruption component may be even more powerful for DiT models.

---

## Tier 2: Highly Relevant Papers

### 4. "Demystifying Flux Architecture"
- **arXiv**: 2507.09595 (July 2025)
- **Key finding**: First comprehensive reverse-engineering of FLUX. MM-DiT blocks differ fundamentally from UNet — uses Rectified Flow, packed latent sequences, dual-stream attention.
- **Insight**: FLUX's packed latent format means attention patterns are spatially different from UNet. Our current differentiable forward pass in `attack_flux.py` correctly handles packing/unpacking.

### 5. "Fast High-Resolution Image Synthesis with LADD (Latent Adversarial Diffusion Distillation)"
- **SIGGRAPH Asia 2024**
- **Key finding**: Latent-space adversarial perturbations scale to megapixel resolution without expensive pixel decoding.
- **Insight for H3/H5**: Consider operating perturbations partially in latent space rather than purely in pixel space for better high-res scaling.

### 6. "Structure Disruption Attack (SDA): Safeguarding Image Regions Against Inpainting"
- **arXiv**: 2505.19425 (May 2025)
- **Key finding**: Optimizes perturbations by disrupting self-attention queries during initial denoising steps. More architecture-agnostic than full pipeline attacks.
- **Insight**: Initial denoising steps are architecture-agnostic (all models start with noisy latents) — disrupting at t_max timestep transfers better across architectures.

### 7. "Anti-Inpainting: Proactive Defense Against Malicious Diffusion-based Inpainters"
- **arXiv**: 2505.13023 (May 2025)
- **Key finding**: Multi-level deep feature extraction + multi-scale semantic-preserving augmentation. Better transferability through augmentation.
- **Insight**: Data augmentation (mask augmentation, resolution augmentation) during training improves transfer — DiffusionGuard already does mask augmentation; adding resolution augmentation may help.

### 8. "Evaluating Adversarial Protections for Diffusion Personalization: A Comprehensive Study"
- **arXiv**: 2507.03953 (July 2025)
- **Key finding**: Benchmarks 8 protection methods. No single method dominates. Different ε budgets favor different methods.
- **Insight for evaluation**: Use their methodology. Report EDR at multiple perturbation budgets (ε = 8/255, 16/255, 32/255).
- **Code**: https://github.com/vkeilo/DiffAdvPerturbationBench

### 9. "Diffusion Models for Imperceptible and Transferable Adversarial Attack"
- **arXiv**: 2305.08192
- **Key finding**: DDIM-inversion based perturbations using cross-attention maps. Demonstrates latent-space perturbations transfer better than pixel-space.
- **Insight**: Reinforces H4 hypothesis — VAE/latent-space loss terms should improve cross-model transfer.

### 10. "Prompt-Agnostic Adversarial Perturbation for Customized Diffusion Models"
- **NeurIPS 2024**
- **Key finding**: Perturbations must be prompt-agnostic. DiffVax already handles this via diverse prompt training, but DiT models with T5 encoders interpret prompts differently.
- **Insight**: During multi-model training, use diverse prompts per batch to ensure prompt-agnostic immunization for DiT text encoders.

---

## Tier 3: Background & Context

### 11. "PhotoGuard: Raising the Cost of Malicious AI-Powered Image Editing"
- ICML 2023, MIT CSAIL
- Foundational PGD-based immunization baselines (encoder attack, diffusion attack).

### 12. "Adversarial Perturbations Cannot Reliably Protect Visual Privacy"
- ICLR 2025
- Cautionary: immunization has fundamental limits. Important for honest product claims.

---

## Critical Gap Identified

**No papers specifically benchmark immunization across FLUX, SD 3.5, and gpt-image-edit simultaneously** (as of 2026-04-06). This is the primary research contribution of the DiffVax extension project.

---

## New Hypothesis Suggested by Literature (H6)

**H6: Multi-model immunization is purification-resistant**
- "Purify Once, Edit Freely" shows FLUX can purify SD1.5 immunizations.
- If we train DiffVax against FLUX, FLUX-based purification should fail.
- This is a strong product justification: our multi-model immunization resists the state-of-the-art purification attack.
- **Prediction**: FLUX-based purifier (EditorClean from 2603.13028) fails to recover editability after DiffVax-FLUX immunization.
