# Anti-Inpainting: Proactive Defense Against Malicious Diffusion-based Inpainters

**arXiv:** 2505.13023  
**Year:** May 2025  
**Relevance to DiffVax Extension:** HIGH — cross-model transfer via augmentation, directly relevant to H1

## Main Contribution

Proactive defense against inpainting manipulation under UNKNOWN conditions (unknown mask, unknown model, unknown prompt). Key techniques:
1. **Multi-scale, semantic-preserving data augmentation** to enhance transferability across unknown model versions
2. **Selection-based distribution deviation optimization** to improve protection against diverse random seeds

## Key Technique: Multi-Scale Augmentation for Transfer

"A multi-scale, semantic-preserving data augmentation technique to enhance the transferability of adversarial perturbations across unknown conditions."

This is directly relevant to our H1 multi-model training: the idea that training on multiple conditions forces perturbations to transfer across unseen conditions. Their augmentation strategy may be more principled than our random routing.

## Evaluation

- Benchmarks: InpaintGuardBench, CelebA-HQ
- Claims: "transferability across different diffusion model versions"
- Does NOT mention JPEG robustness in abstract

## What We Don't Know (from abstract only)

- Which specific models tested (SD 1.5? FLUX? SD 3.5?)
- Whether it evaluates JPEG compression robustness
- Comparison to DiffVax

## Implications for Our Work

If Anti-Inpainting achieves strong cross-model transfer with augmentation, this validates our H1 direction. Key question: do they test on FLUX/DiT architectures or only different SD versions? If SD versions only, our work (SD + FLUX + SD 3.5) is more ambitious.
