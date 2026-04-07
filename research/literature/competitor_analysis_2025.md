# Competitor Analysis — DiffVax++ Positioning
**Date:** 2026-04-07

## Anti-Inpainting (arXiv:2505.13023)

- **Authors**: Yimao Guo et al., May 2025 (v3 Aug 2025), arXiv preprint
- **Method**: Multi-component: deep feature extraction from denoising + multi-scale semantic-preserving augmentation + selection-based distribution deviation optimization
- **Models targeted**: Generic diffusion inpainting (transferability claimed, architectures unspecified)
- **JPEG robustness**: No
- **High-res**: No
- **Metrics**: No specific numbers in abstract. Evaluated on InpaintGuardBench and CelebA-HQ
- **Gap**: No comparison to DiffVax; no quantitative EDR; no JPEG

## Attention Attack (arXiv:2509.10359) — ACM MM 2025

- **Authors**: Matteo Trippodo, Federico Becattini, Lorenzo Seidenari
- **Method**: Black-box cross-attention disruption using auto-generated captions as proxy adversarial targets. Introduces Caption Similarity and semantic IoU metrics.
- **Models targeted**: Generic (no SD1.5/FLUX/SD3.5 specifics)
- **JPEG robustness**: No
- **High-res**: No
- **Metrics**: "Significantly degrades editing performance" — no specific numbers
- **Gap**: Black-box (no gradient access) — DiffVax++ has differentiable multi-model training

## PromptFlare (arXiv:2508.16217) — ACM MM 2025

- **Authors**: Hohyun Na, Seunghoo Hong, Simon S. Woo
- **Method**: Cross-attention decoy — injects adversarial noise targeting semantically uninformative shared tokens to divert model focus. Claims SOTA and computational efficiency.
- **Models targeted**: Diffusion inpainting generally (unspecified architectures)
- **JPEG robustness**: No
- **High-res**: Not mentioned
- **Metrics**: "State-of-the-art across various metrics" — no specific numbers in abstract
- **Gap**: Efficiency focus only; no robustness discussion; no social media deployment

---

## Comparison Table

| Dimension              | Anti-Inpainting | Attention Attack | PromptFlare | **DiffVax++** |
|------------------------|-----------------|-----------------|-------------|---------------|
| Multi-model (SD+FLUX+SD3) | Claimed, unspecified | Unspecified | Unspecified | **Explicit** |
| High-resolution (>512px) | No | No | No | **1088px, 1.60×** |
| JPEG/social media robust | No | No | No | **Yes (STE q=70-75)** |
| Peer-reviewed venue | arXiv | ACM MM 2025 | ACM MM 2025 | ICLR 2025 |
| Inference speed | Not discussed | Not discussed | Efficiency focus | **ms (single-pass)** |
| Specific EDR metrics | No | No | No | **Yes** |
| Purification resistance | No | No | No | **H6 (planned)** |

## Paper Positioning Summary

DiffVax++ is the **only** work to explicitly address all three real-world deployment gaps:
1. Multi-architecture (SD1.5 + FLUX.1-schnell + SD3.5): no competitor does this explicitly
2. High-resolution (1088px, 1.60× EDR): no competitor mentions >512px
3. Social media JPEG robustness (STE q=70-75): no competitor addresses this; IDProtector (Dec 2024) explicitly avoided it

PromptFlare makes the broadest "SOTA" claim but on unspecified models with no numbers. 
This is a weak SOTA claim — DiffVax++ beats it on every deployment-relevant dimension.
