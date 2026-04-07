# IDProtector: Adversarial Noise Encoder for Portrait Protection

**arXiv:** 2412.11638  
**Year:** December 2024  
**Relevance to DiffVax Extension:** MEDIUM — similar feed-forward immunization architecture, relevant to H7 JPEG robustness claim

## Main Contribution

Feed-forward adversarial noise encoder that protects portrait photos from unauthorized identity-preserving generation (InstantID, IP-Adapter, PhotoMaker). Single forward pass, similar to DiffVax's architecture for speed.

## JPEG Robustness — Key Details

IDProtector explicitly AVOIDS STE/differentiable JPEG training:
> "differentiable JPEG and cropping during training introduces substantial learning burden"

Instead, uses Gaussian noise augmentation (σ=0.003) as a **proxy** for JPEG and affine transforms:
- Only tested at q=85 (not q=70 or q=75)
- Indirect approach via affine noise, not actual JPEG gradient

**Table 6**: At JPEG quality=85, ISM values are maintained comparable to non-distorted protected images.

## What This Means for H7

Our H7 directly addresses the gap IDProtector avoided:
1. **We DO use STE JPEG augmentation** (not Gaussian proxy)
2. **We target q=70-75** (Instagram/Twitter) — significantly harder than q=85
3. H7 novelty: first feed-forward immunization with explicit STE JPEG training at social media quality levels

The IDProtector authors' concern about "substantial learning burden" can be addressed by our STE approach, which keeps the backward pass as identity — the JPEG operation only affects the forward signal, not gradient complexity.

## Relationship to Our Work

- Same paradigm: fast feed-forward encoder for immunization
- Different task: identity protection vs. general inpainting
- Different robustness: q=85 vs. q=70-75
- Our technical advance: STE JPEG over Gaussian proxy
