# DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing

**arXiv:** 2504.17894  
**Venue:** ICCV 2025  
**Relevance to DiffVax Extension:** HIGH — directly addresses JPEG-robust frequency-domain immunization

## Key Findings

- Designs adversarial perturbations that operate in the DCT domain with JPEG-quantization-aware constraints
- Incorporates the JPEG pipeline (DCT → quantization → dequantization → iDCT) into the perturbation optimization
- Result: perturbations that survive JPEG re-compression at typical social media quality settings
- Shows that naive Lp-bounded pixel-space perturbations are wiped out by JPEG at q=70-75
- Shows that naively constraining to high-frequency DCT bands also fails (JPEG aggressively quantizes high frequencies)
- The sweet spot: **DCT coefficients in mid-frequency bands that fall in the JPEG quantization-survivor region** for target quality q

## Architecture

1. Compute JPEG quantization table for target quality q (standard luminance/chrominance tables scaled by q)
2. Identify per-frequency-band "survivor coefficients" — those not zeroed out at quality q
3. Constrain perturbation energy to only these bands during optimization
4. Forward pass through JPEG simulation is differentiable via DCT/IDCT (differentiable)
5. Quantization step uses straight-through estimator for gradients

## Relation to H5 and H7

- **H5 revised**: Frequency constraints help imperceptibility at high resolution BUT must be JPEG-quantization-aware, not just "high-frequency"
- **H7 new**: Implement JPEG-quantization-aware training (augment with JPEG in forward pass, use STE for gradients)

## Compression Data

JPEG quantization table for q=75 zeroes out DCT coefficients at the following frequencies in an 8x8 block:
- Frequencies (u,v) with quantization step > epsilon — these are wiped out
- Low frequencies (0,0)-(2,2): quantization step 2-8 → survives at q=75
- Mid frequencies (2,2)-(5,5): quantization step 8-32 → partially survives
- High frequencies (5,5)+(: quantization step 32+ → wiped out at q=75

## Actionable Insight for DiffVax

Training with JPEG augmentation (using STE to allow gradient flow) forces immunization to occupy JPEG-survivor frequencies. This is more practical than exact DCT-Shield optimization but should capture most of the benefit.

**Implementation**: In `DiffVaxImmunization.train()`:
- Apply random JPEG (q=70-85) to immunized image before attack model
- Use STE: `img_for_attack = img_immunized_jpeg + (img_immunized - img_immunized).detach()`
  (Forward: JPEG-compressed; Backward: straight-through to img_immunized)
