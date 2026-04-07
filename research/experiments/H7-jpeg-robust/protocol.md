# H7 Protocol: JPEG-Robust Immunization for Social Media

## Hypothesis

DiffVax trained with JPEG augmentation (q=70-85, STE gradient) produces immunizations that survive Instagram/Twitter upload compression, while standard DiffVax is defeated by JPEG at q=70-75.

## Background and Motivation

**New finding (2026-04-07):** Literature search revealed that:
- Instagram applies JPEG at ~q=75 equivalent on all uploads
- Twitter/X applies strong JPEG re-compression (~q=70)
- Standard Lp-bounded pixel-space perturbations are wiped out at q=70-75 (Goodfellow et al., 2016)
- High-frequency DCT perturbations are MORE vulnerable to JPEG (DCT-Shield, ICCV 2025)

**Implication:** Without JPEG-aware training, ALL DiffVax immunizations are likely defeated by the upload pipeline before the adversary even tries to edit.

## Method

1. Train DiffVax with JPEG augmentation: with probability p=0.5 per training step, apply JPEG compression (q=70-85, uniformly sampled) to `img_immunized` before passing to the attack model
2. Gradient flow uses Straight-Through Estimator (STE): forward=JPEG-compressed, backward=identity
3. Compare vs H1a (no JPEG aug) baseline on both:
   a. Clean evaluation (no JPEG applied to test images)
   b. JPEG evaluation (apply q=75 to immunized image before editing — simulates Instagram upload)

## Prediction

- H1a (no JPEG aug): EDR drops to near-chance (~0.5) when q=75 JPEG applied to immunized image
- H7 (JPEG aug): EDR maintains ≥0.7 even after q=75 JPEG compression

## Evaluation Protocol

For both H7 and H1a checkpoints:
1. **Clean EDR**: standard eval without JPEG
2. **Post-JPEG EDR**: apply q=75 JPEG to immunized image before editing
3. **Post-JPEG EDR at q=70**: apply q=70 (Twitter/X scenario)
4. **PSNR/SSIM**: imperceptibility of immunized image (verify perturbation is still invisible)

## Config

`configs/train_multimodel_h7.yml`
- SD 25% + FLUX 75% (same as H1a)
- `jpeg_augment_prob: 0.5`
- `jpeg_quality_range: [70, 85]`

## Commit Protocol

- Protocol commit: `research(protocol): H7 JPEG-robust immunization for social media`
- Results commit: `research(results): H7 — [summary]`
