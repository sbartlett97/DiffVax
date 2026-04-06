# H6: Multi-Model Immunization Resists FLUX-Based Purification Attacks

## Hypothesis
DiffVax immunizations trained against FLUX will resist FLUX-based purification
attacks (EditorClean from arXiv:2603.13028) better than SD1.5-only immunizations,
maintaining edit disruption even after the adversary attempts to purify the image.

## Background — The Threat Model
"Purify Once, Edit Freely" (arXiv:2603.13028, March 2025) shows a realistic attack:
1. User publishes immunized image (DiffVax protected)
2. Adversary runs FLUX.1-fill-dev on the protected image to "purify" it
3. The purified image has PSNR +3-6 dB higher and FID -50-70% lower vs naive edit
4. The adversary can now edit the purified image freely

This is a product-level threat. The counter: if our immunization was trained AGAINST
FLUX, then using FLUX to purify will itself fail (the immunization disrupts FLUX).

## Prediction
- Primary: Edit disruption rate after FLUX-purification attempt on DiffVax-FLUX
  immunized images is ≥70%, vs <40% for DiffVax-SD15-only immunized images.
- Secondary: The DiffVax-FLUX model makes FLUX-based purification produce
  visibly degraded outputs (corrupted textures, wrong colors).

## Setup

### Conditions to compare
1. **Baseline**: DiffVax-SD15 (existing 512 checkpoint) → FLUX purification → FLUX edit
2. **H6 exp**: DiffVax-FLUX (H1a checkpoint) → FLUX purification → FLUX edit
3. **Upper bound**: DiffVax-FLUX → direct FLUX edit (no purification)

### Purification procedure (from arXiv:2603.13028)
1. Run FLUX.1-fill-dev on immunized image with empty mask (reconstruct whole image)
2. Use output as "purified" image
3. Run editing attack on purified image

### Evaluation
- EDR on the purified+edited image vs original
- PSNR of purified vs immunized (did purification succeed perceptually?)
- 50 validation images × 3 prompts

## Why This Matters for the Product
This experiment provides the key product claim: "Our immunization resists even
AI-powered de-protection tools." This is a significant differentiator vs existing
tools like Glaze, PhotoGuard, which all fail against FLUX purification.

## Files
- `code/eval_purification_robustness.py`
- `results/`
- `analysis.md`
