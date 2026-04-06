# H2: Patch-Based Inference Enables 1088x1088 Immunization Without Retraining

## Hypothesis
The existing 512-trained NestedUNet can produce effective immunizations at 1088x1088 by
processing overlapping 512x512 patches with Gaussian-weighted blending, achieving
>=80% of the edit disruption rate compared to direct 512x512 immunization.

## Rationale
NestedUNet is fully convolutional — every layer is a convolution with no fixed spatial
dimensions. Given an arbitrary-sized input, it produces a matching-size output. The
512x512 constraint comes only from the training data and loss normalization, not the
architecture. Overlapping patch inference with smooth blending (as used in Stable Diffusion
upscaling and tiled VAE) should produce spatially coherent perturbations.

## Prediction
- Primary: patch_immunize at 1088×1088 achieves EDR >= 80% of direct 512×512 EDR on SD 1.5 evaluation
- Secondary: overlap >= 50% (stride 256) outperforms 25% overlap (stride 384) with <5% EDR gain
- Tertiary: immunized image PSNR vs original >= 38dB (imperceptibility maintained)

## Setup

### Implementation (DONE)
`src/diffvax/patch_immunize.py` — Gaussian-weighted patch blending at arbitrary resolution.

### Conditions
1. **Baseline**: Direct 512×512 immunization with existing checkpoint
2. **Exp-H2a**: patch_immunize at 1088×1088, stride=384 (75% overlap), patch=512
3. **Exp-H2b**: patch_immunize at 1088×1088, stride=256 (50% overlap), patch=512
4. **Exp-H2c**: patch_immunize at 1088×1088, stride=512 (no overlap), patch=512

### Evaluation
- 50 validation images upscaled to 1088×1088 (bicubic)
- Evaluate edit disruption with SD 1.5 inpainting on 1088×1088 images
- Also evaluate PSNR/SSIM of immunized vs original (imperceptibility)

### Metric pipeline
1. Resize val images to 1088×1088
2. Apply patch_immunize with each stride setting
3. Evaluate edit disruption: SSIM(edited_immunized) vs SSIM(edited_clean)
4. Report EDR, immunized PSNR, immunized SSIM

### Compute estimate
- No training needed — uses existing 512 checkpoint
- Evaluation: ~1h on GPU (no backprop, inference only)
- This is the lowest-cost experiment to run first

## Files
- `code/run_patch_eval.py` — evaluation script
- `results/patch_edr_metrics.csv`
- `analysis.md`

## Expected outcome
If H2 succeeds, we have an immediate path to 1088×1088 immunization for the product,
with zero additional training. If it fails (EDR < 80%), we proceed to H3 (fine-tuning).
