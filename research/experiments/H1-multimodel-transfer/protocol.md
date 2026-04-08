# H1: Multi-Model Training Transfers to Unseen DiT Architectures

## Hypothesis
Training DiffVax against both SD 1.5 (UNet) and FLUX.1-schnell (DiT) produces
a model whose immunizations transfer to SD 3.5 and FLUX.1-schnell (unseen at training),
achieving >80% of the edit disruption rate seen on the trained models.

## Rationale
All diffusion models share a VAE bottleneck — the image must be encoded to latents
before any architecture-specific denoising. Perturbations optimised to survive the
encoder and disrupt the latent denoising process may generalize across model families
that use similar (or the same) VAE. FLUX uses a 16-channel VAE; SD 1.5 uses a 4-channel
VAE. However, the pixel-space perturbations produced by DiffVax are independent of the
downstream model's latent dimensionality.

## Prediction
- Primary: EDR on held-out SD 3.5 >= 50% of EDR on trained models (SD 1.5 + FLUX.1-schnell)
- Secondary: Transfer gap to SD 3.5 is smaller when trained on SD+FLUX jointly vs SD alone

## Setup

### Conditions
1. **Baseline**: DiffVax trained on SD 1.5 only (existing checkpoint)
2. **Exp-H1a**: DiffVax trained on SD 1.5 (20%) + FLUX.1-schnell (80%) — `configs/train_multimodel.yml`
3. **Exp-H1b**: DiffVax trained on SD 1.5 (20%) + FLUX.1-schnell (60%) + SD 3.5 (20%)

### Evaluation models (held out during training)
- FLUX.1-schnell (Black Forest Labs)
- SD 3.5 Medium (stabilityai)
- gpt-image-edit (black-box API, qualitative only)

### Metric: Edit Disruption Rate
```
EDR = fraction of test images where SSIM(edited_immunized, original) < SSIM(edited_clean, original) + 0.05
```
Higher is better (immunized edits are disrupted).

### Dataset
- 50 validation images from DiffVaxDataset (held out during training)
- 5 prompts per image × 3 seed runs = 750 edit attempts per model

### Compute estimate
- H1a training: ~8h on A100 80GB (10k iterations, multi-model)
- Evaluation: ~2h per model per 50 images

## Files
- `code/train_h1a.sh` — launch script for H1a
- `code/eval_transfer.py` — evaluation script
- `results/` — metric CSVs
- `analysis.md` — interpretation

## Pre-registration commit
This protocol must be git-committed BEFORE running any experiments.
