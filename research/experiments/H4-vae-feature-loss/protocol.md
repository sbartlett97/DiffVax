# H4: VAE Feature-Space Loss Improves Cross-Architecture Transfer

## Hypothesis
Adding a VAE feature-space loss term — maximizing ||VAE_shared(x+δ) - VAE_shared(x)||₂
where VAE_shared is a widely-used base VAE (e.g. stabilityai/sd-vae-ft-mse) — to the
DiffVax training objective improves cross-model transfer to unseen DiT architectures
by at least 15% absolute EDR over pixel-space loss alone.

## Rationale
The current DiffVax loss operates purely in pixel-space (L1 of model output vs zeros).
Many SOTA models (FLUX, SD 3.5, SDXL) use VAEs that are variants of, or fine-tuned
from, the original SD VAE. A perturbation that maximally corrupts the VAE-encoded
representation will disrupt ANY model that uses that (or a similar) VAE — regardless
of whether the downstream denoising network is a UNet or DiT.

The insight: the VAE is the universal attack surface. If we can corrupt the latent
representation, the downstream model (regardless of architecture) sees garbage and
produces bad outputs.

## Mathematical formulation
New loss: L = L_edit + α * L_perturb + β * L_vae_feature

L_vae_feature = -||VAE(x + δ) - VAE(x)||₂ / resolution²
(maximise distance in VAE latent space — negative because we gradient descent)

where VAE is frozen and shared across all attack models.

Note: β is a new hyperparameter. Start with β = 0.5.

## Prediction
- Primary: EDR on FLUX.1-schnell (unseen) improves by >=15% absolute vs H1a baseline
- Secondary: EDR on SD 3.5 (unseen) improves by >=10% absolute
- Tertiary: Imperceptibility (PSNR) maintained within 1dB of baseline

## Implementation Plan

### Changes to diffvax_immunization.py
Add `vae_feature_loss` computation using a frozen shared VAE.

```python
# In __init__
from diffusers import AutoencoderKL
self.shared_vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse"
).to("cuda").half()
for p in self.shared_vae.parameters():
    p.requires_grad = False

# In training loop
with torch.no_grad():
    orig_latents = self.shared_vae.encode(img_batch.half()).latent_dist.mean
imm_latents = self.shared_vae.encode(img_adv.half()).latent_dist.mean
loss_vae = -(imm_latents - orig_latents.detach()).pow(2).mean()  # maximise distance
loss = loss1 + loss2 + beta * loss_vae
```

## Setup

### Training
- Config: same as `train_multimodel.yml` but with `vae_feature_loss: true` and `beta: 0.5`
- Iterations: 10000 (same as H1a)

### Evaluation
- Same held-out models as H1: FLUX.1-schnell, SD 3.5 Medium
- Compare against H1a checkpoint (identical training setup minus the VAE loss)
- Use `eval_transfer.py` from H1

## Files
- `code/diffvax_immunization_h4.py` — modified training loop with VAE feature loss
- `results/`
- `analysis.md`
