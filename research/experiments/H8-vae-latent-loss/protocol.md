# H8: VAE Latent-Space Disruption Loss

**Status:** Protocol locked  
**Date locked:** 2026-03-27

## What

Add a VAE latent-space disruption loss term that maximises distance between
`vae.encode(img_adv).latent_dist.mode()` and `vae.encode(img_orig).latent_dist.mode()`
in 16/4-channel latent space.

Changes:
- `BaseAttack.get_vae()` — returns None by default
- `SD3Attack.get_vae()`, `FluxAttack.get_vae()`, `Attack.get_vae()` — overrides
- Training loop: compute cosine-distance loss in latent space when `latent_loss.enabled`
- Config key: `latent_loss.enabled: true`, `latent_loss.weight: 1.0`

## Why

All current losses operate in pixel space. For DiT models, generation is governed by
the 16-channel VAE latent. Disrupting `vae.encode(img_adv)` forces denoising from a
corrupted starting point — more direct than downstream pixel loss.

VRAM: VAE encode-only is ~10x cheaper than full denoising (no transformer steps).
This enables latent loss on every batch including SD 1.5.

Basis: PhotoGuard (Salman et al. 2023) encoder-space attacks showed 7-12% better
transfer than pixel-space push-to-black. This extends that insight to the 16-ch VAE.

## Prediction

1. Protection rate on SD3.5 / FLUX improves +8-15% vs baseline without latent loss.
2. Negligible VRAM overhead (VAE encode only).
3. No degradation to SD 1.5 protection.
4. Faster convergence than loss1 (cleaner gradient through shorter compute graph).
