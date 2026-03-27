# H5: Frequency-Domain Perturbation Concentration Loss

**Status:** Protocol locked
**Date locked:** 2026-03-27
**Hypothesis:** H5 (spectral_loss)

---

## What

Add a DCT/FFT frequency-domain regularization term to the perturbation loss (`loss2`)
that penalizes low-frequency energy in the perturbation `δ = img_adv − img_orig`.

**Implementation:**
- Module: `src/diffvax/losses/spectral_loss.py`
- Loss function: `SpectralLoss` class, config-gated via `spectral_loss.enabled`
- Mechanism: `torch.fft.rfft2(δ)` → magnitude spectrum → penalize components
  with normalized distance from DC < `low_freq_radius` (default 0.1)
- Integration: Added to `LossComposer` in `src/diffvax/losses/__init__.py`
- Config key in `configs/research_v3.yml`: `spectral_loss.enabled: true`

**Why rfft2 over explicit DCT:**
`torch.fft.rfft2` is natively hardware-accelerated, and the magnitude spectrum is
equivalent to energy distribution — the penalization of low-frequency components
works identically whether using DFT or DCT for this regularization purpose.

---

## Why

At high resolution (1088×1088), the L-inf ε=32/255 perturbation budget is spread
over 4× more pixels than at 512px. L1 loss2 penalizes all frequencies uniformly.
Human visual sensitivity is strongest in the 2–10 cycle/degree band (mid-frequency).
Perturbation energy in the DC/low-frequency band (smooth color shifts) is maximally
visible; energy in the high-frequency band (fine texture noise) is minimally visible.

Concentrating perturbation energy in high frequencies allows:
- **Same SSIM budget → larger effective epsilon** (more protection at same imperceptibility)
- **Better transfer to AI models**: diffusion VAEs are effectively low-pass filters —
  encoding a high-frequency perturbed image still disrupts the latent representation
  because the VAE struggles to compress the high-frequency noise pattern faithfully.

**Literature basis:**
- DDAP (arXiv:2407.20141): DCT-domain adversarial perturbations with frequency-aware
  constraints outperform pixel-domain L-inf attacks on imperceptibility metrics.
- AdvAD (NeurIPS 2024): Frequency-aware adversarial diffusion — frequency concentration
  in high bands improves SSIM by 0.02–0.05 at same epsilon budget.

---

## Prediction (CONFIRMATORY)

1. SSIM of protected images vs originals will improve by **0.02–0.05** at same
   training iteration count vs baseline without spectral_loss.
2. PSNR will improve by **1–3 dB** at 1088px stage.
3. Protection rate (loss1 at evaluation) will be **maintained or slightly improved**
   (≤2% degradation vs baseline) since high-frequency perturbations still disrupt
   VAE encoding.
4. No meaningful VRAM increase (rfft2 is O(N log N), negligible vs attack forward pass).

---

## Evaluation Protocol

- Compare `research_v3.yml` with `spectral_loss.enabled: true` (weight=0.5) vs
  `research_v3.yml` with `spectral_loss.enabled: false` at equal training iterations.
- Metrics: mean SSIM, mean PSNR, mean loss1 (protection signal), mean loss2.
- Validation: 100 held-out images from `image_prompt_pairs_with_validation.json`.
- Training budget: 5,000 iterations at 512px resolution (fast convergence check).

---

## Notes

- `low_freq_radius=0.1` penalizes the central 10% of normalized frequency space
  (DC + lowest harmonics). This is conservative — can increase to 0.2 if needed.
- Weight of 0.5 relative to `loss2` is a reasonable starting point; may need tuning.
- The spectral loss operates on `float32` regardless of AMP precision (rfft2 requires
  real-valued input; perturbation in float16 is upcast automatically).
