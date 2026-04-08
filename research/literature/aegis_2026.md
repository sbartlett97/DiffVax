# AEGIS: Diffusion-Guided Adversarial Perturbation Injection
**arXiv**: 2604.01635 (April 2, 2026)
**Authors**: Yue Li et al.

## Key Finding
Injects adversarial perturbations into latent space along the DDIM denoising trajectory. Trajectory-aware injection (perturbation effective across multiple denoising steps) is more robust than single-step injection. Works against both GAN and diffusion-based generators for facial deepfake protection.

## Relevance to DiffVax++
- **Confirms latent-space approach**: Our VAE feature loss (H4) aligns with this finding
- **Trajectory-aware = more robust**: Our multi-step differentiable pass (4 denoising steps) already does this
- **No JPEG robustness**: Does not address compression. Gap that DiffVax++ H7 fills.
- **Facial-specific**: Narrower scope than DiffVax++ (general images). Not a direct competitor.

## Use in Paper
Brief mention in Related Work: "Concurrent work AEGIS (Li et al., 2026) confirms latent-space trajectory injection is robust; our approach extends this to multi-model and compression-resistant settings."
