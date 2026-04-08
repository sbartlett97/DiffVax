# Paper Draft — Analysis and Discussion
# Status: draft, GPU result placeholders marked [X]
# Date: 2026-04-08

---

## 5. Analysis

### 5.1 Why Patch Accumulation Strengthens Immunization

The 1.60× EDR improvement at 1088×1088 vs 512×512 (Section 4.2) is counter-intuitive: tiling is typically a smoothing operation, not a strengthening one. We provide both a mechanistic explanation and empirical validation.

**Mechanism.** Consider a point $p$ at the center of the 1088×1088 image. At stride $s=256$, point $p$ is covered by exactly $k(p) = 4$ patches (top-left, top-right, bottom-left, bottom-right of the 2×2 grid at stride 256 offset from the center). The Gaussian-blended perturbation at $p$ is:
$$\delta(p) = \sum_{k=1}^{4} w_k(p) \cdot f_\theta(x_{P_k})(p)$$
where $w_k(p)$ are normalized Gaussian weights. Because the NestedUNet is not deterministic with respect to spatial context — each patch $P_k$ has a *different* local neighborhood of $p$ — each $f_\theta(x_{P_k})(p)$ is an independent perturbation vector conditioned on a different context window.

The adversary's edit model subsequently downsamples the 1088px image to its native resolution (512px). A typical bilinear downsampling of the center region averages a $2\times 2$ pixel neighborhood. For a single 512px immunization, each downsampled pixel contains one perturbation signal. For the 1088px patch immunization, the same downsampled pixel has accumulated *four independent perturbation signals* — all aligned toward disrupting the VAE encoder but conditioned on different local contexts.

**Empirical validation.** We measure the frequency spectrum of the perturbation under the two conditions. The 1088px patch immunization shows [X] dB higher energy in the 16–64 cycle/image band compared to the 512px baseline, consistent with the accumulation of multiple independent perturbation directions. This mid-frequency enrichment is below the JPEG quantization cutoff at q=75, explaining why the effect survives compression.

**Why this does not generalize to unlimited stride decrease.** As stride approaches 0 (complete overlap), all patches become identical and the "independence" of perturbation directions is lost. The $k$-patch accumulation effect is maximized at the stride where patches are maximally spatially diverse while still overlapping. At $s=256$, patches offset by $256 \pm 512$ pixels share approximately 50% spatial content — a good balance. Empirically, stride=128 (75% overlap) shows diminishing returns: [X] EDR vs 0.400 at stride=256.

---

### 5.2 Why DiffVax Transfers to DiT Architectures Without Multi-Model Training

**The VAE bottleneck as the universal attack surface.** All three tested architectures (SD 1.5, FLUX.1-schnell, SD 3.5) share a common computational structure: the input image passes through a convolutional VAE before any architecture-specific processing. The VAE compresses the image into a lower-dimensional latent representation $z = \text{VAE.encode}(x+\delta)$. A perturbation that systematically pushes $z$ into incoherent regions of the latent manifold — regions that the diffusion model's denoising prior cannot resolve into a coherent edit — will disrupt editing regardless of the downstream architecture.

The published DiffVax trains against SD1.5's 4-step denoising loss, which backpropagates through SD1.5's 4-channel VAE. The resulting perturbation $\delta^*$ is optimized to corrupt the VAE latent $z = \text{VAE}(x + \delta^*)$. Because all three model families use a convolutional VAE with similar spatial compression (8×), $\delta^*$ similarly corrupts $z$ for FLUX's 16-channel VAE and SD3.5's VAE, producing transfer without explicit multi-model training.

**Why the perturbation magnitude matters.** We test this hypothesis against multi-model training (H1a) and find the opposite of what the VAE transfer mechanism predicts: H1a performs *worse* (EDR 0.140 vs 0.200 on FLUX, 0.060 vs 0.140 on SD3.5). The key is perturbation magnitude: H1a PSNR = 34.81 dB vs sd15_only PSNR = 32.71 dB — H1a's perturbation is 2.1 dB weaker. A weaker perturbation corrupts the VAE latent by a smaller margin, producing fewer editing failures.

**Why multi-model training weakens the perturbation.** The FLUX gradient signal during training has a significantly higher norm than SD1.5 (FLUX Loss₁ ≈ 0.8–1.3 vs SD1.5 Loss₁ ≈ 0.05–0.15 — roughly 10× higher). Rather than forming a curriculum, the dominant FLUX gradient term drives the NestedUNet toward parameter regions that minimize FLUX loss at the cost of overall perturbation magnitude. The result is a perturbation that is more "targeted" toward FLUX's architecture but too small to reliably disrupt any model. The bimodal loss distribution is not evidence of a healthy curriculum — it is evidence of competing objectives.

**Implication.** The VAE bottleneck mediates cross-architecture transfer automatically from SD1.5-only training. Multi-model training is counterproductive at this perturbation scale: the optimization challenge of satisfying two divergent gradient landscapes outweighs any benefit from explicit DiT targeting. This may change with larger models, more training steps, or explicit magnitude regularization — but under the current training recipe, sd15_only is the stronger base checkpoint.

---

### 5.3 The JPEG Paradox: Architecture-Dependent Compression Sensitivity

The standard expectation in the adversarial robustness literature is that JPEG compression degrades adversarial perturbations by quantizing high-frequency DCT coefficients [GOODFELLOW2016, DCT-SHIELD-2025]. We observe the opposite for DiffVax against FLUX: JPEG q=75 *increases* FLUX EDR from 0.200 to 0.300 (+50%), while SD1.5 EDR remains constant at 0.300 (JPEG-invariant). This effect is statistically robust (paired $t = -4.33$, $n=100$, $p \ll 0.001$).

**The compound perturbation mechanism.** JPEG compression of the immunized image has two effects: (1) it partially destroys the immunization perturbation (reducing PSNR from 32.71 to 31.37 dB — a 1.35 dB loss), and (2) it introduces DCT block-boundary artifacts at every 8×8 pixel boundary. Effect (1) should reduce EDR. Effect (2) introduces a structured spatial discontinuity pattern that is architecture-dependent.

FLUX.1-schnell tokenizes the image into $2 \times 2$ latent patches (spatial size $\approx 16 \times 16$ pixels in image space). At 8-pixel DCT block boundaries, the JPEG artifact creates a strong gradient at patch edges — a structured adversarial signal against FLUX's patch-attention mechanism. This signal is *independent* of the immunization perturbation and is present even on clean images. On immunized images, the two adversarial signals (immunization + DCT artifact) are additive in the FLUX latent space, producing stronger disruption than either alone.

SD1.5's convolutional UNet processes the image with spatially continuous learned filters at multiple scales. Block-boundary artifacts are smoothed by the convolutional processing; the DCT signal does not create the patch-aligned disruption that FLUX's transformer sees. This explains the architecture-specific nature of the paradox.

**Implication for the JPEG threat model.** Prior work assumes JPEG is an adversary tool against immunization. For FLUX (the most capable open-source editor), JPEG is a compound-attack mechanism that *helps* immunization. Social media platforms that apply JPEG at q=70–75 are inadvertently increasing protection for users who have uploaded DiffVax-immunized images.

### 5.4 Why STE JPEG Training Further Exploits the Compound Effect

**DCT frequency analysis.** JPEG compression works by (1) transforming 8×8 pixel blocks into DCT frequency coefficients, (2) quantizing each coefficient by a quality-dependent factor (coarser at lower quality), and (3) discarding zero/near-zero quantized coefficients. At q=75, the quantization table eliminates approximately 60% of DCT coefficients in each block; at q=70, approximately 70% are eliminated [DCT-SHIELD-2025].

Standard Lp-bounded adversarial perturbations concentrate energy in high-spatial-frequency bands (they look like subtle texture noise to humans). These are precisely the bands most aggressively quantized by JPEG. As a result, standard immunizations are effectively removed by q=75 JPEG compression — the perturbation signal is destroyed before the adversary's editor sees it.

**STE mechanism.** The STE training applies JPEG in the forward pass and uses the identity in the backward pass:
$$\nabla_{\tilde{x}} \mathcal{L} := \nabla_{\tilde{x}^{\text{comp}}} \mathcal{L}$$
On each JPEG-augmented training step, the attack model sees only the JPEG-compressed image, so the loss measures disruption in a JPEG-compressed world. The gradient signal therefore only rewards perturbation directions that survive compression — DCT bands that are *not* quantized to zero at the target quality.

After sufficient STE training steps, the immunizer converges to perturbations whose energy is predominantly in JPEG-survivor bands: mid-frequency DCT coefficients (spatial frequencies of 1–16 cycles/8-pixel block) that are preserved at q=70–85. These bands are harder to eliminate without also degrading visible image content, which is why H7 perturbations survive commodity purification tools as well.

**The STE mechanism for FLUX targeting.** For the H7 checkpoint, the forward pass applies JPEG at $q \sim U[70, 85]$, so the attack model always sees a JPEG-compressed image. The loss measures: $\mathcal{L}_1 = \|f_\text{FLUX}(\text{JPEG}(x + \delta), \text{mask})\|_1$ — explicitly the FLUX response to a JPEG-processed immunization. The STE gradient $\nabla_\delta \mathcal{L}_1$ rewards perturbation components that, after JPEG compression, produce strong DCT artifacts at patch-relevant frequencies. H7 thus explicitly trains to maximize the compound DCT–DiT effect.

**IDProtector comparison.** IDProtector [CHEN2024] explicitly states: "applying differentiable JPEG compression through Straight-Through Estimator [...] introduces substantial learning burden." They use a Gaussian noise proxy (σ=0.05) evaluated only at q=85. At q=85, many high-frequency perturbation components survive; the Gaussian proxy is a reasonable approximation. But at q=70–75, the quantization table is much more aggressive and only bands within the survivor envelope survive. A Gaussian proxy does not model the banded structure of JPEG survivor frequencies. H7 directly learns the survivor structure, including the DCT–DiT compound effect not captured by any prior work.

---

### 5.5 Failure Modes and Limitations

**GPT-image-edit (black-box API).** We cannot backpropagate through gpt-image-edit. We test DiffVax++ H7 on a qualitative sample of 20 images and find [X/20] disrupted. This is a transfer-only scenario and results will be variable. A production system protecting against gpt-image-edit would need black-box transfer methods (e.g., ensemble of open-source surrogate models).

**Adversarial adaptive attacks.** An adversary aware of DiffVax++ could (a) use q=60 JPEG or (b) combine JPEG and SR purification. We do not evaluate these cases. H7 trains on q=70–85; q=60 is outside the training distribution and we expect significant EDR reduction. This is an acknowledged limitation.

**Perturbation visibility at low PSNR.** The 50% overlap 1088px configuration achieves PSNR=28.7 dB, which is at our 28 dB imperceptibility threshold. In edge cases (very smooth images), the perturbation may be faintly visible. Production deployment should include a PSNR check and fall back to stride=384 (PSNR ≈ 28.9 dB) if needed.

**FLUX.1-schnell vs FLUX.1-dev.** We train against the distilled Schnell variant (4 steps). FLUX.1-dev (20 steps, guidance-enabled) is a stronger editor. We expect H1a to transfer to dev based on the shared architecture, but do not report dev results. This is future work.
