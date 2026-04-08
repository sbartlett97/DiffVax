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

### 5.2 Why Multi-Model Training Generalizes to Held-Out Architectures

**The VAE bottleneck hypothesis.** All three tested architectures (SD 1.5, FLUX.1-schnell, SD 3.5) share a common computational structure: the input image is encoded by a VAE before any architecture-specific processing. The VAE's role is to compress the image into a lower-dimensional latent representation $z = \text{VAE.encode}(x)$. If the immunization perturbation systematically corrupts the latent code $z$ — pushing it into regions of the latent space that no diffusion model can interpret coherently — it should disrupt editing regardless of the downstream architecture.

H1a's training objective does not explicitly target the VAE. However, backpropagating through FLUX's 4-step denoising pass necessarily flows gradients through FLUX's 16-channel VAE encoder, which shares architectural principles with SD 3.5's VAE. The implicit regularization from multi-model training thus forces the perturbation to corrupt shared representational structure, not model-specific features.

**The bimodal curriculum effect.** The observed bimodal loss distribution (FLUX batches: Loss₁ ≈ 0.8–1.3; SD1.5 batches: Loss₁ ≈ 0.05–0.15) has an important optimization consequence. FLUX's stronger editing capability means FLUX-targeting gradients have higher norm — they push the immunizer strongly toward disrupting high-capability editing. SD1.5 epochs, by contrast, have lower-norm gradients that act as a stabilizing force, keeping the perturbation in a regime that also disrupts simpler editing models.

This dynamic is analogous to GAN training: the "generator" (immunizer) must satisfy two discriminators (attack models) with different capacities. The high-capacity discriminator (FLUX) drives strong perturbations; the low-capacity discriminator (SD1.5) prevents mode collapse toward perturbations specialized for FLUX. We hypothesize this is why H1a transfers to SD3.5: the mixed-capacity training regime produces perturbations that lie in the intersection of "disrupts high-capacity models" and "disrupts low-capacity models" — a region that generalizes across architectures.

**Comparison to SD1.5-only training.** The original DiffVax checkpoint is optimized exclusively against SD1.5, a relatively weak editor. Its perturbations likely exploit SD1.5-specific architectural features (e.g., 4-channel VAE tokenization, specific cross-attention patterns). FLUX's 16-channel VAE and joint image-text attention (DiT architecture) are architecturally distinct enough that SD1.5-targeting perturbations provide little signal — consistent with Zhao et al.'s finding that FLUX.1-fill-dev can purify SD1.5 immunizations with +3–6 dB recovery.

---

### 5.3 Why STE JPEG Training Forces Energy Into Survivor Bands

**DCT frequency analysis.** JPEG compression works by (1) transforming 8×8 pixel blocks into DCT frequency coefficients, (2) quantizing each coefficient by a quality-dependent factor (coarser at lower quality), and (3) discarding zero/near-zero quantized coefficients. At q=75, the quantization table eliminates approximately 60% of DCT coefficients in each block; at q=70, approximately 70% are eliminated [DCT-SHIELD-2025].

Standard Lp-bounded adversarial perturbations concentrate energy in high-spatial-frequency bands (they look like subtle texture noise to humans). These are precisely the bands most aggressively quantized by JPEG. As a result, standard immunizations are effectively removed by q=75 JPEG compression — the perturbation signal is destroyed before the adversary's editor sees it.

**STE mechanism.** The STE training applies JPEG in the forward pass and uses the identity in the backward pass:
$$\nabla_{\tilde{x}} \mathcal{L} := \nabla_{\tilde{x}^{\text{comp}}} \mathcal{L}$$
On each JPEG-augmented training step, the attack model sees only the JPEG-compressed image, so the loss measures disruption in a JPEG-compressed world. The gradient signal therefore only rewards perturbation directions that survive compression — DCT bands that are *not* quantized to zero at the target quality.

After sufficient STE training steps, the immunizer converges to perturbations whose energy is predominantly in JPEG-survivor bands: mid-frequency DCT coefficients (spatial frequencies of 1–16 cycles/8-pixel block) that are preserved at q=70–85. These bands are harder to eliminate without also degrading visible image content, which is why H7 perturbations survive commodity purification tools as well.

**IDProtector comparison.** IDProtector [CHEN2024] explicitly states: "applying differentiable JPEG compression through Straight-Through Estimator [...] introduces substantial learning burden." They use a Gaussian noise proxy (σ=0.05) evaluated only at q=85. At q=85, many high-frequency perturbation components survive; the Gaussian proxy is a reasonable approximation. But at q=70–75, the quantization table is much more aggressive and only bands within the survivor envelope survive. A Gaussian proxy does not model the banded structure of JPEG survivor frequencies. H7 directly learns the survivor structure.

---

### 5.4 Failure Modes and Limitations

**GPT-image-edit (black-box API).** We cannot backpropagate through gpt-image-edit. We test DiffVax++ H7 on a qualitative sample of 20 images and find [X/20] disrupted. This is a transfer-only scenario and results will be variable. A production system protecting against gpt-image-edit would need black-box transfer methods (e.g., ensemble of open-source surrogate models).

**Adversarial adaptive attacks.** An adversary aware of DiffVax++ could (a) use q=60 JPEG or (b) combine JPEG and SR purification. We do not evaluate these cases. H7 trains on q=70–85; q=60 is outside the training distribution and we expect significant EDR reduction. This is an acknowledged limitation.

**Perturbation visibility at low PSNR.** The 50% overlap 1088px configuration achieves PSNR=28.7 dB, which is at our 28 dB imperceptibility threshold. In edge cases (very smooth images), the perturbation may be faintly visible. Production deployment should include a PSNR check and fall back to stride=384 (PSNR ≈ 28.9 dB) if needed.

**FLUX.1-schnell vs FLUX.1-dev.** We train against the distilled Schnell variant (4 steps). FLUX.1-dev (20 steps, guidance-enabled) is a stronger editor. We expect H1a to transfer to dev based on the shared architecture, but do not report dev results. This is future work.
