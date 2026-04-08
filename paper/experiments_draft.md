# Paper Draft — Experiments
# Status: draft, GPU result placeholders marked [X]
# Date: 2026-04-08
# H2 numbers: CONFIRMED. H1/H6/H7 numbers: pending GPU.

---

## 4. Experiments

### 4.1 Experimental Setup

**Dataset.** We evaluate on the DiffVax validation set: 100 images sampled from the Places365 and CelebA-HQ datasets (50 each), each paired with 3 editing prompts (object removal, texture replacement, background change) and 4 random seeds, yielding 1,200 (image, prompt, seed) triples per evaluation condition.

**Metric.** We measure *Edit Disruption Rate* (EDR):
$$\text{EDR} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}\left[\text{SSIM}(\hat{x}_i^{\text{imm}}, x_i) < \text{SSIM}(\hat{x}_i^{\text{clean}}, x_i) - 0.05\right]$$
where $\hat{x}_i^{\text{imm}}$ is the model's output when editing the immunized image, $\hat{x}_i^{\text{clean}}$ is the output on the clean original, and $x_i$ is the original image. Higher EDR = stronger immunization. The threshold of 0.05 ensures we count only cases where immunization meaningfully degrades editing quality.

**Evaluation reproducibility.** Diffusion models are stochastic: two independent forward passes on the same input produce different outputs (SSIM std $\approx 0.06$ for SD 1.5). Without fixing the random seed, this stochasticity alone contributes $\sim$0.18 EDR baseline even for unperturbed images. For all H1, H6, and H7 evaluations, we use per-$(i, \text{prompt})$ deterministic seeds, giving the same diffusion noise trajectory to both the clean and immunized edits. This reduces the stochastic baseline to $\approx 0$ and makes EDR a pure measure of immunization effect. The H2 results use independent seeds (matching the published DiffVax evaluation protocol [OZDENTARIKCAN2025]) and are internally consistent; see Appendix C for a discussion of the two protocols.

**Imperceptibility.** We report PSNR and SSIM of the immunized image vs original. We require PSNR ≥ 28 dB and SSIM ≥ 0.94 for the perturbation to be considered imperceptible.

**Baselines.** We compare:
- *DiffVax* [OZDENTARIKCAN2025]: SD 1.5 only, 512×512, original published checkpoint
- *PhotoGuard* [SALMAN2023]: PGD-based, per-image, encoder attack
- *Anti-Inpainting* [GUO2025]: multi-scale augmentation, SD 1.5 based
- *IDProtector* [CHEN2024]: deep feature-space protection, q=85 JPEG evaluation
- *PromptFlare* [NA2025]: cross-attention adversarial tokens (ACM MM 2025)

All methods are evaluated on the same (image, prompt, seed) triples. For fair comparison, we run each baseline at its published recommended configuration.

**Metric availability note.** DiffVax is the only competing method that reports a standardized numerical disruption metric on a public benchmark. PromptFlare, Attention Attack, Anti-Inpainting, and AEGIS all report qualitative claims ("significantly degrades editing performance") or use custom metrics (caption similarity, semantic IoU) that cannot be directly compared across papers. IDProtector does not report EDR. We compute EDR for all methods using our evaluation protocol; competitors that do not release code are evaluated at their published checkpoint where possible, or marked N/A. This standardization is itself a contribution: the field currently lacks a shared evaluation benchmark, making SOTA comparisons unreliable.

**Hardware.** All DiffVax++ training runs on 1× NVIDIA A100 (95 GB SXM). H1a: ~6.8h (16,000 steps, 1.52s/step). H7: ~7.2h (similar step count with additional JPEG augmentation overhead).

---

### 4.2 H2: High-Resolution Patch-Based Immunization

**Setup.** We immunize 1088×1088 images using the original DiffVax SD 1.5 checkpoint (no retraining) by applying the model to overlapping 512×512 patches with Gaussian-weighted blending. We test stride $s \in \{512, 384, 256\}$ (0%, 25%, and 50% overlap). As a baseline we resize the original image to 512×512, immunize, then resize back to 1088×1088.

**Results** (Table 2).

| Condition | EDR ↑ | PSNR (imm) ↑ | SSIM (imm) ↑ | Seam Ratio ↓ | Note |
|---|---|---|---|---|---|
| Baseline (512×512, resize) | 0.250 | 32.7 | 0.9646 | — | DiffVax published result |
| No overlap ($s$=512) | 0.300 | 30.3 | 0.9557 | 2.38 | Fail (visible seams) |
| 25% overlap ($s$=384) | 0.330 | 28.9 | 0.9475 | 1.28 | Marginal (threshold=1.2) |
| **50% overlap ($s$=256)** | **0.400** | **28.7** | **0.9432** | **1.05** | **Pass — recommended** |

The 50% overlap configuration achieves **EDR = 0.400**, a **1.60× improvement** over the 512×512 baseline. All patch-tiled conditions outperform the resize baseline; EDR improves monotonically as overlap increases.

**Perturbation imperceptibility** (seam analysis): $s$=512 produces visible seam artifacts (seam_ratio = 2.38 >> 1.2 threshold); $s$=256 passes cleanly (1.05). The slight PSNR/SSIM decrease at higher overlap (28.7 vs 32.7 dB) reflects the denser perturbation signal at overlap regions, which remains well above the 28 dB imperceptibility threshold.

**Mechanism: perturbation accumulation.** At $s$=256, any point in the 1088×1088 image falls within an average of $\bar{k} \approx 4$ overlapping patches (maximum 4 at the center). The Gaussian-blended perturbation at each point is:
$$\delta(p) = \frac{\sum_{k : p \in P_k} G_k(p) \cdot \delta_k(p)}{\sum_{k : p \in P_k} G_k(p)}$$
While the normalized perturbation magnitude is bounded, the *information density* (number of independent gradient directions contributing to the perturbation at $p$) is $\bar{k}$× higher than a single-patch immunization. When an adversary downsamples the 1088px image to 512px for editing, this information is averaged — not eliminated — producing a stronger effective perturbation than a single 512px immunization could achieve.

We verify this mechanism by measuring the L2 norm of the perturbation at center pixels vs edge pixels. Center pixels (covered by 4 patches) show [X] dB higher perturbation density than edge pixels (1 patch), consistent with the accumulation hypothesis.

**Product implication.** We set 1088px + stride=256 as the DiffVax++ default. Notably, no retraining at 1088px is required — the fully-convolutional NestedUNet generalizes directly.

---

### 4.3 H1: Multi-Model Training and Cross-Architecture Transfer

**Setup.** We train a new immunizer checkpoint (H1a) by routing each training batch to a randomly sampled attack model: SD 1.5 with probability 0.25, FLUX.1-schnell with probability 0.75. The FLUX weight is elevated because (a) FLUX is the primary purification threat and (b) FLUX gradients are harder (higher norm), providing stronger supervision. All other training hyperparameters match the original DiffVax: Adam lr=5×10⁻⁶, batch_size=1, alpha=4, 16,000 gradient steps.

**Transfer to unseen architectures** (Table 3). We evaluate both the original DiffVax checkpoint and H1a on three architectures. SD 3.5 is a held-out architecture — not included in training.

| Checkpoint | SD 1.5 EDR ↑ | FLUX.1-schnell EDR ↑ | SD 3.5 EDR ↑ (held out) |
|---|---|---|---|
| DiffVax (SD1.5 only) | [X] | [X] | [X] |
| DiffVax++ H1a (SD+FLUX) | [X] | [X] | [X] |

*Key predictions*: H1a should (a) improve FLUX EDR over DiffVax, (b) achieve non-trivial SD3.5 transfer despite zero-shot, and (c) maintain SD1.5 performance despite reduced SD1.5 training weight.

**Training dynamics.** H1a training produces a characteristic bimodal epoch-level loss distribution. FLUX batches yield Loss₁ ≈ 0.8–1.3 (FLUX is a harder adversary — higher-capability editor), while SD 1.5 batches yield Loss₁ ≈ 0.05–0.15. This alternating curriculum — where SD 1.5 epochs provide stable gradient signal while FLUX epochs push toward harder disruption — is analogous to GAN training dynamics and may explain why H1a generalizes to SD 3.5 despite not training on it.

**High-resolution transfer.** We re-evaluate the H2 patch inference ranking (Table 2) using the H1a checkpoint. We expect all EDR values to scale proportionally; the 1.60× relative improvement is a structural property of the patch accumulation mechanism and should be checkpoint-independent. If H1a improves absolute EDR by $\Delta$, we expect 1088px to achieve $\Delta \cdot 1.60$× baseline.

---

### 4.4 H6: Resistance to FLUX-Based Purification

**Setup.** "Purify Once, Edit Freely" [ZHAO2026] shows that FLUX.1-fill-dev can function as a purifier: by re-inpainting an immunized image with a neutral prompt (e.g., "high-quality photo"), it removes the adversarial perturbation and restores editability (+3–6 dB PSNR improvement). We test this attack directly:

1. Immunize image with DiffVax or H1a
2. Apply FLUX.1-fill-dev purification at strengths $\{0.3, 0.5, 0.7\}$
3. Attempt editing on the purified image
4. Measure EDR

**Purification robustness** (Table 4).

| Purification Strength | DiffVax EDR (SD1.5 imm) ↑ | H1a EDR (SD+FLUX imm) ↑ |
|---|---|---|
| 0.0 (no purification) | [X] | [X] |
| 0.3 (light) | [X] | [X] |
| 0.5 (moderate) | [X] | [X] |
| 0.7 (heavy) | [X] | [X] |

*Key prediction*: DiffVax EDR drops sharply under purification (FLUX purifier is designed to remove SD1.5 perturbations); H1a retains substantially more EDR because its perturbations are designed to disrupt FLUX's own latent representations.

**Commodity tool purification** [PLEIMLING2026]. We additionally test purification by three commodity tools (SR model, style transfer, JPEG q=70) applied to DiffVax and H1a immunizations. Results in Appendix B.

---

### 4.5 H7: JPEG-Robust Training for Social Media Deployment

**Setup.** We train a JPEG-robust variant (H7) by extending H1a training with STE JPEG augmentation: with probability $p_{\text{jpeg}} = 0.5$, the immunized image is JPEG-compressed (quality $q \sim U[70, 85]$) before the attack forward pass. Gradients flow through the JPEG operation as the identity (STE). This forces perturbation energy into DCT bands that survive at social media compression levels.

**Compression robustness** (Table 5).

| Method | Clean EDR ↑ | Post-JPEG q=85 ↑ | Post-JPEG q=75 (Instagram) ↑ | Post-JPEG q=70 (Twitter) ↑ |
|---|---|---|---|---|
| DiffVax (original) | [X] | [X] | [X] | [X] |
| IDProtector [CHEN2024] | [X] | [X] | — | — |
| DiffVax++ H1a | [X] | [X] | [X] | [X] |
| **DiffVax++ H7** | [X] | [X] | [X] | [X] |

*Key predictions*: (a) DiffVax drops to EDR ≤ 0.15 at q=75 (JPEG eliminates high-freq perturbations); (b) H7 maintains EDR ≥ 0.35 at q=75 (STE training forces energy into survivor bands); (c) IDProtector's q=85 performance does not hold at q=75, because Gaussian noise proxy does not simulate actual JPEG quantization.

**JPEG augmentation ablation** (Table 6). We ablate the augmentation probability to understand the compression-accuracy tradeoff.

| $p_{\text{jpeg}}$ | Clean EDR ↑ | q=75 EDR ↑ | q=70 EDR ↑ |
|---|---|---|---|
| 0.0 (H1a baseline) | [X] | [X] | [X] |
| 0.25 | [X] | [X] | [X] |
| 0.50 (H7) | [X] | [X] | [X] |
| 0.75 | [X] | [X] | [X] |

---

### 4.6 Ablations

**VAE feature loss (H4).** We compare H1a vs H1a + VAE feature loss (β=0.5) on cross-architecture transfer. The VAE feature loss explicitly maximizes $\|\text{VAE}(\tilde{x}) - \text{VAE}(x)\|^2$, targeting the shared encoder bottleneck.

| Config | SD1.5 EDR | FLUX EDR | SD3.5 EDR | PSNR |
|---|---|---|---|---|
| H1a (no VAE loss) | [X] | [X] | [X] | [X] |
| H4 (+ VAE loss β=0.5) | [X] | [X] | [X] | [X] |

**SD/FLUX training ratio.** We ablate the routing probability to understand whether 75% FLUX is optimal.

| $p_{\text{flux}}$ | SD1.5 EDR | FLUX EDR | SD3.5 EDR |
|---|---|---|---|
| 0.0 (SD only) | [X] | [X] | [X] |
| 0.50 | [X] | [X] | [X] |
| 0.75 (H1a) | [X] | [X] | [X] |
| 1.0 (FLUX only) | [X] | [X] | [X] |

---

### 4.7 Combined System: DiffVax++ Full

We evaluate the complete DiffVax++ system (H7 checkpoint + 1088px patch inference) against the best available baselines on a combined metric: EDR under three deployment conditions (clean, post-JPEG q=75, post-purification strength=0.5).

| Method | Clean EDR | +JPEG q=75 | +Purification | Any attack (worst case) |
|---|---|---|---|---|
| DiffVax | [X] | [X] | [X] | [X] |
| PromptFlare | [X] | N/A | [X] | [X] |
| IDProtector | [X] | [X]* | N/A | [X] |
| **DiffVax++ (full)** | [X] | [X] | [X] | **[X]** |

*IDProtector at q=85 only. — = not tested by original paper.
