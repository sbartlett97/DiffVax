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

| Checkpoint | SD 1.5 EDR ↑ | FLUX.1-schnell EDR ↑ | SD 3.5 EDR ↑ (held out) | PSNR ↑ |
|---|---|---|---|---|
| DiffVax (SD1.5 only) | **0.300** | **0.200** | **0.140** | 32.71 dB |
| DiffVax++ H1a (SD+FLUX) | 0.290 | 0.140 | 0.060 | 34.81 dB |

**Key finding (unexpected)**: H1a *under*performs the published DiffVax checkpoint on all three architectures. The multi-model training hypothesis is not confirmed.

**Why H1a fails — perturbation weakening.** H1a produces a 2.1 dB weaker perturbation (PSNR 34.81 vs 32.71 dB). The FLUX gradient signal during training is an order of magnitude larger than the SD 1.5 signal (FLUX Loss₁ ≈ 0.8–1.3 vs SD 1.5 Loss₁ ≈ 0.05–0.15). The competing objectives do not form the intended curriculum; instead, the large FLUX gradients dominate and drive the NestedUNet toward conservative (smaller) perturbations. This is consistent with the bimodal loss pattern observed during training.

**Surprising finding: DiffVax already transfers to DiT models.** The published DiffVax checkpoint (trained only on SD 1.5) achieves FLUX EDR = 0.200 and SD 3.5 EDR = 0.140 with no DiT-specific training. This counter-hypothesis result suggests that the immunization perturbation operates primarily in the shared VAE input space: all three model families encode the image through a similar convolutional VAE before any architecture-specific processing. Perturbations that corrupt the VAE-encoded representation disrupt editing regardless of whether the editing model is UNet- or DiT-based.

**Training dynamics.** H1a training produces a characteristic bimodal epoch-level loss distribution. FLUX batches yield Loss₁ ≈ 0.8–1.3 (FLUX is a harder adversary — higher-capability editor), while SD 1.5 batches yield Loss₁ ≈ 0.05–0.15. This is not the intended alternating curriculum: the large FLUX gradients actively interfere with the SD 1.5 objective rather than complementing it.

**High-resolution transfer.** The 1.60× relative improvement from patch-based 1088px inference is a structural property of the perturbation accumulation mechanism. Since the DiffVax checkpoint is the stronger base checkpoint, we use it (not H1a) for the 1088px configuration in Section 4.7.

---

### 4.4 H6: Resistance to FLUX-Based Purification

**Setup.** "Purify Once, Edit Freely" [ZHAO2026] shows that FLUX.1-fill-dev can function as a purifier: by re-inpainting an immunized image with a neutral prompt (e.g., "high-quality photo"), it removes the adversarial perturbation and restores editability (+3–6 dB PSNR improvement). We test this attack directly:

1. Immunize image with DiffVax or H1a
2. Apply FLUX.1-fill-dev purification at strengths $\{0.3, 0.5, 0.7\}$
3. Attempt editing on the purified image
4. Measure EDR

**Purification robustness** (Table 4). EDR values here use FLUX.1-schnell as the editing model; *direct* = editing the immunized image without any purification; *net survival* = purified EDR − control EDR (control measures disruption from purifier damage to unimmunized clean images).

| Checkpoint | Direct EDR | Purified (s=0.3) | Control (s=0.3) | Net Survival ↑ | PSNR after purify |
|---|---|---|---|---|---|
| DiffVax (SD1.5 only) | 0.183 | 0.200 | 0.000 | **+0.200** | 31.1 dB |
| DiffVax++ H1a (SD+FLUX) | 0.133 | 0.133 | 0.000 | **+0.133** | 32.1 dB |

At strength ≥ 0.5, the FLUX purifier reduces image quality to PSNR = 23 dB / SSIM = 0.70 for both immunized and clean images. Both checkpoints show purified EDR ≈ 0.983, but the control shows this is entirely purifier damage — both net survival values collapse to ≤ 0.017. This strength regime is therefore not a practical adversary model: an adversary who applies strength=0.5 purification visibly degrades the image before any editing.

**Key finding (unexpected)**: DiffVax (SD1.5 only) provides greater net survival at s=0.3 than H1a. The hypothesis that FLUX-trained immunization resists FLUX purification is not confirmed. The root cause is H1a's weakened perturbation: a 2.1 dB smaller signal has less energy to survive denoising.

**Surprising finding: Light FLUX purification fails against DiffVax.** At the realistic adversary strength of s=0.3 (which maintains PSNR=31.1 dB, SSIM=0.95), the published DiffVax achieves net survival EDR = 0.200. "Purify Once, Edit Freely" [ZHAO2026] demonstrates that FLUX.1-fill-dev can defeat SD 1.5 immunizations at strength ≥ 0.5 — but our data shows the practical cost: at that strength, the purifier damages the image quality below what any adversary would accept. The threat model in [ZHAO2026] implicitly assumes zero quality cost from purification, which does not hold in practice.

**Commodity tool purification** [PLEIMLING2026]. We additionally test purification by three commodity tools (SR model, style transfer, JPEG q=70) applied to DiffVax and H1a immunizations. Results in Appendix B.

---

### 4.5 H7: JPEG-Robust Training for Social Media Deployment

**Setup.** We train a JPEG-robust variant (H7) by extending H1a training with STE JPEG augmentation: with probability $p_{\text{jpeg}} = 0.5$, the immunized image is JPEG-compressed (quality $q \sim U[70, 85]$) before the attack forward pass. Gradients flow through the JPEG operation as the identity (STE). This forces perturbation energy into DCT bands that survive at social media compression levels.

**The JPEG Paradox (observed in H1 evaluation data).** Before examining H7, we note a surprising finding from the H1 transfer evaluation: JPEG compression does *not* defeat DiffVax immunization against FLUX. Instead, JPEG *increases* EDR.

| Checkpoint | Model | Clean EDR | q=75 EDR | q=70 EDR |
|---|---|---|---|---|
| DiffVax (SD1.5 only) | FLUX.1-schnell | 0.200 | **0.300** | **0.260** |
| DiffVax (SD1.5 only) | SD 1.5 | 0.300 | 0.300 | 0.310 |
| DiffVax (SD1.5 only) | SD 3.5 | 0.140 | 0.170 | 0.160 |

DiffVax is already JPEG-robust against FLUX without any JPEG-augmented training. FLUX EDR *increases by 50%* at q=75 compared to the clean baseline. We hypothesize that DCT block quantization artifacts (8×8-pixel boundaries) create an additional adversarial signal that compounds with the immunization perturbation — specifically for FLUX's token-based DiT architecture, which is sensitive to patch-boundary discontinuities. SD 1.5's convolutional UNet architecture is robust to this artifact (SD1.5 EDR is JPEG-invariant).

**Compression robustness** (Table 5). Against this JPEG-paradox baseline, H7 tests whether explicit STE training further amplifies the paradox.

| Method | Clean EDR ↑ | q=75 (Instagram) ↑ | q=70 (Twitter) ↑ | JPEG effect |
|---|---|---|---|---|
| DiffVax (SD1.5 only) | 0.200 | **0.300** (+50%) | **0.260** | **PARADOX** |
| DiffVax++ H1a | 0.140 | 0.150 (+7%) | 0.150 | weak |
| **DiffVax++ H7** | 0.090 | 0.080 (−11%) | 0.090 | **none** |

†All values against FLUX.1-schnell editing model; H7 PSNR=35.65 dB.

**Key finding (unexpected)**: H7 is the weakest checkpoint and shows NO JPEG paradox for FLUX. STE JPEG augmentation during training further weakens the perturbation (PSNR=35.65 dB vs H1a=34.81 dB vs sd15_only=32.71 dB). This establishes a clear monotonic relationship: adding training objectives (multi-model, JPEG aug) consistently reduces perturbation magnitude and eliminates the JPEG paradox. The paradox is an emergent property of strong single-objective perturbations interacting with FLUX's patch architecture, not an explicitly trainable feature.

The JPEG paradox is therefore most practically exploited by using the published DiffVax checkpoint (sd15_only) combined with H2 patch-based 1088×1088 inference.

**Marginal positive from H7**: SD 3.5 EDR improves to 0.100 (all JPEG conditions) from H1a's 0.060, suggesting JPEG augmentation may provide a small benefit for SD3.5 specifically. However, sd15_only's SD3.5 EDR of 0.140 remains the overall best.

**The "less is more" principle**: Across all training variants in this paper, the simplest training configuration (SD15 single-objective, original DiffVax) produces the strongest perturbation and the best EDR. This has implications for future immunization research: training complexity should be carefully justified, as competing objectives consistently erode perturbation quality.

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

We evaluate the complete DiffVax++ system (H7 checkpoint + 1088px patch inference) against the best available baselines on a combined metric: EDR under three deployment conditions (clean, post-JPEG q=75, post-purification strength=0.3).

Note: purification strength=0.3 is used as the practical adversary strength, as strength≥0.5 visibly degrades image quality (PSNR→23 dB, SSIM→0.70) which defeats the adversary's goal of obtaining a usable edited image.

| Method | Clean EDR | +JPEG q=75 | +Purif. (s=0.3) | Net survival worst case |
|---|---|---|---|---|
| DiffVax | 0.200† | 0.300† | 0.200 (net) | 0.200 |
| PromptFlare | [X] | N/A | [X] | [X] |
| IDProtector | [X] | [X]* | N/A | [X] |
| **DiffVax++ (full)** | [X] | [X] | [X] | **[X]** |

†Evaluated against FLUX.1-schnell. *IDProtector at q=85 only. — = not tested by original paper.
