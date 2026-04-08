# Paper Draft — Method
# Status: draft
# Date: 2026-04-08

---

## 3. DiffVax++ Method

### 3.1 Background: DiffVax

DiffVax [OZDENTARIKCAN2025] trains a NestedUNet $f_\theta: \mathbb{R}^{B \times 3 \times H \times W} \to \mathbb{R}^{B \times 3 \times H \times W}$ to produce an immunization perturbation via end-to-end backpropagation through a differentiable inpainting forward pass.

Given an image $x$, mask $m$ (1 = edit region), and text prompt $p$, the immunized image is:
$$\tilde{x} = \text{clip}(x + f_\theta(x) \odot (1 - m), -1, 1)$$

The perturbation is applied only outside the masked region, so the immunized image looks identical to the original. The training objective minimizes:
$$\mathcal{L} = \mathcal{L}_1(\text{Edit}(\tilde{x}, m, p), \mathbf{0}) + \alpha \cdot \mathcal{L}_1(f_\theta(x) \odot (1 - m), \mathbf{0})$$

where $\text{Edit}(\cdot)$ is the differentiable inpainting forward pass and $\mathbf{0}$ is an all-zeros target (blank output). The first term maximizes editing disruption; the second regularizes perturbation magnitude. Both terms are normalized by image resolution to be resolution-agnostic.

### 3.2 Multi-Model Training (H1)

The key architectural difference between SD 1.5 and DiT-based models (FLUX, SD 3.5) is the **VAE bottleneck**: all models encode the image to latent space before any architecture-specific processing. An immunization effective at corrupting VAE-encoded representations should generalize across model families.

We implement multi-model training by routing each training step to a randomly selected attack model:

$$\text{attack} \sim \text{Categorical}\left(\{(\text{SD1.5}, p_{sd}), (\text{FLUX.1-schnell}, p_{flux}), (\text{SD3.5}, p_{sd3})\}\right)$$

For the H1a configuration: $p_{sd} = 0.25$, $p_{flux} = 0.75$. The FLUX weight is higher because (a) FLUX is the primary purification threat and (b) FLUX steps are harder (higher gradient norm), providing stronger supervision.

**Architecture differences handled:**
- *SD 1.5*: 4-channel VAE, UNet backbone, 50-step DDPM (we use 4 accelerated steps)  
- *FLUX.1-schnell*: 16-channel VAE, 12B DiT, distilled 4-step flow matching, dual-stream CLIP+T5 conditioning, `shift_factor` and `scaling_factor` from config
- *SD 3.5*: 16-channel VAE, MM-DiT architecture, 7.0 CFG, tri-encoder (CLIP-L + CLIP-G + T5-XXL)

All attack wrappers expose an identical `attack(prompt, masked_image, mask, height, width, num_inference_steps, batch_size)` interface. Gradient checkpointing is enabled on all transformer/attention layers to fit the 12B FLUX model alongside the immunizer during backpropagation.

**Training observation:** Multi-model training produces a bimodal epoch-level loss distribution: FLUX steps yield Loss1 ≈ 0.8–1.3 (harder adversary), while SD1.5 steps yield Loss1 ≈ 0.05–0.15. This alternating curriculum — analogous to GAN training dynamics — stabilizes optimization: the SD1.5 signal guides the immunizer toward generalizable perturbations while FLUX epochs push toward stronger disruption.

### 3.3 Patch-Based 1088×1088 Inference (H2)

NestedUNet is fully convolutional with no fixed positional encodings, enabling inference on arbitrary-resolution inputs. We exploit this to immunize 1088×1088 images using overlapping 512×512 patches with Gaussian-weighted blending.

**Patch extraction.** Given an input image $x \in \mathbb{R}^{3 \times 1088 \times 1088}$ and stride $s$, we extract all $512 \times 512$ patches $\{x_{ij}\}$. With $s = 256$: we get patches at offsets $(0, 0), (0, 256), \ldots, (576, 576)$ — 9 patches in a $3 \times 3$ grid, with each patch overlapping its neighbors by 256 pixels.

**Perturbation blending.** For each patch, we compute $\delta_{ij} = f_\theta(x_{ij})$. We reconstruct the full-resolution perturbation by accumulating patches weighted by a 2D Gaussian window $G$ (matching the SD upscaler blending function):
$$\delta = \frac{\sum_{ij} G_{ij} \odot \text{place}(\delta_{ij})}{\sum_{ij} G_{ij}}$$

**Perturbation accumulation effect (key result).** At stride $s = 256$, the center of the 1088×1088 image falls within approximately 4 overlapping patches. The Gaussian-weighted blending assigns each patch's perturbation proportionally to its weight, but the *unnormalized accumulation before weighting* is denser at overlap regions. When an adversary subsequently downscales the 1088px image to 512px for editing, the perturbation density at the downscaled center is higher than what a single 512px immunization would provide — explaining the 1.60× EDR improvement observed.

We choose $s = 256$ (50% overlap) based on two criteria:
1. *Seam artifact analysis*: $s = 512$ (no overlap) produces visible seam_ratio = 2.38 (threshold 1.2); $s = 384$ produces 1.28 (marginal); $s = 256$ produces 1.05 (pass).
2. *EDR performance*: EDR monotonically improves with decreasing stride: 0.300 (no overlap) → 0.330 (25%) → **0.400** (50%), confirming the accumulation hypothesis.

### 3.4 JPEG-Augmented Training (H7)

Social media platforms apply JPEG compression to all uploads (Instagram ≈ q=75, Twitter/X ≈ q=70 equivalent). Standard Lp-bounded perturbations are largely destroyed at these quality levels because they concentrate energy in high-frequency DCT bands that JPEG quantization eliminates.

We address this with **Straight-Through Estimator (STE) JPEG augmentation**: with probability $p_{jpeg}$ per training step, the immunized image $\tilde{x}$ is JPEG-compressed before the attack forward pass:

$$\tilde{x}^{\text{compressed}} = \text{JPEG}_q(\tilde{x}), \quad q \sim U[70, 85]$$

**Forward pass:** $\text{Edit}(\tilde{x}^{\text{compressed}}, m, p)$ — the attack model sees the compressed image, as it would in deployment.

**Backward pass (STE):** Gradients flow as if JPEG were the identity:
$$\frac{\partial \mathcal{L}}{\partial \tilde{x}} := \frac{\partial \mathcal{L}}{\partial \tilde{x}^{\text{compressed}}}$$

This is a valid STE because JPEG's DCT→quantize→dequantize→IDCT pipeline is piecewise constant and cannot be differentiated directly, but the gradient with respect to the input is bounded and informative about which directions increase loss. Over many iterations, the gradient signal drives perturbation energy into quantization-table survivor bands at the target quality level.

**Configuration:** $p_{jpeg} = 0.5$ (50% of steps use JPEG augmentation), $q \sim U[70, 85]$ (spanning Instagram and Twitter levels). We sample $q$ uniformly per step rather than fixing it, forcing generalization across the full quality range.

**Novelty:** IDProtector [CHEN2024] explicitly avoids STE JPEG training and uses Gaussian noise as a proxy, testing only at q=85. DiffVax++ is the first immunization method to apply STE JPEG training targeting q=70–75, directly closing the social media deployment gap.

### 3.5 VAE Feature Loss (H4, ablation)

As an optional extension, we add a VAE feature-space loss term that explicitly maximizes the distance between original and immunized image encodings in latent space:
$$\mathcal{L}_{vae} = -\|\text{VAE}(\tilde{x}) - \text{VAE}(x)\|^2$$

A negative sign because we want to *maximize* latent distance. This loss is compatible with multi-model training since any VAE-similar encoder will be disrupted by large latent-space perturbations. We ablate this loss in Section 4.5.

### 3.6 Training Details

- **Immunizer**: NestedUNet (UNet++, 9.2M params [ZHOU2018]), fully convolutional
- **Optimizer**: Adam, lr=5e-6
- **Training**: 8,000 gradient steps (batch_size=1), max_steps enforced via config
- **Resolution**: 512×512 during training; patch-based at inference for 1088×1088
- **Attack wrappers**: 4 denoising steps for all models during training (FLUX default; truncated for SD1.5/SD3.5 for training speed)
- **Gradient checkpointing**: enabled on FLUX transformer (12B params, ~50GB activation memory without checkpointing)
- **Loss normalization**: divide by spatial dimension $H$ (not hardcoded /512)
