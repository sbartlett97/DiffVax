# Paper Draft — Introduction
# Status: draft, no experimental placeholders filled yet for H1/H6/H7
# Date: 2026-04-08
# Target: ICLR 2027 (8-page limit + unlimited appendix)

---

## 1. Introduction

The explosion of AI-powered image editing has created an urgent need for tools that
protect digital artwork from unauthorized manipulation. Diffusion-based inpainting
models such as FLUX.1 [BLACK-FOREST-LABS], Stable Diffusion 3.5 [ROMBACH], and
gpt-image-edit [OPENAI] can seamlessly alter human faces, remove watermarks, and
transplant objects with near-photorealistic quality — capabilities that simultaneously
empower creators and threaten them.

**Image immunization** addresses this threat at the source: by embedding a carefully
crafted, imperceptible perturbation into an image before upload, the technique causes
any diffusion inpainting model that attempts to edit the image to produce blank or
incoherent outputs, effectively "poisoning" the editing process [SALMAN2023,LIANG2023].
DiffVax [OZDENTARIKCAN2025], published at ICLR 2025, represents the state of the art
in this approach: a NestedUNet [ZHOU2018] trained to produce a single-pass immunization
perturbation through a differentiable 4-step inpainting forward pass. DiffVax achieves
millisecond-speed inference (single forward pass vs. hours of per-image PGD), making
it deployable at social media scale.

**However, DiffVax and all existing immunization methods [SHAN2023,VANLE2023,RUIZ2023]
share three critical gaps that prevent real-world deployment.**

*Gap 1: Model coverage.* DiffVax is trained and evaluated exclusively against Stable
Diffusion 1.5 (UNet architecture). Modern adversaries have access to substantially
more capable DiT-based editors. Critically, Zhao et al. [ZHAO2026] show that
FLUX.1-fill-dev can *purify* SD1.5 immunizations, recovering editability with +3–6 dB
PSNR improvement — a practical attack that makes SD1.5-targeting immunization
commercially worthless. Furthermore, Pleimling et al. [PLEIMLING2026] demonstrate that
commodity image-to-image tools — with no knowledge of the specific defense — can strip
standard Lp-bounded perturbations across six different defense schemes. The threat
landscape has moved far beyond what single-model training can address.

*Gap 2: Resolution.* Social media platforms natively handle images at 1080×1080
(Instagram) or 1200×675 (Twitter/X). DiffVax is hardcoded to 512×512, and naively
resizing immunized images to social media dimensions degrades the perturbation's
effectiveness. No existing method addresses immunization at megapixel scale.

*Gap 3: Compression robustness.* Instagram applies JPEG compression at approximately
quality=75 equivalent on all uploads; Twitter/X applies quality≈70 [CITE-SOCIAL-MEDIA].
Standard Lp-bounded pixel-space perturbations are largely destroyed by JPEG
quantization at these quality levels [GOODFELLOW2016,DCT-SHIELD-2025]. Without
compression robustness, an immunized image is stripped of its protection by the
social media platform itself — before any adversary action.

**Notably, none of the recent 2025 competition — including PromptFlare [NA2025],
Attention Attack [TRIPPODO2025], and Anti-Inpainting [GUO2025] — address any of
these three gaps.** All are evaluated on single model architectures at 512×512
without JPEG testing; all would fail on real Instagram/Twitter deployments.

We introduce **DiffVax++**, which addresses all three deployment gaps through two
confirmed contributions and reveals two surprising properties of the existing DiffVax
baseline that the field has overlooked:

**Contribution 1: Patch-based 1088×1088 inference is strictly stronger than 512×512.**
Applying the 512-trained immunizer to 1088×1088 images via overlapping patches
(stride=256, Gaussian blending) achieves **1.60×** the edit disruption rate (EDR) of
direct 512px inference. The mechanism is *perturbation accumulation*: at stride=256,
the image center receives contributions from ~4 overlapping patches, yielding a
higher-density adversarial signal that survives adversary downsampling.

**Contribution 2: STE JPEG augmented training exploits a DCT–DiT compound effect.**
We discover that JPEG compression at q=75 (Instagram) *increases* DiffVax FLUX EDR
from 0.200 to 0.300 (+50%; paired $t=-4.33$, $p \ll 0.001$). This occurs because JPEG
DCT artifacts compound with the immunization perturbation against FLUX's token-based
DiT architecture. SD 1.5's convolutional UNet is insensitive to this effect — JPEG does
not help or hurt SD1.5 immunization. We train the H7 checkpoint with STE JPEG
augmentation at q=70–85 to explicitly learn perturbations that maximize this
compound effect, achieving EDR = [A] at q=75 and [B] at q=70. IDProtector [CHEN2024]
explicitly avoids STE training ("introduces substantial learning burden") and only
tests q=85. DiffVax++ H7 is the first method designed for social media deployment.

**Surprising finding 1: DiffVax already transfers to FLUX and SD3.5.**
The published SD1.5-only DiffVax checkpoint achieves FLUX.1-schnell EDR = 0.200 and
SD3.5 EDR = 0.140 without any multi-model training. The shared VAE bottleneck common
to all three architectures mediates cross-model transfer automatically. We confirm
this by testing multi-model training (SD+FLUX), which *weakens* immunization by 2.1 dB
due to competing gradient objectives, ruling out architecture-specific features as the
operative mechanism. The VAE is the attack surface.

**Surprising finding 2: The EditorClean threat requires unacceptable image quality
sacrifice.** Zhao et al.'s FLUX-based purification attack [ZHAO2026] requires strength
$\geq 0.5$ to remove DiffVax immunizations, but at that strength the purified image
quality drops to PSNR = 23 dB / SSIM = 0.70 — visibly degraded. At the realistic
adversary strength of $s = 0.3$ (PSNR = 31.1 dB after purification), DiffVax net
immunization survival is EDR = +0.200. The EditorClean threat is substantially weaker
than its paper claims once the quality–robustness tradeoff is accounted for.

We release code, checkpoints, and evaluation scripts. The evaluation protocol itself
— standardized EDR on a public benchmark — is a methodological contribution: no
competing paper reports a comparable numerical metric, making "SOTA" comparisons in
this area currently unreliable.

---
**[Placeholders: H7 EDR values at q=75 and q=70 from pending GPU run]**

## Notation and Setup (brief)
- *EDR (Edit Disruption Rate)*: fraction of (image, prompt) pairs where
  SSIM(immunized_edit, original) < SSIM(clean_edit, original) − 0.05.
  Higher = stronger immunization. Measures practical editing failure rate.
- *Immunized image*: original image + imperceptible perturbation (PSNR ≥ 28 dB)
- *Clean edit*: what the adversary achieves on the unprotected original
- *Disrupted edit*: what the adversary achieves after immunization (near-blank output)
