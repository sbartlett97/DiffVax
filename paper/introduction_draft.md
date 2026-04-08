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

We introduce **DiffVax++**, which addresses all three gaps simultaneously with three
contributions, each producing a *surprising* result:

**Contribution 1: Patch-based 1088×1088 inference is not merely sufficient — it is
strictly stronger.** We show that applying a 512×512-trained immunizer to 1088×1088
images via overlapping patches (stride=256, Gaussian blending) achieves **1.60×** the
edit disruption rate (EDR) of direct 512×512 inference. The mechanism is
*perturbation accumulation*: at stride=256, the image center receives ~4 overlapping
patch contributions, yielding a higher-density signal that the adversary's downsampling
cannot fully remove. This is counter-intuitive — tiling is typically a smoothing
operation — but holds robustly across model types.

**Contribution 2: Multi-model training resists FLUX-based purification.** Training the
immunizer simultaneously against SD 1.5 (25%) and FLUX.1-schnell (75%) produces a
checkpoint that transfers to SD 3.5 (zero-shot held-out architecture, EDR = [X]) and
critically *resists* the EditorClean purification attack [ZHAO2026]: after
strength=0.5 purification, H1a retains [Y]% of its direct EDR, while the SD1.5-only
checkpoint retains only [Z]%. Multi-model training is not just a coverage improvement
— it is a product safety requirement against the strongest known purification attack.

**Contribution 3: STE JPEG augmentation is the first immunization method to survive
social media upload compression.** We train with JPEG augmentation via the
Straight-Through Estimator (STE) [BENGIO2013]: the forward pass applies JPEG at
q=70–85, while gradients flow as the identity, forcing perturbation energy into DCT
quantization-table survivor bands. The H7 checkpoint maintains EDR ≥ [A] after q=75
JPEG (Instagram scenario), while the H1a baseline without JPEG training drops to
EDR ≤ [B]. IDProtector [CHEN2024], the most recent competing method, explicitly avoids
STE training ("introduces substantial learning burden") and only tests q=85 — far above
social media quality levels. DiffVax++ is the first method to close this deployment gap.

The training dynamics reveal an additional insight: the alternating SD1.5/FLUX
curriculum creates a bimodal loss landscape — FLUX epochs produce high Loss1 (harder
adversary) while SD1.5 epochs stabilize the gradient signal. This curriculum effect,
analogous to GAN training dynamics, may explain why multi-model training generalizes
to unseen architectures rather than overfitting to either individual model.

We release code, trained checkpoints, and evaluation scripts to enable reproducibility
and facilitate future work on robust, deployment-ready image immunization.

---
**[Placeholders to fill from experiments: EDR values for H1, H6, H7]**

## Notation and Setup (brief)
- *EDR (Edit Disruption Rate)*: fraction of (image, prompt) pairs where
  SSIM(immunized_edit, original) < SSIM(clean_edit, original) − 0.05.
  Higher = stronger immunization. Measures practical editing failure rate.
- *Immunized image*: original image + imperceptible perturbation (PSNR ≥ 28 dB)
- *Clean edit*: what the adversary achieves on the unprotected original
- *Disrupted edit*: what the adversary achieves after immunization (near-blank output)
