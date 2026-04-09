# Literature: JPEG Artifacts + ViT/Transformer Sensitivity
# Source: Background agent search (2026-04-09)
# Relevance: Mechanistic support for JPEG paradox in DiffVax++

## Key Papers

### RSPC (CVPR 2023) — ViTs structurally vulnerable to patch corruptions
URL: https://openaccess.thecvf.com/content/CVPR2023/html/Guo_Improving_Robustness_of_Vision_Transformers_by_Reducing_Sensitivity_To_Patch_CVPR_2023_paper.html
Finding: Corrupting 10% of ViT patches causes severe accuracy drops via global attention propagation.
Relevance: FLUX's 2x2 patch tokenization → same structural vulnerability.
Key: cite as guo2023rspc

### RGB no more (arXiv:2211.16421, CVPR 2023) — ViTs accept DCT blocks natively
URL: https://arxiv.org/abs/2211.16421
Finding: ViTs are architecturally compatible with JPEG 8x8 DCT blocks; CNNs require surgery. 39.2% faster training, no accuracy loss when feeding DCT coefficients directly to patch embeddings.
Relevance: DCT block artifacts survive into FLUX token space intact; CNNs don't have this vulnerability.
Key: cite as li2022jpegvit

### Transform-Dependent Adversarial Attacks (arXiv:2406.08443)
URL: https://arxiv.org/abs/2406.08443
Finding: JPEG quality-factor as an explicit adversarial trigger, +17-31% over SOTA in black-box.
JPEG compression activates/amplifies adversarial perturbations.
Relevance: Directly parallels JPEG paradox (50% EDR increase at q=75).
Key: cite as chen2024transformadv

### JPEG Bypasses AI Editing Protections (arXiv:2304.02234)
URL: https://arxiv.org/abs/2304.02234
Finding: JPEG destroys pixel-space adversarial protections for UNet-based diffusion models.
Relevance: Baseline paper. Our FLUX finding INVERTS this baseline — novel.
Key: cite as zhao2023jpegbypass

### Pixel is a Barrier (arXiv:2404.13320)
URL: https://arxiv.org/abs/2404.13320
Finding: Pixel-space diffusion models robust; latent diffusion models vulnerable via encoder.
Relevance: If JPEG artifacts survive VAE encoding and align with FLUX patch boundaries, compound distortion path.
Key: cite as pierson2024pixel

### Adversarial Perturbations Cannot Protect Artists (ICLR 2025)
URL: https://proceedings.iclr.cc/paper_files/paper/2025/hash/af800ac69aea2fb5c4bcf8c6a5f3c79d-Abstract-Conference.html
Finding: Glaze, Mist, Anti-DreamBooth defeated by JPEG/upscaling. Community consensus.
Relevance: Our FLUX result is a notable exception. Architecture matters.
Key: cite as vanle2025cannot

### Artifacts and Attention Sinks (arXiv:2507.16018)
URL: https://arxiv.org/html/2507.16018
Finding: Artifact tokens and attention sinks in ViTs emerge from patch-boundary-aligned spatial patterns.
Relevance: JPEG block boundaries at FLUX patch edges -> artificial attention sinks.
Key: cite as luo2025artifacts

## Critical synthesis: NO prior papers study JPEG+FLUX adversarially
This is novel. Our JPEG paradox observation fills a gap in the literature.
