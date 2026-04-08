# Off-The-Shelf Image-to-Image Models Are All You Need To Defeat Image Protection Schemes
**arXiv**: 2602.22197 (February 25, 2026)
**Authors**: Xavier Pleimling, Sifat Muhammad Abdullah, Gunjan Balde, Peng Gao, Mainack Mondal, Murtuza Jadliwala, Bimal Viswanath

## Key Finding
Standard commodity image-to-image generative AI tools (style transfer, enhancement models) can act as denoising/purification mechanisms, stripping protective perturbations across six diverse defense schemes — without any knowledge of the specific defense being used. Current Lp-bounded pixel-space protections offer insufficient security against commodity transforms.

## Why This Matters for DiffVax++
- **Directly threatens standard DiffVax baseline**: If any style-transfer or SR model can purify the perturbation, single-model training is inadequate
- **Strongly motivates H7**: STE JPEG augmentation trains perturbations into DCT bands that survive image transformations (JPEG is the most common transform applied). A perturbation that survives q=70-75 JPEG also survives many commodity tools that internally compress
- **Motivates multi-model training (H1)**: The paper shows attacks work because of model mismatch. Training against multiple architectures eliminates the mismatch assumption
- **New related work for H6 section**: Purification threat is not just EditorClean — commodity tools also threaten. DiffVax++ must be robust to the broader class

## Positioning in Paper
Add to threat model section in Introduction: "Adversaries need not train specialized purifiers — commodity image-to-image tools (Pleimling et al., 2026) can strip standard Lp perturbations." This makes H7 even more motivated.
