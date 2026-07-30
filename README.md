## DiffVax: Optimization-Free Image Immunization Against Diffusion-Based Editing (ICLR 2026)

[![arXiv](https://img.shields.io/badge/arXiv-2411.17957-b31b1b.svg)](https://arxiv.org/pdf/2411.17957)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://diffvax.github.io/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/ozdentarikcan/DiffVaxDataset)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=ozdentarikcan.DiffVax)

[Tarik Can Ozden](https://ozdentarikcan.github.io/)\*,
[Ozgur Kara](https://karaozgur.com/)\*,
[Oguzhan Akcin](https://scholar.google.com/citations?user=2elIEXoAAAAJ&hl=en),
[Kerem Zaman](https://keremzaman.com/),
[Shashank Srivastava](https://scholar.google.com/citations?user=-vKI5s0AAAAJ&hl=en),
[Sandeep P. Chinchali](https://scholar.google.com/citations?user=262ASa4AAAAJ&hl=en),
[James M. Rehg](https://rehg.org/)

\* Equal Contribution

![motivation](assets/diffvax_motivation.png)

> **About this fork.** This repository extends the original DiffVax release with
> a research track aimed at protecting images against modern hybrid
> diffusion/transformer ("DiT") editors — SD3/SD3.5, FLUX.1/FLUX.2, and (via
> proxy losses) closed-source tools like DALL·E 3 and Gemini image editing —
> in addition to the original SD 1.5 inpainting target, and at scaling
> training to higher resolutions. The core single-surrogate SD 1.5 pipeline
> (`app.py`, `scripts/demo.py`, `configs/train.yml`) is unchanged from
> upstream. The multi-surrogate / DiT-targeting features described below are
> **implemented and unit-tested (CPU) but not yet validated on GPU** — see
> [Project Status](#project-status) before relying on their protection claims.

## Abstract
<b>TL; DR:</b> DiffVax is a scalable, lightweight, and optimization-free image immunization framework designed to protect images and videos from diffusion-based editing.

<details><summary>Click for the full abstract</summary>


> Current image immunization defense techniques against diffusion-based editing embed imperceptible noise into target images to disrupt editing models. However, these methods face scalability challenges, as they require time-consuming optimization for each image separately, taking hours for small batches. To address these challenges, we introduce DiffVax, a scalable, lightweight, and optimization-free framework for image immunization, specifically designed to prevent diffusion-based editing. Our approach enables effective generalization to unseen content, reducing computational costs and cutting immunization time from days to milliseconds, achieving a speedup of 250,000×. This is achieved through a loss term that ensures the failure of editing attempts and the imperceptibility of the perturbations. Extensive qualitative and quantitative results demonstrate that our model is scalable, optimization-free, adaptable to various diffusion-based editing tools, robust against counter-attacks, and, for the first time, effectively protects video content from editing.


</details>

## Installation

```bash
# Clone the repository
git clone https://github.com/ozdentarikcan/DiffVax.git
cd DiffVax

# Create conda environment (recommended)
conda create -n diffvax python=3.12 -y
conda activate diffvax

# Install dependencies
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

`requirements.txt` includes the dependencies needed for the full v2/v3
research pipeline: `kornia` (differentiable JPEG for EoT augmentation —
required whenever a config sets `eot.p_jpeg > 0`, fails loudly if missing)
and `open-clip-torch` (CLIP-based losses/metrics). For development:

```bash
pip install pytest
python -m pytest tests/     # CPU-only; all but one GradScaler test run without a GPU
```

FLUX support requires a `diffusers` build with `Flux2KleinPipeline`; install
from source (`pip install git+https://github.com/huggingface/diffusers.git`)
if your installed release predates it. SD3.5 and FLUX.2 Klein are gated
models on Hugging Face — accept the license on the model pages and set
`HF_TOKEN` before training against them.

## Dataset

The DiffVax dataset is hosted on Hugging Face: [`ozdentarikcan/DiffVaxDataset`](https://huggingface.co/datasets/ozdentarikcan/DiffVaxDataset)

Download it with the provided script:

```bash
python scripts/download_dataset.py
```

This places the dataset in `data/` with the following structure:

```
data/
├── train/
│   ├── images/          # Training images (512x512 PNG)
│   ├── masks/           # Corresponding masks
│   └── metadata.jsonl   # Image-prompt pairs
└── validation/
    ├── images/
    ├── masks/
    └── metadata.jsonl
```

The training and demo scripts will also auto-download the dataset on first run if it's not present.

## Project Structure

```
DiffVax/
├── app.py                              # Gradio web demo (SD 1.5 only)
├── src/diffvax/                        # Main package
│   ├── model.py                        # NestedUNet (UNet++), GroupNorm, configurable filters (H6)
│   ├── utils.py                        # Image I/O, data loading, seeding
│   ├── attack_base.py                  # BaseAttack interface shared by all surrogates
│   ├── attack.py                       # SD 1.5 inpainting surrogate (4-ch VAE, UNet)
│   ├── sd3_attack.py                   # SD3 / SD3.5 surrogate (16-ch VAE, MM-DiT)
│   ├── flux_attack.py                  # FLUX.2 Klein surrogate (16-ch VAE, single-stream DiT)
│   ├── attack_manager.py               # Multi-surrogate selection + adaptive ensemble weighting
│   ├── curriculum.py                   # Multi-resolution training curriculum (512→768→1024→1088)
│   ├── eot.py                          # Differentiable EoT augmentation (JPEG/resize/blur/noise)
│   ├── reporter.py                     # JSON training log + webhook notifications
│   ├── losses/
│   │   ├── clip_loss.py                # CLIP feature/semantic disruption (architecture-agnostic)
│   │   ├── spectral_loss.py            # Frequency-domain perturbation concentration
│   │   ├── latent_loss.py              # VAE latent-space disruption (16-ch VAE targets)
│   │   ├── attention_loss.py           # Cross-attention entropy disruption for DiT models
│   │   └── flat_minima.py              # Sharpness-aware gradient regularization
│   ├── immunization/
│   │   ├── diffvax_immunization.py     # DiffVax training loop (multi-surrogate, all loss terms)
│   │   ├── photoguard_immunization.py  # PhotoGuard baseline (PGD encoder attack)
│   │   └── diffusionguard_immunization.py  # DiffusionGuard baseline (PGD noise maximization)
│   └── metrics/                        # Image quality metrics
│       ├── base.py, factory.py
│       └── psnr.py, ssim.py, fsim.py, clip_score.py
│
├── scripts/
│   ├── train.py                        # Train the immunization model (single or multi-surrogate)
│   ├── demo.py                         # End-to-end demo with comparison output
│   ├── evaluate.py                     # Calculate image quality metrics
│   ├── eval_multimodel.py              # Evaluate a checkpoint against a zoo of editing models
│   ├── jpeg_robustness.py              # Measure protection retention under JPEG recompression
│   ├── compare_baselines.py            # Multi-image baseline comparison figure
│   └── download_dataset.py             # Download dataset from Hugging Face
│
├── tests/
│   ├── test_gradient_flow.py           # Gradient-flow/loss-signal unit tests (stub denoising loop)
│   ├── test_attack_gradient_flow.py    # Same properties through the REAL SD3/FLUX attack code
│   └── test_training_smoke.py          # Full training-loop smoke test on CPU
│
├── notebooks/
│   ├── diffvax_demo.ipynb              # Interactive demo notebook
│   └── diffvax_comparison.ipynb        # Multi-method comparison notebook
│
├── configs/
│   ├── train.yml                       # v2 baseline: SD 1.5 only, all new phases off (== v1)
│   ├── sd_only.yml                     # Explicit v1-equivalent single-surrogate config
│   ├── sd3_only.yml                    # SD3/SD3.5 surrogate only (ablation)
│   ├── eot_clip.yml                    # SD 1.5 + EoT + CLIP loss (preprocessing robustness)
│   ├── dual_surrogate.yml              # SD 1.5 + FLUX + adaptive ensemble weighting
│   ├── full_v2.yml                     # All three surrogates + all seven v2 phases
│   ├── research_v3.yml                 # v3 hypothesis bundle (H1–H8) at 512px — start here
│   ├── train_1088_v3.yml               # v3 production config, 1024→1088px curriculum
│   ├── train_multi.yml                 # Multi-surrogate 1088px fine-tuning (stage 2)
│   └── resume_finetune.yml             # Resume/fine-tune an existing checkpoint at higher res
│
├── research/                           # Research log, hypotheses, findings, GPU validation plan
│   ├── research-log.md
│   ├── findings.md
│   ├── research-state.yaml
│   ├── gpu-validation-runbook.md
│   └── experiments/H1..H8-*/protocol.md
│
├── checkpoints/
│   └── diffvax_trained.pth             # Pre-trained model weights (v1, SD 1.5)
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Quick Start

### Web demo (Gradio)

Launch the interactive web interface:

```bash
python app.py
```

This starts a Gradio app at `http://localhost:7860` where you can upload an image and mask, enter an editing prompt, and see how DiffVax protects the image.

### Run the demo

The demo script downloads the dataset (if needed), loads the pre-trained checkpoint, immunizes a validation image, runs the same edit on both original and immunized versions, and saves a side-by-side comparison.

```bash
python scripts/demo.py
```

Options:

```bash
# Use a different validation image and prompt
python scripts/demo.py --image-index 2 --edit-prompt "a watercolor painting"

# Run on a specific GPU
CUDA_VISIBLE_DEVICES=4 python scripts/demo.py

# Headless server (no display)
python scripts/demo.py --no-display --save-dir outputs/my_demo
```

If the dataset is not found locally, the demo generates a synthetic sample image automatically. Output files are saved to `outputs/demo/` by default:
- `<name>_original.png` — the original image
- `<name>_immunized.png` — the DiffVax-protected image
- `<name>_edited_original.png` — inpainting edit on the original
- `<name>_edited_immunized.png` — inpainting edit on the immunized image (edit should be disrupted)
- `<name>_comparison.png` — 2x2 comparison grid

You can also provide your own image and mask directly:

```bash
python scripts/demo.py --image photo.png --mask mask.png --edit-prompt "a cat sitting"
```

### Baseline Comparisons

DiffVax includes implementations of two prior-work baselines for side-by-side comparison:

- **PhotoGuard** (Salman et al., ICML 2023) — PGD encoder attack that forces `VAE.encode(x+δ)` toward a target latent (1000 L∞-PGD steps, ε=0.06), plus a diffusion attack that backpropagates through the full inpainting pipeline (200 L2-PGD steps, ε=16, 10 gradient averaging reps).
- **DiffusionGuard** (Li et al., ICLR 2025) — PGD optimization that maximizes the UNet's noise prediction norm at the highest timestep, with contour-based mask augmentation for robustness (800 L∞-PGD steps, ε=16/255).

Unlike these methods, DiffVax uses a trained NestedUNet for single-pass inference (no per-image optimization).

**Generate a multi-image comparison figure:**

```bash
python scripts/compare_baselines.py
python scripts/compare_baselines.py --images 1 5 9 29 33
```

**Run the demo with all methods compared:**

```bash
python scripts/demo.py --run-baselines --no-display
```

This produces the standard DiffVax 2x2 grid plus a 1x5 baseline comparison figure:
`Original | Edited (No Defense) | Edited (PhotoGuard) | Edited (DiffusionGuard) | Edited (DiffVax)`

You can tune PGD steps:

```bash
python scripts/demo.py --run-baselines --pg-steps 200 --dg-steps 100
```

### Train a new model

**Before training**, download the DiffVax dataset:

```bash
python scripts/download_dataset.py
```

This places the dataset in `data/` with the structure shown in the [Dataset](#dataset) section above. Alternatively, the training script will automatically download the dataset from Hugging Face Hub on first run if it's not present.

```bash
python scripts/train.py \
    --config configs/train.yml \
    --data-dir data \
    --output-dir outputs
```

`configs/train.yml` reproduces the original v1 behaviour exactly (SD 1.5
only, all v2/v3 phases disabled). To resume from a checkpoint, pass
`--checkpoint path/to/model.pth` (overrides `load_path` in the config).

### Multi-surrogate training against SD3/FLUX

To train against SD3.5 and/or FLUX.2 Klein in addition to (or instead of) SD
1.5, use one of the multi-surrogate configs and set `sd3_model_link` /
`flux_model_link` with nonzero probabilities that sum to 1.0 with
`sd_probability`:

```bash
# Start here: all H1–H8 research hypotheses, 512px, fits a 24 GB GPU
python scripts/train.py --config configs/research_v3.yml

# Full three-surrogate config (needs ~40 GB for SD3.5 peak; see Requirements below)
python scripts/train.py --config configs/full_v2.yml

# Stage 2: fine-tune a 512px checkpoint up through 1024→1088px
python scripts/train.py --config configs/train_1088_v3.yml \
    --checkpoint outputs/models/1/..._final.pth
```

Only one attack surrogate is resident on GPU at a time — `AttackModelManager`
swaps models in/out of VRAM as training randomly selects between them each
batch, and offloads each surrogate's text encoder(s) to CPU during the
backward pass, so peak VRAM is bounded by whichever single surrogate is
currently active, not the sum of all configured surrogates.

## Configuration

Every config in `configs/` is a full standalone training run; see the header
comment in each file for its intended use case and VRAM estimate. Core keys:

| Parameter | Default (`train.yml`) | Description |
|-----------|---------|-------------|
| `iter_num` | 1000000 | Number of training epochs |
| `learning_rate` | 0.00001 | Adam optimizer learning rate |
| `batch_size` | 5 | Training batch size |
| `alpha` | 4 | Weight for perturbation imperceptibility loss (`loss2`) |
| `attack_model_link` | `runwayml/stable-diffusion-inpainting` | SD 1.5 surrogate; set `sd_probability: 0` to disable |
| `sd3_model_link` / `sd3_probability` | `null` / `0.0` | SD3/SD3.5 surrogate (16-ch VAE, MM-DiT) |
| `flux_model_link` / `flux_probability` | `null` / `0.0` | FLUX.2 Klein surrogate (16-ch VAE, single-stream DiT) |
| `train_all` | true | Train on all images (false = use `image_index_list`) |

Every research phase below is opt-in via an `enabled: true/false` flag in its
own config section; leaving all of them `false` reproduces v1 behaviour
identically:

| Section | Purpose |
|---|---|
| `eot` | Differentiable JPEG/resize/blur/noise augmentation so the perturbation survives real-world recompression pipelines. `p_jpeg > 0` requires `kornia`. |
| `clip_loss` | CLIP feature + prompt-alignment disruption — the one representation shared across virtually all image generators, including closed-source tools. |
| `spectral_loss` | Penalizes low-frequency perturbation energy, concentrating the perturbation in less-visible frequency bands. |
| `latent_loss` | Pushes the adversarial image's VAE latent away from the clean image's latent — a direct attack on 16-ch VAE (SD3/FLUX) models. |
| `attention_loss` | Maximizes attention entropy in DiT transformer blocks to disrupt cross-attention context propagation (targets `target_blocks: "middle"` per DeContext). |
| `flat_minima` | Gradient-norm/SAM-style sharpness penalty intended to improve cross-model transfer. |
| `curriculum` | Multi-stage resolution schedule (e.g. 512→768→1024→1088px) with per-stage batch size. |
| `adaptive_ensemble` | Reweights surrogate-selection probability by gradient disparity instead of a fixed random draw. |
| `sd3_attack` / `flux_attack` | Per-surrogate `gradient_timestep_fraction` (partial-timestep backprop for VRAM reduction), `token_gradient_regularization` (TGR), and `use_gradient_checkpointing`. |

## Architecture

**NestedUNet (UNet++)** — a hierarchical encoder-decoder with dense nested skip connections:

- **Input**: 3-channel RGB image, any resolution that's a multiple of 16
- **Encoder**: 5 levels, configurable filter depths (`nb_filter`), default `[32, 64, 128, 256, 512]`
- **Decoder**: dense skip connections at every level (not just symmetric pairs), GroupNorm (batch-size-independent, safe at `batch_size=1`)
- **Output**: 3-channel perturbation map, applied additively to the input image
- **Parameters**: ~9M (default filters); ~37M with the larger `[64,128,256,512,1024]` variant

The perturbation is applied additively (full image for the multi-surrogate
pipeline; masked-region-aware for the original SD 1.5 inpainting path), and
the resulting image is clamped to the valid pixel range.

**Attack surrogates** (`src/diffvax/attack_base.py::BaseAttack` implementations):

| Surrogate | VAE | Backbone | Native resolution |
|---|---|---|---|
| `Attack` (SD 1.5 inpainting) | 4-channel | UNet | 512px |
| `SD3Attack` (SD3 / SD3.5) | 16-channel | MM-DiT (joint bidirectional attention) | 1024px |
| `FluxAttack` (FLUX.2 Klein) | 16-channel | Single-stream DiT | 1024px |

Each surrogate's `attack()` is fully differentiable end to end (VAE encode →
denoising loop → VAE decode) so gradients flow from the loss back through
the frozen diffusion model to the perturbation network. `gradient_timestep_fraction`
runs only a fraction of denoising steps' transformer forward with gradients
enabled (the rest run the transformer under `no_grad`) while the scheduler's
additive integration step always stays differentiable, trading VRAM for
signal from later timesteps without severing the gradient chain.

## Evaluation

```bash
# Standard image-quality + protection metrics against SD 1.5
python scripts/evaluate.py --checkpoint outputs/models/1/..._final.pth

# Evaluate a checkpoint against a zoo of editing models (SD 1.5, SD3.5, FLUX, ...)
python scripts/eval_multimodel.py --checkpoint ckpt.pth
python scripts/eval_multimodel.py --checkpoint ckpt.pth --models "SD 1.5 Inpainting" --max-images 2

# Measure protection retention under JPEG recompression (EoT robustness check)
python scripts/jpeg_robustness.py --checkpoint ckpt.pth --images data/validation/images
```

`eval_multimodel.py` reports, per model, PSNR/SSIM/FSIM between original and
immunized images, plus `clip_delta` (CLIP-prompt alignment on the unprotected
edit minus the protected edit) — positive `clip_delta` indicates the edit was
disrupted.

## Testing

The training-signal correctness of the multi-surrogate pipeline is covered by
a CPU-only pytest suite (no GPU required, real attack code paths driven by
lightweight fake diffusers pipelines — see `research/findings.md` for what
this does and doesn't prove):

```bash
pip install pytest
python -m pytest tests/
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU with 8 GB+ VRAM for the original SD 1.5-only pipeline
  (16 GB+ recommended for training)
- For multi-surrogate SD3/FLUX training: a 24 GB GPU covers `research_v3.yml`
  and `train_1088_v3.yml` (design estimates, not yet measured on real
  hardware — see [Project Status](#project-status)); `full_v2.yml` with all
  three surrogates recommends 40 GB for SD3.5's peak. Budget ~64 GB system
  RAM — offloaded text encoders (SD3.5's T5-XXL, FLUX's Qwen3) and inactive
  surrogates live in host memory between batches.
- ~5 GB disk space for the Stable Diffusion Inpainting model, more for
  SD3.5/FLUX.2 Klein weights if those surrogates are enabled (downloaded
  automatically; SD3.5 and FLUX.2 Klein require accepting their gated
  license and an `HF_TOKEN`)

## Project Status

The multi-surrogate / DiT-targeting extensions in this repository (SD3/FLUX
support, EoT, CLIP/spectral/latent/attention losses, curriculum, TGR) have
been audited end to end: gradient-path integrity, loss sign conventions, and
end-to-end learning are all verified by the CPU test suite in `tests/`
against the real attack code. What remains is empirical GPU validation —
actual protection rates, ablations of each loss term, and 1088px production
training. See `research/findings.md` for the full verification writeup and
`research/gpu-validation-runbook.md` for the staged validation plan. Until
that validation runs, treat protection-rate claims for SD3/FLUX/closed-source
models as unverified; the original SD 1.5 pipeline (`app.py`,
`scripts/demo.py`, `configs/train.yml`, the pretrained
`checkpoints/diffvax_trained.pth`) reflects the published, evaluated paper
result and is unaffected by this research track.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/diffvax_demo.ipynb` | Interactive demo: immunize an image and compare edits in a 1x3 grid |
| `notebooks/diffvax_comparison.ipynb` | Compare DiffVax against PhotoGuard and DiffusionGuard side-by-side |

## Citation 

```
@inproceedings{ozden2026diffvax,
  title={DiffVax: Optimization-Free Image Immunization Against Diffusion-Based Editing},
  author={Ozden, Tarik Can and Kara, Ozgur and Akcin, Oguzhan and Zaman, Kerem and Srivastava, Shashank and Chinchali, Sandeep P and Rehg, James M},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},      
}

``` 

## License

MIT License. See [LICENSE](LICENSE) for details.
