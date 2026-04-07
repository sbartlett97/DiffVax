# Research Log — DiffVax Extension

## 2026-04-07 — H7: JPEG-Robust Training + GPU Debugging

**Critical discovery**: Social media compression (Instagram q=75, Twitter q=70) defeats standard immunization perturbations. High-frequency DCT perturbations are the MOST vulnerable to JPEG — opposite of what H5 assumed.

**Actions**:
- Added H7 hypothesis: JPEG-robust training with STE gradient augmentation
- Implemented `src/diffvax/jpeg_augment.py` (STE JPEG for training, CPU/GPU compatible)
- Integrated JPEG augmentation into DiffVaxImmunization training loop (optional, `jpeg_augment_prob` config)
- Created `configs/train_multimodel_h7.yml` (SD 25% + FLUX 75% + JPEG aug 50%)
- Deprioritized H5 — needs redesign around JPEG-quantization-aware frequency constraints
- Saved DCT-Shield paper notes to `research/literature/dct_shield_2025.md`
- Fixed `torch.cuda.amp.GradScaler()` deprecation warning for PyTorch 2.4+

**GPU debugging (active)**:
- H2 OOM at 1088px: SD 1.5 self-attention = 136×136 tokens = 18× 512px; fixed by editing at 512px
- H1a TypeError `load_image() unexpected keyword 'resolution'`: GPU instance needs `git pull` (fix in commit 9c26387)

**Reference**: DCT-Shield (ICCV 2025, arXiv:2504.17894)

## 2026-04-07 — H2 CPU Prelim + API Verification + Data Pipeline Fixes

**Actions**:
- Added `--size` argument to `scripts/generate_masks.py` (supports 1088px training data generation for H3)
- Updated `patch_immunize.py` default stride from 384 → 256 (50% overlap required at 1088px)
- Validated `attack_flux.py` against current diffusers docs — all API calls confirmed correct

**H2 CPU Seam Analysis**:
- No overlap (stride=512): seam_ratio=2.377 → FAIL
- 25% overlap (stride=384): seam_ratio=1.275 → marginal
- 50% overlap (stride=256): seam_ratio=1.046 → PASS
- Product default: stride=256 for 1088×1088 immunization

**FLUX API — all confirmed correct**:
- `FluxInpaintPipeline` import path OK
- `img_ids`/`txt_ids` shape handling correct
- Timestep/1000 scaling correct
- `guidance` tensor vs None handling correct for distilled/non-distilled
- VAE `shift_factor`/`scaling_factor` read from `pipe.vae.config` — works across all FLUX variants

**State**: All CPU-computable work complete. Ready for GPU experiments. Run order: H2 → H1a → H6.

## 2026-04-07 — Inner Loop Setup

**Actions**:
- Validated NestedUNet is correctly resolution-agnostic (256×384, 512×512, 768×768 all work)
- Validated patch_immunize mask constraint: zero perturbation in edit region confirmed
- Implemented H4 VAE feature loss in DiffVaxImmunization training loop
- Implemented AttentionHookManager + FluxAttentionHookManager in losses.py
- Created train_multimodel_h4.yml config (vae_loss_beta=0.5)
- Created H6 evaluation script (purification robustness)
- Created comprehensive run_experiments.sh launcher
- Recreated train_multimodel.yml (was untracked/missing)

**Sanity checks passed**:
- All model imports OK
- MultiAttack sampling: 50/50 distribution confirmed with seed
- VAE feature loss returns negative scalar (correct)
- Empty attention map returns 0 loss (correct)

**Remaining work before GPU experiments**:
- data preparation script update for 1088px (generate_masks.py --size flag)
- Verify FLUX attack_flux.py forward pass shape math for non-512 resolutions

## 2026-04-06 — Bootstrap

**Action**: Set up research workspace. Surveyed existing codebase.

**Key findings from code review**:
- NestedUNet is fully convolutional — resolution-agnostic in principle
- Loss function has `/512` hardcoded — will produce wrong gradient scaling at other resolutions
- `attack.py` wraps only `StableDiffusionInpaintPipeline`. FLUX wrapper needed.
- `configs/train_multimodel.yml` and `train_1024.yml` reference FLUX.2-Klein-4B — this was clearly intended but not yet implemented in attack.py
- No GPU available locally — experiments need cloud compute

**Next actions**:
1. Literature search: adversarial immunization against DiT models, resolution extension of perturbation methods
2. Implement FLUX attack wrapper (differentiable FluxInpaintPipeline)
3. Fix resolution-agnostic loss
4. Design patch-based inference for H2
5. Run H1 experiment (multi-model training + cross-model eval)
