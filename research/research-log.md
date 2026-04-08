# Research Log — DiffVax Extension

## 2026-04-08 — H1/H6 Results In + Critical Eval Bug Found + Fixes Applied

**H1a + H6 results arrived from GPU run.**

**CRITICAL BUG: model.eval() collapses NestedUNet perturbation 78x**
- Root cause: DiffVax NestedUNet trained with batch_size=1. BN running_var is near-zero (e.g., 0.005) because single-sample BN computes variance over spatial dimensions of one image, not a batch.
- In eval mode: BN normalizes by sqrt(running_var + eps) ≈ 0.07 → very strong normalization → signal collapses
- In train mode: BN normalizes by current-batch statistics (spatial variance) → matches training
- Effect: published checkpoint gives PSNR=89.9 dB (essentially no perturbation) in eval mode vs PSNR=34.6 dB in train mode (close to published 32.71 dB)
- Fix: eval_transfer.py and eval_purification_robustness.py changed model.eval() → model.train()
- ALL H1 and H6 results from first GPU run are INVALID. Re-run required.

**H1 first run results (INVALID - must re-run):**
- sd15_only (eval mode): PSNR=83.5 dB → near-zero perturbation. EDR=0.16-0.18 (from JPEG compression noise, not immunization)
- multimodel_h1a (eval mode): PSNR=44.2 dB → weak perturbation (4x weaker than expected). EDR=0.01-0.16

**H6 first run results (INVALID - must re-run):**
- Both checkpoints: direct_EDR≈0 (no perturbation due to eval mode)
- Both checkpoints: purified_EDR=0.983 at strength=0.5-0.7 (purifier damage confound, not immunization)
- H6 eval improved: added clean-image purification control (purification_control_disrupted column)
  to detect false positives from purifier damage on non-immunized images

**Patch coverage analysis (CPU-only, no GPU needed):**
- patch_coverage_analysis.py run and verified:
  - stride=256: center pixel covered by 4 patches ✓ (paper claim)
  - stride=512, 384: center pixel covered by 1 patch each
  - EDR vs center coverage correlation: r=0.9563 ✓ SUPPORTS perturbation accumulation hypothesis
  - Figure saved: research/to_human/figures/patch_coverage_density.png
- Note: stride=512/384 have same center coverage (1), so the EDR gap from 0.30→0.33 is not from center
  coverage but from overall coverage increase (max=4 at corners). The 0.33→0.40 jump is the accumulation effect.

**Action required (GPU):**
```bash
git pull
bash scripts/run_post_h1a.sh \
    --h1a-checkpoint <path_to_h1a_checkpoint> \
    --sd15-checkpoint checkpoints/diffvax_trained.pth
```

---

## 2026-04-08 — Baseline Audit + Paper Polish + Figures

**Baseline metrics audit** (web search across all 6 competitor papers):
- No paper in 2024-2026 reports EDR on a public benchmark
- PromptFlare makes SOTA claims with zero numeric evidence
- This validates using EDR as a standardized metric AND is a secondary contribution
- Paper updated: comparison table gains "Reports EDR?" column; experiments section notes metric gap

**fill_paper_results.py** (scripts/): automated placeholder-filling script.
When H1/H6/H7 CSVs arrive, run:
  `python scripts/fill_paper_results.py --h1-csv ... --h6-csv ...`
Outputs formatted tables and key claim values ready to paste into paper drafts.

**Figures** (research/to_human/figures/):
- teaser_figure.png: 3-panel paper figure (Panel A H2 confirmed, B/C pending GPU)
- training_dynamics.png: H1a bimodal loss curve from observed 26-epoch run
- h2_patch_inference.png: regenerated with correct 28dB PSNR threshold (was 30dB)

---

## 2026-04-08 — Pipeline Audit: Two More Critical Bugs Fixed

**Bug 1: train.py not forwarding jpeg_augment_prob to DiffVaxImmunization (CRITICAL for H7)**
- Root cause: immunization_config in train.py only had 5 keys; DiffVaxImmunization reads jpeg_augment_prob from self.config
- Effect: H7 JPEG augmentation would run silently with prob=0.0 (disabled) regardless of config
- Fix: Added jpeg_augment_prob, jpeg_quality_range, checkpoint_every, stop_file to immunization_config
- File: scripts/train.py

**Bug 2: H6 purification mask was all-zeros → purification was a no-op (CRITICAL for H6)**
- Root cause: FluxAttack denoising step: latents = latents * mask + image_latents * (1-mask)
  With mask=0 everywhere, this resets to image_latents on every step → identity
- Effect: H6 eval would show 0% purification for ALL methods — making H6 meaningless
- Fix: Changed empty_mask (all zeros) → full_mask (all ones) in purify_with_flux()
- File: research/experiments/H6-purification-robustness/code/eval_purification_robustness.py

**H2 numbers cross-verified from raw CSV data:**
- baseline_512: EDR=0.250, PSNR=32.71, SSIM=0.9646 ✓
- no_overlap: EDR=0.300, PSNR=30.29, SSIM=0.9557 ✓
- 25pct_overlap: EDR=0.330, PSNR=28.91, SSIM=0.9475 ✓
- 50pct_overlap: EDR=0.400, PSNR=28.69, SSIM=0.9432 ✓
- Ratio: exactly 1.600× ✓ — paper numbers are correct

**Literature sweep (2025-2026):** No new immunization competitors found that beat DiffVax++
on any of the three deployment dimensions. Competitive positioning confirmed.

---

## 2026-04-08 — Config Fix: max_steps Corrected + Paper Sections Drafted

**max_steps recalculation** (all multi-model configs updated):
- Previous: max_steps=8000, estimated at 20s/step (44h). WRONG.
- Actual observed: 1.52s/step, 1600 items/epoch → 1 epoch ≈ 40min
- 8000 steps = 5 epochs = 3.4h → stops BEFORE convergence (epoch 10-13 needed)
- Fix: max_steps=8000 → **16000** (10 epochs, 6.8h). Covers observed convergence.
- Files updated: train_multimodel.yml, train_multimodel_h7.yml, train_multimodel_h4.yml
- H1a GPU run: needs kill → git pull → restart with corrected config

**Paper sections drafted**:
- experiments_draft.md: Full structure with table templates + [X] placeholders for GPU results
- analysis_draft.md: Mechanism explanations (accumulation, bimodal loss, STE) + failure modes
- conclusion_draft.md: Complete draft (no GPU numbers needed)

---

## 2026-04-08 — Pre-writing + Competitive Analysis + Eval Code Audit

**Competitive analysis** (all three 2025 SOTA papers audited):
- Anti-Inpainting (2505.13023, arXiv): no JPEG, no multi-model, no high-res, no metrics
- Attention Attack (2509.10359, ACM MM 2025): same gaps
- PromptFlare (2508.16217, ACM MM 2025): claims SOTA, no model specifics, no numbers
- **Result**: DiffVax++ is the only work to address all three deployment gaps simultaneously
- Saved to: `research/literature/competitor_analysis_2025.md`

**eval_transfer.py audit + fixes**:
- model.train-False bug -> model-dot-eval() (bypasses child module mode)
- Per-model inference steps: FLUX.1-schnell=4, sd15/sd35=20 (schnell is 4-step distilled)

**eval_purification_robustness.py fixes**:
- Same model-dot-eval() fix
- PURIFICATION_STEPS 20 -> 4, EDIT_STEPS 20 -> 4 (same schnell reasoning)
- Using 20 steps with schnell degrades quality -> would bias H6 toward appearing more robust

**Paper pre-writing**:
- Draft abstract with [X] placeholders in `paper/draft_abstract_outline.md`
- Full paper outline: section structure, table templates, key claims table
- Pre-registered H1 and H6 analysis templates in each experiment dir
  with predictions locked before results arrive

**State**: Infrastructure fully audited and ready. Waiting for H1a GPU checkpoint.

---

## 2026-04-07 — SD3Attack + H7 STE Validation

**SD3Attack CPU offload bug fixed** (`src/diffvax/attack_sd3.py`):
- Root cause: `pipe.enable_model_cpu_offload()` offloads transformer to CPU; our custom denoising loop calls `pipe.transformer()` directly without pipeline hooks → transformer is on CPU → crash
- Fix: Removed `enable_model_cpu_offload()`, added `pipe.transformer.enable_gradient_checkpointing()`
- SD3.5-medium is ~16 GB fp16, fits on 95 GB GPU alongside other models

**H7 STE JPEG gradient flow validated** (`src/diffvax/jpeg_augment.py`):
- Forward: output equals JPEG-compressed reference at q=70 and q=75 ✓
- Backward: gradient exists, mean=0.000081 ≈ 1/(1×3×64×64) — confirms identity backward pass ✓
- Probability control: p=0.0 → 0%, p=0.5 → ~45%, p=1.0 → 100% ✓
- All validation tests passed on CPU. Ready for GPU training.

---

## 2026-04-07 — H2 GPU Results: CONFIRMED at 1.60×

**Result**: 50% overlap patch immunization (stride=256) at 1088×1088 achieves EDR=0.400 vs baseline_512 EDR=0.250 → **1.60× ratio** (prediction: ≥0.80×).

**Ranking**: 50pct_overlap (EDR=0.400) > 25pct_overlap (0.330) > no_overlap (0.300) > baseline_512 (0.250). All conditions beat baseline.

**Mechanism discovered**: Perturbation accumulation from overlapping patches. Center of 1088px image is covered by ~4 patches at stride=256; blended sum is denser than single 512px immunization.

**Product decision**: Default to 1088px + stride=256. H3 (native high-res training) deprioritized.

**Note**: Absolute EDR (25-40%) is checkpoint-limited. Will re-run with H1a checkpoint when available.

## 2026-04-07 — FLUX Model Fix + Eval Infrastructure

**Critical fix**: FLUX.2-Klein-4B is T5-only (no CLIP encoder), incompatible with `FluxInpaintPipeline`. Changed all configs to `FLUX.1-schnell` (distilled, dual-encoder, 4 steps, guidance_scale=0).

**`attack_flux.py`**: added try/except for T5-only FLUX models, auto-detection of distilled vs non-distilled (schnell → guidance_scale=0, dev → 3.5), default guidance_scale changed to 0.0.

**`eval_transfer.py` rewritten**: added `--jpeg-qualities` flag for H7 protocol, `torch.no_grad()`, `torch.cuda.empty_cache()`, clean edits precomputed once per image.

**`research/src/plot_results.py`**: new publication-ready plotting for H1/H2/H6/H7.

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
