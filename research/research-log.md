# Research Log — DiffVax Extension

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
