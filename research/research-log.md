# Research Log — DiffVax Extension

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
