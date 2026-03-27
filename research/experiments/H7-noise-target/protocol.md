# Experiment Protocol: H7 — Noise Target Image for loss1

**Date locked:** 2026-03-27
**Status:** Protocol locked — CONFIRMATORY

## Hypothesis
The current loss1 target is a zero tensor (all-black image). Per Mist (arXiv:2305.12683),
target image selection significantly impacts perturbation effectiveness — sharp-edges /
high-contrast / noise patterns outperform smooth targets. Transformer-based DiT models
have strong semantic priors that can reconstruct plausible content from a nearly-black
output; a random-sign noise target provides a harder constraint with no easily-learned
structure to exploit.

## Prediction
Replacing the zero target with a random-sign (±1) high-frequency noise target will:
- Improve loss1 magnitude and training speed (higher-contrast loss surface)
- Improve protection rate against SD3.5 and FLUX by 5-10% (DiT models benefit most)
- Have neutral or mildly positive effect on SD 1.5 protection

## Change
**File:** `src/diffvax/immunization/diffvax_immunization.py` line ~461

Before:
```python
target_image_t = torch.zeros_like(img_out).cuda()
```

After (config-gated, noise_target enabled):
```python
if self._config.get("noise_target", {}).get("enabled", False):
    target_image_t = torch.randint(0, 2, img_out.shape, device="cuda", dtype=img_out.dtype) * 2 - 1
else:
    target_image_t = torch.zeros_like(img_out).cuda()
```

Config addition:
```yaml
noise_target:
  enabled: true
```

## Proxy Metric
- loss1 value at epoch 5, 10, 20 (lower = better disruption against attack model)
- Visual inspection: does the edited output look more corrupted with noise target?

## Evaluation
Run 20 epochs at 512px against SD1.5 with noise_target=false vs noise_target=true.
Compare per-epoch loss1 curves. No GPU required beyond baseline training.

## Prediction Was...
(To be filled after experiment)
