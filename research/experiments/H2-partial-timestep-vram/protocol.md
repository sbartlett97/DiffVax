# Experiment Protocol: H2 — Partial-Timestep Gradient for VRAM Reduction

**Date locked:** 2026-03-27
**Status:** Protocol locked — CONFIRMATORY

## Hypothesis
"Distraction Is All You Need" (CVPR 2024) demonstrated that computing gradients
only at a subset of critical timesteps achieves equivalent protection efficacy
at ~50% of the VRAM cost vs full-timestep backpropagation.

At 1088×1088, SD3.5 requires ~26-28GB VRAM with full-timestep backprop (4 steps
with 18,496 tokens each). Early-high-sigma timesteps (t close to 1.0, i.e., the
FIRST timesteps in the denoising chain for rectified flow) are the most critical
for protection because they determine the global structure of the generated output.

By using `torch.no_grad()` on late (low-sigma) timesteps and only computing
gradients through the early timesteps, we reduce the backward graph size proportional
to the fraction of timesteps skipped.

## Prediction
With gradient_timestep_fraction=0.5 (only backprop through first 50% of timesteps):
- VRAM reduction: ~40-50% for SD3.5 at 1088px
- Protection efficacy: <5% degradation vs full-timestep backprop
- SD3.5 1088px training becomes feasible on 24GB GPUs

## Change
**File:** `src/diffvax/sd3_attack.py` and `src/diffvax/flux_attack.py`

In the MM-DiT denoising loop, wrap late timesteps in `torch.no_grad()`:

```python
grad_fraction = self._gradient_timestep_fraction  # from config, default 1.0
n_grad_steps = max(1, int(len(timesteps) * grad_fraction))

for step_idx, t in enumerate(timesteps):
    use_grad = step_idx < n_grad_steps
    ctx = torch.enable_grad() if use_grad else torch.no_grad()
    with ctx:
        # ... transformer forward pass ...
```

Config addition:
```yaml
sd3_attack:
  gradient_timestep_fraction: 0.5  # 1.0 = full backprop (default)
flux_attack:
  gradient_timestep_fraction: 0.5
```

## Proxy Metric
- VRAM peak usage at 1088px with fraction=1.0 vs 0.5 vs 0.25
- loss1 value after 20 epochs (protection quality)
- Training step time (seconds/batch)

## Evaluation
Run 20 epochs at 1024px (scale test) against SD3.5 with three settings:
  - fraction=1.0 (baseline), fraction=0.5, fraction=0.25
Compare: VRAM (nvidia-smi), loss1 curve, step time.

## Prediction Was...
(To be filled after experiment)
