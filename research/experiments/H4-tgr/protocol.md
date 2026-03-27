# Experiment Protocol: H4 — Token Gradient Regularization (TGR) in DiT Backprop

**Date locked:** 2026-03-27
**Status:** Protocol locked — CONFIRMATORY

## Hypothesis
TGR (arXiv:2303.15754, CVPR 2023) found that high token-to-token gradient variance
within ViT attention blocks is the primary cause of poor adversarial transfer.
At 1088px, SD3.5's MM-DiT joint attention has ~18,496 tokens (image + text).
The variance of gradients through this joint attention is extremely high, causing
unstable training and poor perturbation quality at high resolution.

Token-wise gradient normalization during backpropagation — scaling each token's
gradient to have unit norm — reduces variance and produces smoother perturbation
directions that generalize better across model variants.

## Prediction
Adding TGR backward hook to SD3.5/FLUX transformer blocks will:
- Reduce gradient norm variance at UNet++ input by >50%
- Improve loss1 convergence stability at 1088px (lower standard deviation in loss)
- Improve protection rate on SD3.5 by 5-8% over no-TGR baseline
- Not significantly affect training throughput (backward hook is O(seq_len * dim))

## Change
**File:** `src/diffvax/sd3_attack.py` (and similarly `flux_attack.py`)

Register a backward hook on transformer blocks that normalizes token gradients:

```python
def _make_tgr_hook(self):
    """Token Gradient Regularization backward hook.
    Normalizes gradient tensor token-wise to reduce variance.
    Input grad shape: (B, seq_len, dim)
    """
    def hook_fn(grad):
        # Normalize each token's gradient to unit norm
        norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return grad / norm
    return hook_fn
```

Applied to the hidden_states output of each transformer block during backward.

Config:
```yaml
sd3_attack:
  token_gradient_regularization: true
flux_attack:
  token_gradient_regularization: true
```

## Proxy Metric
- Gradient norm std at UNet++ input (lower = more stable)
- loss1 std across batches (lower = more stable training)
- Protection rate after 50 epochs

## Evaluation
Run 50 epochs at 512px against SD3.5 with TGR=false vs TGR=true.
Record gradient norm std per epoch and loss1 variance.

## Prediction Was...
(To be filled after experiment)
