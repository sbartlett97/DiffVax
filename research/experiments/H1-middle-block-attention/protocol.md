# Experiment Protocol: H1 — Middle-Block Cross-Attention Disruption for DiT Models

**Date locked:** 2026-03-27
**Status:** Protocol locked — CONFIRMATORY

## Hypothesis
DeContext (arXiv:2512.16625) demonstrated that context propagation in DiT-based
image editing (FLUX, SD3) flows primarily through MIDDLE transformer blocks during
EARLY-TO-MID denoising timesteps (t > 0.7). The current DiffVax Phase 7 attention
loss hooks "early" blocks (block indices 0 to num_hooks). This is mechanistically
wrong for DiT protection: early blocks in MM-DiT / FLUX DiT handle primarily
positional embedding and initial token mixing, not semantic context propagation.

## Prediction
Changing target_blocks from "early" to "middle" in AttentionDisruptionLoss will:
- Improve Phase 7 attention entropy loss convergence in DiT training runs
- Improve protection rate on SD3.5 and FLUX by >10% vs early-block targeting
- Have minimal or neutral effect on SD1.5 (UNet, no Phase 7 in SD1.5 runs)

## Change
**File:** `src/diffvax/losses/attention_loss.py`

Add "middle" option to `_should_hook`:
```python
elif self.target_blocks == "middle":
    # Hook the middle third of blocks — most transferable per DeContext
    third = total_blocks // 3
    return third <= block_idx < (2 * third)
```

Config change:
```yaml
attention_loss:
  enabled: true
  target_blocks: "middle"  # was "early"
  num_hooks: 8             # more hooks needed since middle has more blocks
  weight: 0.4              # slight increase from 0.3
  only_with_dit: true
```

## Proxy Metric
- Attention entropy value at step 100, 500, 1000 (higher entropy = better disruption)
- Protection rate (loss1 magnitude) per DiT model after 50 epochs

## Evaluation
Run 50 epochs at 512px against FLUX.2 with target_blocks="early" vs "middle".
Record attention entropy (from loss_attn) and loss1 each batch.
Compare convergence curves.

## Prediction Was...
(To be filled after experiment)
