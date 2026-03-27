# Experiment Protocol: H3 — CLIP ViT-H/14 Loss for DALL-E 3 Coverage

**Date locked:** 2026-03-27
**Status:** Protocol locked — CONFIRMATORY

## Hypothesis
DALL-E 3 uses CLIP ViT-H/14 (or similar) image embeddings to condition generation.
The current DiffVax CLIP loss uses ViT-B/32, which has limited representational
overlap with the DALL-E 3 vision backbone. Upgrading to ViT-H/14 (via OpenCLIP
`ViT-H-14` with `laion2b_s32b_b79k` weights) should produce feature disruptions
that transfer to DALL-E 3 without direct model access.

Literature (Glaze, DTIA, NL Adversarial) all confirm CLIP-H embedding disruption
as a valid black-box proxy for DALL-E 3 protection.

## Prediction
- CLIP-H cosine distance shift (feat_adv vs feat_orig) will be larger than ViT-B/32
  because ViT-H/14 has richer feature space with more granular adversarial gradient
- Protection transfer to DALL-E 3 (proxied by CLIP-H distance) >0.3 cosine distance
- May slightly increase training VRAM (~2-3GB for ViT-H/14 vs ViT-B/32)

## Change
**File:** `src/diffvax/losses/clip_loss.py` — already config-gated via `model` field

Config change:
```yaml
clip_loss:
  enabled: true
  model: "ViT-H-14"                # was "ViT-B/32"
  pretrained: "laion2b_s32b_b79k"  # ViT-H/14 OpenCLIP weights
  feature_weight: 1.0
  semantic_weight: 0.5
```

No code changes required — the model/pretrained config keys already control this.

## Proxy Metric
- CLIP-H cosine distance (feat_adv vs feat_orig) — proxy for DALL-E 3 disruption
- Overall training loss curve (should not diverge)
- VRAM usage change

## Evaluation
Run 20 epochs at 512px, SD1.5, with clip_loss.model=ViT-B/32 vs ViT-H-14.
Record CLIP cosine distance on a held-out set of 10 images.
Compare loss curves and CLIP-H distance per epoch.

## Prediction Was...
(To be filled after experiment)
