# H6: Scaled-Up NestedUNet Capacity

**Status:** Protocol locked
**Date locked:** 2026-03-27
**Hypothesis:** H6 (nb_filter configuration)

---

## What

Make NestedUNet filter counts configurable via a new `nb_filter` constructor
parameter.  Default remains `[32, 64, 128, 256, 512]` (original, ~1.8M params).
Larger variant: `[64, 128, 256, 512, 1024]` (~7M params).

**Changes:**
- `src/diffvax/model.py`: Add `nb_filter: list[int] | None = None` parameter
  to `NestedUNet.__init__`; propagate through all VGGBlock instantiations.
- `src/diffvax/immunization/diffvax_immunization.py`: Read `nb_filter` from
  config (key `nb_filter`) and pass to `NestedUNet(num_classes=3, nb_filter=...)`.
- `configs/research_v3.yml`: Add commented-out `nb_filter: [64, 128, 256, 512, 1024]`
  as the H6 ablation option (default stays small to validate other hypotheses first).

---

## Why

At 1088×1088, the NestedUNet bottleneck is 68×68 pixels with 512 channels
(~1.8M parameters total).  The model processes 4× more spatial positions than
at 512px, but with the same capacity.  Doubling filter counts at every stage
gives ~7M parameters and quadruples the representational capacity at the
bottleneck, potentially allowing more coherent long-range perturbation patterns.

**However:** TGR results suggest gradient *quality* (token-wise variance) is
the primary bottleneck, not network capacity.  H6 is low priority relative to
H1–H5.  Implement now for completeness and to enable ablation studies without
a code change.

---

## Prediction (CONFIRMATORY)

1. Larger model (`nb_filter=[64,128,256,512,1024]`) will reduce training loss
   at 1088px faster (lower loss at same iteration count) vs small model.
2. Protection rate improvement: **+3-5%** vs baseline at 1088px, less at 512px.
3. VRAM overhead of larger NestedUNet (~28MB extra) is negligible vs attack models.
4. If gradient quality (TGR) is the bottleneck, improvement will be <2% —
   this would support TGR as the dominant factor.

---

## Notes

- `nb_filter` is stored as a constructor argument so `PyTorchModelHubMixin`
  serialises it to `config.json` — Hub checkpoints for the larger model can be
  loaded back correctly via `from_pretrained()`.
- When loading an existing 512px checkpoint into the larger model, weights will
  be randomly initialised (architecture mismatch) — stage-2 fine-tuning from
  v3-small checkpoint is NOT possible for H6.  Must train from scratch.
- To run H6 ablation: uncomment `nb_filter: [64, 128, 256, 512, 1024]` in
  the config, then train with a fresh checkpoint (`load_existing: false`).
