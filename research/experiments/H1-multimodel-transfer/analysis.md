# H1 Analysis: Multi-Model Transfer + Purification Robustness
# Status: PENDING — awaiting H1a checkpoint (~44h from training start after 2026-04-08 restart)

## Hypothesis Recap
Training against SD 1.5 + FLUX.1-schnell jointly should produce immunizations that
(a) transfer to SD 3.5 (held-out architecture) and
(b) resist FLUX-based purification (EditorClean, arXiv:2603.13028).

## Prediction (locked before results)
- H1a EDR on FLUX.1-schnell ≥ 80% of DiffVax original EDR on SD 1.5
- H1a EDR on SD 3.5 ≥ 50% of H1a EDR on trained models (zero-shot transfer)
- After FLUX purification at strength=0.5: H1a retains ≥ 70% of direct EDR

---

## Results (fill in when available)

### Transfer Evaluation (from results/transfer_edr_metrics.csv)

| Checkpoint | SD 1.5 | FLUX.1-schnell | SD 3.5 | Mean |
|---|---|---|---|---|
| DiffVax (SD1.5 only) | [?] | [?] | [?] | [?] |
| DiffVax++ H1a (SD+FLUX) | [?] | [?] | [?] | [?] |

### Purification Robustness (from results/purification_edr.csv)

| Purification strength | DiffVax direct | DiffVax purified | H1a direct | H1a purified | H1a retained % |
|---|---|---|---|---|---|
| 0.3 | [?] | [?] | [?] | [?] | [?] |
| 0.5 | [?] | [?] | [?] | [?] | [?] |
| 0.7 | [?] | [?] | [?] | [?] | [?] |

### JPEG Robustness Baseline (from transfer eval with --jpeg-qualities 75 70)

| Checkpoint | No JPEG | q=75 | q=70 | Drop at q=75 |
|---|---|---|---|---|
| DiffVax | [?] | [?] | [?] | [?] |
| DiffVax++ H1a | [?] | [?] | [?] | [?] |

---

## Analysis Questions to Answer

### Q1: Does multi-model training improve EDR on trained models?
- If H1a EDR on SD 1.5 << DiffVax original: multi-model training degrades per-model performance
- If H1a EDR on SD 1.5 ≈ DiffVax original: multi-model is strictly better (transfers too)
- **Expected**: slight regression on SD 1.5 (~5%), large gain on FLUX

### Q2: Does it transfer to SD 3.5 (held-out)?
- SD 3.5 uses MM-DiT (different from FLUX's DiT), same 16-ch VAE
- If EDR on SD 3.5 >> DiffVax original on SD 3.5: transfer confirmed
- **Key mechanism question**: is transfer via shared VAE or via pixel-space generalization?

### Q3: Does FLUX purification fail?
- Null hypothesis: H1a EDR after purification ≈ DiffVax original after purification
- Alternative: H1a retains much higher EDR after purification
- **Product claim**: if H1a EDR retained% > 70% after strength=0.5 purification, the claim
  "multi-model training is a product safety requirement" is substantiated

### Q4: What's the JPEG baseline for H1a?
- This is the starting point for evaluating H7's improvement
- If H1a already drops to ≤ 0.5 at q=75: H7 prediction holds, clear gap to fill
- If H1a unexpectedly holds above 0.6 at q=75: H7 improvement target may be less dramatic

---

## Interpretation Template (fill in after results)

### SUPPORTED / PARTIALLY SUPPORTED / REFUTED

**Transfer to FLUX**: [SUPPORTED/REFUTED] — H1a achieves [?] EDR on FLUX vs DiffVax's [?] ([?]× improvement)

**Transfer to SD 3.5**: [SUPPORTED/REFUTED] — [?] EDR zero-shot on held-out architecture

**Purification resistance**: [SUPPORTED/REFUTED] — After q=0.5 purification, H1a retains [?]% vs DiffVax [?]%

**JPEG baseline**: H1a drops to [?] at q=75 (vs [?] at no-JPEG); gap for H7 = [?] EDR points

### Implications for paper
- If all three SUPPORTED: paper story is complete, all three contributions substantiated
- If transfer only partially works: narrow claim to "improves multi-model coverage" + H6 for safety
- If purification fails to be defeated: H6 is the key result, not transfer

### Implications for next experiments
- If JPEG drop > 0.3 at q=75: H7 is critical (high urgency)
- If JPEG drop < 0.1: H7 is less urgent, consider H4 (VAE loss) next
