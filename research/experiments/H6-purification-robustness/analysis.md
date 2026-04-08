# H6 Analysis: Purification Robustness
# Status: PENDING — requires H1a checkpoint

## Hypothesis Recap
FLUX-based EditorClean purification (arXiv:2603.13028) fails on DiffVax-FLUX
immunized images (H1a checkpoint) because the immunization specifically disrupts
FLUX's latent representation. Tested at purification strengths 0.3, 0.5, 0.7.

## Prediction (locked before results)
- DiffVax (SD1.5 only): purification at strength=0.5 recovers editability substantially
  (retained_EDR ≤ 0.5 relative to direct EDR)
- DiffVax++ H1a (SD+FLUX): purification at strength=0.5 largely fails
  (retained_EDR ≥ 0.7 relative to direct EDR)
- Effect grows with purification strength (harder purification → clearer gap)

---

## Results (fill in when available from results/purification_edr.csv)

### EDR Before and After Purification

| Checkpoint | Direct EDR | Purify 0.3 | Purify 0.5 | Purify 0.7 |
|---|---|---|---|---|
| DiffVax (SD1.5 only) | [?] | [?] | [?] | [?] |
| DiffVax++ H1a (SD+FLUX) | [?] | [?] | [?] | [?] |

### EDR Retained After Purification (%)

| Checkpoint | Purify 0.3 | Purify 0.5 | Purify 0.7 |
|---|---|---|---|
| DiffVax (SD1.5 only) | [?]% | [?]% | [?]% |
| DiffVax++ H1a (SD+FLUX) | [?]% | [?]% | [?]% |

---

## Analysis Questions

### Q1: Does purification defeat DiffVax (baseline)?
- Expected: yes, as shown in arXiv:2603.13028 for SD1.5 immunizations
- If not: our DiffVax baseline is already surprisingly robust — note as unexpected finding

### Q2: Does H1a resist purification better?
- Key comparison: retained_EDR(H1a, 0.5) vs retained_EDR(DiffVax, 0.5)
- Paper claim threshold: H1a retains ≥ 70% (i.e., purification degrades EDR by ≤ 30%)
  while DiffVax retains ≤ 50% (purification degrades by ≥ 50%)

### Q3: What is the mechanism?
- Hypothesis: H1a's perturbation disrupts FLUX's VAE encoding directly
- Test: compare ssim_imm_orig before vs after purification — if immunized SSIM improves
  substantially after purification but EDR doesn't improve, the perturbation survived
  but was rendered ineffective → different mechanism than "purification removes perturbation"

---

## Interpretation Template (fill in after results)

**Purification resistance**: [SUPPORTED/REFUTED]
- Reasoning: [fill in]
- Unexpected finding (if any): [fill in]

**Paper impact**:
- If STRONGLY SUPPORTED: "Multi-model training is a product safety requirement against EditorClean" is a
  direct, testable, novel claim. Table 4 becomes a centrepiece table.
- If WEAKLY SUPPORTED: "Multi-model training reduces purification success" is a softer claim.
- If REFUTED: investigate whether H4 (VAE feature loss) would add purification resistance.
  Note in paper as "limitation" and future work.
