# GPU Validation Runbook

Everything code-provable about the training method has been verified by the
CPU test suite (see `findings.md` → *Verification Status & Confidence
Assessment*). This runbook is the remaining Tier 2 work: empirical
validation on GPU. Follow it top to bottom; each stage has an explicit
go/no-go criterion.

## 0. Environment

- GPU: 24 GB minimum (gtf=0.5 + gradient checkpointing keep SD3.5/FLUX at
  512px well under this; 1088px is tight but within range per H2 estimates).
- `pip install -e . && pip install kornia open-clip-torch` (kornia is
  REQUIRED when `eot.p_jpeg > 0` — the loop fails loudly otherwise).
- `python -m pytest tests/` first. All non-CUDA-skipped tests must pass;
  the CUDA-only scaler test (`test_t3_unscale_corrects_inflated_gradients`)
  should now run and pass too.

## 1. Signal sanity run (~2-4 h)

    python scripts/train.py --config configs/research_v3.yml  # cap at ~2k epochs

Watch for, in order of importance:

1. **`[Scaler] scale=…` lines** (every 100 batches): the scale should
   stabilize (typically 2^14–2^16). A scale that keeps halving means
   NaN/Inf gradients — abort and bisect loss terms via config gates.
2. **Per-model loss1** (`[flux] loss1=… [sd3] loss1=…` epoch lines): must
   TREND DOWN for the DiT models, not only for `[sd]`. Flat DiT loss1 with
   falling loss2 is the historical failure signature (C8) — now covered by
   tests, but this is the empirical confirmation.
3. **`Lat:` term**: starts near 1.0 (cosine similarity of near-identical
   latents) and should fall. If it climbs toward 1.0, the H8 sign
   regressed (test A3 guards this).
4. **No `AttentionDisruptionLoss … detached` warning**: if it appears,
   Phase 7 is dead in that configuration — file it, don't ignore it.

Go criterion: DiT-model loss1 down ≥10% from its epoch-5 average within
2k epochs, no scaler collapse.

## 2. Protection eval at 512px

    python scripts/eval_multimodel.py --checkpoint <best.pth> ...

- `clip_delta` (clip_no_defense − clip_with_defense) must be positive on
  SD1.5, SD3.5, FLUX; magnitude is the protection strength.
- `orig_vs_imm_ssim` ≥ 0.95 keeps the perturbation imperceptible-ish;
  trade off against clip_delta via `alpha`/`spectral_loss.weight`.
- Compare against `scripts/compare_baselines.py` (PhotoGuard) — the method
  must beat the PhotoGuard baseline on DiT models to justify itself.

Go criterion: mean clip_delta > 0 on FLUX and SD3.5 with SSIM ≥ 0.93.

## 3. Ablations (one config knob each, ~1 run per knob)

| Knob | Question |
|------|----------|
| `latent_loss.enabled` | Does H8 add DiT protection beyond loss1? |
| `attention_loss.enabled` | Does Phase 7 entropy help or just add noise? |
| `*_attack.token_gradient_regularization` | TGR: transfer gain vs none? |
| `*_attack.gradient_timestep_fraction` 0.5 vs 1.0 | Same protection at half VRAM? |
| `spectral_loss.enabled` | SSIM gain at equal clip_delta? |

Keep everything else fixed at the stage-1 config; 2k epochs each is enough
for ranking.

## 4. 1088px production run

Only after stages 1-3: `configs/train_1088_v3.yml`, staged from the 512px
checkpoint (`load_existing` + `load_path`). Watch VRAM at the 1024→1088
curriculum boundary — the fixed noise target cache allocates a new target
per output shape (expected, small).

## 5. Black-box transfer probe (Tier 3, best-effort)

For nano-banana / DALL-E-class services there is no gradient access; the
only honest measurement is: upload N protected + N unprotected images,
request the same edits, score CLIP-similarity of results to prompts, and
report the delta with confidence intervals. Expect substantially weaker
transfer than white-box results; do NOT extrapolate white-box protection
rates to these services.
