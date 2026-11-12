# DiffVax: 1088px protection against DiT editors

Goal: train one NestedUNet that protects images up to 1088x1088 against
modern editors (FLUX.2 / FLUX.1 Kontext / SD3.5; Nano Banana measured
black-box). Full evaluation and rationale: `research/evaluation-2026-11-11.md`.

Status 2026-11-11: decisions made (80 GB RunPod, dead terms off by default,
Nano Banana eval-only, epsilon 16/255). P0, P1 code and P2 code implemented and
CPU-tested (138 tests). All "Gate:" lines and P3-P5 need the GPU / the user.

## P0. Hygiene (local)
- [x] P0.1 bf16 on CUDA in `resolve_dtype()`; GradScaler off for bf16; update `test_device_resolution.py`
- [x] P0.2 Hard epsilon: `delta = eps * tanh(unet_out)`; config `epsilon`; test bound
- [x] P0.3 Surrogate selection in blocks (`hold_batches`, default 32); test swap count
- [x] P0.4 New default config: noise_target / attention_loss / flat_minima / clip_loss off
- [x] P0.5 Fix 7 MPS-host test failures (5 FLUX "accelerator available" guard; 2 DDP: gloo ranks resolve to MPS, `allgather_` unsupported, force CPU); suite green on this Mac
- [x] P0.6 Fix `research_v3.yml` curriculum ordering; drop dead `load_path` in `train_1088_v3.yml`
- [ ] Gate: tests green; 10-batch stub smoke run shows eps bound and loss1 falling

## P1. Correct FLUX surrogate + loss
- [x] P1.1 `src/diffvax/flux_edit_attack.py::FluxKleinEditAttack` (reference-token edit path, differentiable)
- [x] P1.2 Hidden-state distance loss on middle DiT blocks (clean vs protected reference tokens)
- [x] P1.3 Replace latent cosine with normalised L2; weight from gradient-norm balance
- [x] P1.4 Gradient-flow tests A9+ for the edit path (gtf 0.25/0.5/1.0, reference never noised, tiny-UNet learns)
- [x] P1.5 Eval zoo: Kontext dev + Qwen-Image-Edit entries (inference only)
- [ ] Gate: parity_check_klein.py "PARITY OK" on real weights, then GPU 2k steps @512 Klein-4B: loss1 down >= 10%, hidden-state distance up, no NaN, VRAM logged

## P2. Evaluation protocol
- [x] P2.1 `scripts/eval_protocol.py`: >= 50 imgs x 2 prompts x 2 seeds; resize-1080 + JPEG q80; controls (clean, noise @ matched PSNR, PhotoGuard); LPIPS/SSIM/CLIP-delta; protection rate + bootstrap CI
- [x] P2.2 (zoo entries added; Kontext/Qwen opt-in) Editors in real modes and step counts
- [ ] Gate (runbook 1c): `PROTECTED` verdict for `checkpoints/diffvax_trained.pth` vs SD 1.5 Inpainting without the transform; noise control ~0

## P3. GPU validation 512 -> 1088
- [ ] P3.1 Signal run Klein-4B @512, eps 16/255 and 32/255
- [ ] P3.2 Ablations (hidden-state, latent L2, spectral, gtf, EoT)
- [ ] P3.3 Ensemble + transfer to unseen Kontext / Qwen-Image-Edit
- [ ] P3.4 Curriculum 512 -> 768 -> 1024 -> 1088; VRAM per stage
- [ ] Gate: @1088 protection > matched-PSNR control with CI; LPIPS <= 0.05, SSIM >= 0.97

## P4. Black-box probe (Nano Banana)
- [ ] 50 clean + 50 protected through Gemini edit API; report delta with CI in README

## P5. Data scale-up (only if P3 passes)
- [ ] 2-5k images >= 1088px, varied subjects

## Review
2026-11-11 local implementation complete; GPU stages pending.
- Found and fixed while implementing: image export truncated (budget measured 17
  at eps 16); bf16 quantised the inference/eval image tensor and the training
  loader's images; `immunize_image_pil` and the dataset now stay float32.
- Parity: `FluxKleinEditAttack` matches the real `Flux2KleinPipeline` to 1e-4
  (distilled and CFG) on tiny random weights. Real-weight parity is untested.
- NOT verified: anything on CUDA, any protection result, real Klein VRAM/step time.
