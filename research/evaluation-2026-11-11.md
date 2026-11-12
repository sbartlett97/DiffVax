# Evaluation: can this fork reach 1088px protection against DiT editors?

**Date:** 2026-11-11
**Scope:** Code audit of the research track, plus the empirical results already
in `outputs/`. No new training runs. Companion checklist: `tasks/todo.md`.

## 1. Verdict

The training harness is sound at the code level. The CPU test suite proves
gradient-path integrity and loss signs. But the method, as currently wired,
does not produce measurable protection against SD3.5, FLUX.2 Klein or SDXL.
The evaluation reports in `outputs/` show this directly (section 2).

The main cause is structural, not a bug. The DiT surrogates attack the wrong
mechanism. They run img2img: they noise the latent of the protected image and
denoise it. Real modern editors do not work this way. FLUX.2 Klein, FLUX.1
Kontext, Qwen-Image-Edit and (by all public evidence) Nano Banana take the
clean reference image as **context tokens** that stay un-noised at every
denoising step. The generation latent starts from pure noise. The original
DiffVax result against SD 1.5 inpainting works for the same reason: the
inpainting UNet sees the clean masked-image latent at every step. The
perturbation has a direct path into every step. The img2img surrogates give
the perturbation only one weak path: the initial noised latent, where 50-100%
of the signal is noise.

Secondary causes: three of the five extra loss terms are dead or near-dead
by construction, the training dtype on CUDA (fp16 + GradScaler) caused the
C-SCALER collapses, there is no hard perturbation budget, and the evaluation
protocol (1-3 images, no controls) cannot detect protection even if it exists.

Recommendation: rebuild the FLUX surrogate around the real edit mechanism,
simplify the loss, switch CUDA to bf16, add a hard epsilon, then validate on
a rented GPU with a proper protocol. Treat Nano Banana as a black-box
evaluation target only. Details in sections 4 and 5.

## 2. Evidence already in the repo

All reports are at 1088px, seed 5, `person` masks, checkpoints from the Hub.
`clip_delta` = CLIP(edit of clean, prompt) - CLIP(edit of protected, prompt).
Positive = protection. `edit_ssim` = SSIM between the two edits. 1.0 = no
effect. `imm_psnr` = PSNR of protected image vs original (visibility).

| Report dir | Checkpoint | Editor | n | clip_delta | edit_ssim | imm_psnr |
|---|---|---|---|---|---|---|
| eval_report | v3-research-sd3only-goose | SD3.5 | 1 | +1.5 | 0.92 | 33.0 |
| eval_steps28 | same | SD3.5 (28 steps) | 3 | -0.2 | 0.95 | 32.6 |
| eval_steps6 | same | SD3.5 (6 steps) | 3 | +0.3 | 0.93 | 32.6 |
| new_test | same | SD3.5 | 2 | +1.0 | 0.65 | 21.7 |
| new_test_gated_mask | same | SD3.5 | 2 | -0.6 | 0.70 | 25.3 |
| sd3_5_paper_replica | dit-research-black-noise | SD3.5 | 2 | -2.0 | 0.83 | 35.1 |
| sd3_5_paper_replica_latest | same | SD3.5 | 2 | +1.2 | 0.80 | 26.3 |
| ..._latest_ensemble | same | SD 1.5 inpaint | 2 | -0.8 | 0.90 | 26.5 |
| ..._latest_ensemble | same | SD3.5 | 2 | +0.3 | 0.80 | 26.5 |
| ..._latest_ensemble | same | FLUX.2 Klein | 2 | +0.6 | 0.91 | 26.5 |
| ..._latest_sdxl | same | SDXL img2img | 2 | -1.0 | 0.92 | 27.8 |

Reading:

- `clip_delta` is within noise of zero and changes sign between runs. No
  checkpoint protects against any DiT editor.
- The only runs with low `edit_ssim` (0.65-0.70) are the runs where the
  perturbation itself is strongly visible (PSNR 22-25 dB). A random-noise
  control at matched PSNR would likely produce the same drop. The eval has no
  such control, so these numbers prove nothing.
- Even the SD 1.5 inpaint row shows no protection from the DiT-trained
  checkpoint, so the DiT training did not preserve the one thing that
  originally worked.
- n is 1-3 images. Standard deviations are as large as the means.

The CPU suite as run today on this Mac: 98 passed, 7 failed, 1 skipped.
Five FLUX tests in `tests/test_attack_gradient_flow.py` raise "pipeline is
on CPU but an accelerator (mps) is available", and the two 2-rank DDP tests
in `tests/test_distributed.py` fail because the gloo workers resolve to
MPS and `c10d::allgather_` is not implemented for MPS (DDP under gloo must
force CPU). The suite
was written for CUDA/CPU hosts, where it is reported green. Fix in Phase 0.

## 3. What the code does well (keep)

- `NestedUNet` is fully convolutional and GroupNorm-based. It runs at any
  multiple of 16, so 1088px needs no architecture change. The dataset is
  already 1088x1088 (172 train, 20 validation images).
- `gradient_timestep_fraction` with the straight-through scheduler path is
  correct and tested. Keep it.
- Non-reentrant gradient checkpointing, text-encoder offload on CUDA, one
  resident surrogate at a time, DDP ensemble sharding: all sensible for
  24-48 GB cards.
- `_weighted_l1` float32 reduction, the NaN and scale-zero abort guards, and
  the reporter are good operational hygiene.
- EoT (kornia JPEG, resize, blur, noise) is differentiable and fail-loud.
  This is the right tool for the social-media pipeline.

## 4. Findings, ranked by impact on the goal

### F1. The DiT surrogates attack the wrong mechanism (structural, blocking)

`flux_attack.py` and `sd3_attack.py` both: VAE-encode the protected image,
mix it with noise at strength 0.5-1.0, denoise, decode. The gradient reaches
the perturbation only through that one noised initial latent.

`diffusers` 0.39 `Flux2KleinPipeline.__call__` (the model the eval script
actually calls, and the public FLUX.2 editing API) does this instead:

1. `prepare_image_latents()` VAE-encodes the reference image, patchifies,
   BN-normalises, packs, and gives the tokens a time coordinate T = 10 (the
   generation tokens have T = 0).
2. Generation latents start as pure noise.
3. At every step: `latent_model_input = cat([latents, image_latents], dim=1)`.
   The transformer attends across both. The prediction is sliced to the
   generation tokens only. The reference tokens are never noised and never
   updated.

Consequences:

- Training and evaluation disagree. The eval's "FLUX.2 Klein" row runs the
  real edit mode. The surrogate never trained against it.
- The img2img surrogate under-estimates the perturbation's leverage. In edit
  mode the perturbed tokens are present, clean, at every block of every step.
  That is a much stronger and more direct gradient path, and it is the one
  that matters for the threat model.
- The SD 1.5 inpainting success in the original paper is the same mechanism.
  The method is known to work when the editor conditions on the clean image.

FLUX.1 Kontext (`FluxKontextPipeline`), Qwen-Image-Edit
(`QwenImageEditPipeline`) and FLUX.2 Klein inpaint
(`Flux2KleinInpaintPipeline`) are all available in the installed diffusers
and all use reference-token conditioning. Of these, **FLUX.2 Klein 4B** is
the only one whose weights plus activations fit a 24 GB card with gradients
enabled. Kontext dev (12B) needs 48-80 GB or quantised weights. Qwen-Image-
Edit (20B) is out of reach for training and belongs in the eval zoo only.

SD3.5 img2img is a weak proxy for anything an attacker would actually use in
2026. Keep it as a second architecture family for transfer measurement, but
stop treating it as the primary target.

### F2. Nano Banana has no white-box path (threat-model limit)

Gemini image editing is closed. No gradient, no surrogate with the same
architecture, no public evidence that it uses a CLIP-H image encoder. The
CLIP-H proxy argument (H3) rests on DALL-E 3 era assumptions. Protection
against Nano Banana can only be measured black-box: upload N protected and N
clean images, request the same edits, compare. Published results on this
class of service (and Google's own preprocessing) say transfer is weak.
Plan for it as a measurement, not a training target. State this in the README
before any claim is made.

### F3. Dead or near-dead loss terms

- `attention_loss.py` does not use attention maps. It takes
  `softmax(mean |attn_out| over channels)` per token and maximises entropy.
  After LayerNorm the per-token mean magnitude is close to uniform, so the
  entropy sits at ln(N) regardless of input. `findings.md` already records
  the value pinned at -6.9277 = -ln(1024). The term carries no useful
  gradient. Replace with a hidden-state distance loss (section 5, P1.3) or
  remove.
- `latent_loss.py` returns cosine similarity of 300k-dim latents. Under a
  32/255 budget the achievable change is a few hundredths. With weight 1.0 it
  is a no-op next to loss2. Logged values of 0.97-0.99 confirm this. Replace
  with a normalised L2 distance (or PhotoGuard's target-latent L2) with a
  weight tuned so its gradient norm is comparable to loss1's.
- `clip_loss.py` feature term downsamples 1088px to 224px (4.9x). Most of a
  high-frequency perturbation averages out before CLIP sees it. The only
  perturbations that survive are low-frequency, which the spectral loss then
  penalises. The two terms fight. The semantic term (edit output vs prompt)
  is reasonable but is one more gradient path through the whole surrogate.
  Deprioritise both. ViT-H/14 at 1088px batch is also a large VRAM and time
  cost for a term with no demonstrated effect.
- `noise_target` (H7, random +/-1 target) saturates: E|x - t| = 1 for any x.
  Already diagnosed in the code; the `image_path` variant exists. The
  original zeros target is still the best-evidenced choice (it is what the
  paper used). Make zeros the default again.
- `flat_minima.py` `grad_norm` method scales the whole gradient by
  (1 + 2 lambda ||g||). That is an adaptive learning-rate factor, not a
  sharpness penalty. Harmless, pointless. Disable by default.

Net: of loss1, loss2, CLIP, spectral, latent, attention, flat-minima, only
loss1, loss2 and spectral are doing what their docstrings say.

### F4. fp16 + GradScaler on CUDA caused the C-SCALER collapses

Every DiT surrogate here ships in bf16. SD3.5 and FLUX are known to overflow
in fp16 in some layers. `resolve_dtype()` picks fp16 on CUDA, which forces
GradScaler, which produced the scale-to-zero lock-up that consumed the
2026-09 investigation. Any Ampere-or-newer card (A10, A100, L40S, 4090,
A6000) supports bf16. Switching CUDA to bf16 removes the scaler, the
found_inf skips, and the whole class of fp16 reduction bugs that `_weighted_l1`
and the TGR hook now work around. The 1088px `weight.sum()` overflow story
also disappears. This is a one-line default change plus a test update.

### F5. No hard perturbation budget

`self.eps = 32/255` is assigned and never used. The perturbation is bounded
only by the pixel clamp and the soft L1 penalty (alpha). The eval reports
show protected-image PSNR of 22-27 dB in several runs. That is clearly
visible, which defeats the social-media use case. Bound the output:
`delta = eps * tanh(unet_out)` (or a hard clamp to [-eps, eps]) with eps a
config value, and report PSNR/SSIM/LPIPS of the perturbation in every eval.
Then alpha tunes imperceptibility inside the budget instead of fighting it.

### F6. The evaluation protocol cannot answer the question

- n = 1-3 images. Need >= 50 validation images x 2 prompts x 2 seeds.
- No controls. Need: clean image, random noise at matched PSNR, PhotoGuard at
  matched PSNR. Without the noise control, any SSIM drop is uninterpretable.
- No social-media transform before the edit. Need: resize to 1080 wide + JPEG
  q 75-85 (Instagram/X-like), applied to both protected and clean images.
- No protection-rate metric. Need a per-image success criterion (for example
  LPIPS(edit_clean, edit_protected) > tau, or CLIP-score drop > tau) and its
  rate with a confidence interval, alongside the means.
- Editors not run in their real modes or at real step counts. Train uses
  2-4 steps; the eval that matters uses 28 (SD3.5) or 4 (Klein distilled).
  Klein edit mode is already what the eval calls; make training match.
- Add FLUX.1 Kontext dev and Qwen-Image-Edit to the eval zoo as unseen
  transfer targets (inference only, so VRAM is manageable with offload).

### F7. Throughput and fidelity of the training proxy

- `select_and_load()` picks a surrogate per batch. With three surrogates the
  swap probability per batch is about 0.45, and each swap moves 5-15 GB of
  weights between host and device. Select in blocks (hold a surrogate for K
  batches, K ~ 32) or pin per rank under DDP. Expect a large wall-clock gain.
- `num_inference_steps: 2` with gtf 0.5 is a one-step proxy of a 28-step
  edit. Klein is a 4-step distilled model, so 4 steps with gtf 0.5 is a
  faithful proxy there. For SD3.5 use 6-8 steps with gtf 0.25-0.5.
- `research_v3.yml` curriculum stages are non-monotonic (580 then 500) and
  the run is 500 epochs, so stage 4 never runs. `train_1088_v3.yml` points at
  a checkpoint that does not exist. Fix when the configs are rewritten.

### F8. Dataset

172 training images, all people, one source. Enough to prove the signal
exists at 512px. Not enough for a general social-media protector. Plan a
2-5k image set at >= 1088px with varied subjects once Phase 3 shows signal.

### F9. Hardware

This Mac (M5, 32 GB unified) cannot train the DiT surrogates at 1088px. The
last local run died at 42 GB MPS allocation. Local work = tests, small-res
smoke runs with stub surrogates, eval of single images. All real training and
the protocol in F6 need a rented 48 GB card (A6000/L40S) or 80 GB (A100/H100)
for comfort at 1088px with Klein 4B. The config headers' VRAM estimates are
unverified; measure with `scripts/profile_memory.py` in the first GPU hour.

## 5. Plan

Each phase has a go/no-go gate. Do not start a later phase on hope.

### P0. Hygiene (local, ~1 day)

- P0.1 `resolve_dtype()`: bf16 on CUDA when `torch.cuda.is_bf16_supported()`.
  GradScaler stays disabled in that case. Update `test_device_resolution.py`.
- P0.2 Hard epsilon: `delta = eps * tanh(unet_out)` in the training loop and
  in `immunize_image_pil`. Config key `epsilon` (default 32/255). Test: output
  never exceeds eps.
- P0.3 Surrogate selection in blocks (`attack_manager.hold_batches`, default
  32). Test: swap count over N batches.
- P0.4 Defaults: `noise_target.enabled: false`, `attention_loss.enabled:
  false`, `flat_minima.enabled: false`, `clip_loss.enabled: false` in the new
  configs. Keep the modules; stop recommending them until an ablation earns
  them back.
- P0.5 Fix the MPS-host test failure in `test_attack_gradient_flow.py`
  (fake FLUX pipe must be on the resolved device, or the guard must accept
  the stub). Full suite green on this Mac.
- P0.6 Fix `research_v3.yml` stage ordering; remove the dead `load_path`.

Gate: tests green locally; a 10-batch stub-surrogate smoke run shows the
perturbation bounded by eps and loss1 falling.

### P1. Correct FLUX surrogate + loss (local code, GPU smoke, ~3-5 days)

- P1.1 New `flux_edit_attack.py::FluxKleinEditAttack(BaseAttack)`.
  Differentiable copy of the Klein edit path: VAE-encode the protected image
  with `.mode()`, patchify, BN-normalise, pack, build reference ids with
  T = 10; generation latents from `randn` with T = 0 ids; at each step
  `cat([latents, image_latents])`, transformer, slice to generation tokens,
  scheduler step; `gradient_timestep_fraction` applies to the transformer
  call as today; decode. Guidance: none for the distilled 4B (matches the
  pipeline). Optional mask mode via the Klein inpaint pipeline later.
- P1.2 Loss for edit-mode surrogates. Primary: loss1 = weighted L1 of the
  decoded edit against zeros (the paper's objective). Secondary: a
  **hidden-state distance** term: run the transformer once more under
  `no_grad` with the clean reference, hook the middle blocks, and maximise
  normalised L2 between clean and protected hidden states at the reference
  token positions. This is the DeContext-style objective that
  `attention_loss.py` was meant to be, and it does not saturate.
- P1.3 Replace `latent_loss` with normalised L2 (`||z_adv - z||^2 /
  ||z||^2`) and set its weight from a gradient-norm balance measured on the
  first GPU batches, not by guess.
- P1.4 Tests, same style as A1-A8: gradient from loss1 reaches the input
  through the edit path at gtf in {0.25, 0.5, 1.0}; reference tokens are
  never noised; hidden-state term is non-constant and carries gradient; a
  tiny NestedUNet lowers loss1 through the real path on CPU with a fake pipe.
- P1.5 `eval_multimodel.py`: label Klein as edit mode (it already is); add
  Kontext dev and Qwen-Image-Edit entries (inference, `enable_model_cpu_offload`).

Gate: on a GPU, 2k steps at 512px, Klein-4B-only config: loss1 trends down
>= 10% from its step-200 average; hidden-state distance rises; no NaN; VRAM
and step time recorded.

### P2. Evaluation protocol (local code, GPU run, ~2 days)

- P2.1 `scripts/eval_protocol.py` (or extend `eval_multimodel.py`): >= 50
  val images x 2 prompts x 2 seeds; social-media transform (resize 1080 +
  JPEG q80) before every edit; controls: clean, Gaussian noise at matched
  PSNR, PhotoGuard at matched PSNR; metrics: perturbation PSNR/SSIM/LPIPS,
  edit LPIPS and SSIM vs clean-edit, CLIP-score delta, protection rate at a
  stated threshold with a bootstrap CI; JSON + markdown table output.
- P2.2 Editor zoo in real modes: SD 1.5 inpaint, SD3.5 img2img (0.75, 28
  steps), FLUX.2 Klein 4B edit (4 steps), FLUX.1 Kontext dev edit (unseen),
  Qwen-Image-Edit (unseen, if VRAM allows), SDXL img2img (unseen).

Gate: the protocol runs end to end on the existing
`checkpoints/diffvax_trained.pth` and reproduces the SD 1.5 protection the
paper reports, and shows ~zero for the noise control at matched PSNR.

### P3. GPU validation, 512 -> 1088 (rented GPU, ~1-2 weeks of runs)

- P3.1 Signal run: Klein-4B edit surrogate only, 512px, eps 16/255 and
  32/255, 5-10k steps. Eval with P2. Go if protection rate on Klein > noise
  control by a clear margin at PSNR >= 36 dB.
- P3.2 Ablations, one knob each: hidden-state term on/off; latent L2 on/off;
  spectral on/off; gtf 0.5 vs 1.0; EoT on/off (measured through the
  social-media transform).
- P3.3 Ensemble: add SD3.5 img2img and SD 1.5 inpaint. Measure transfer to
  Kontext and Qwen-Image-Edit (unseen). Decide the mixture from the numbers.
- P3.4 Curriculum 512 -> 768 -> 1024 -> 1088 from the best 512 checkpoint.
  Native 1088 into Klein (multiple of 16; no downsample needed). Measure VRAM
  at each stage. Keep `gradient_timestep_fraction` and checkpointing as the
  VRAM levers; consider the 37M NestedUNet (H6) only if 512px loss plateaus.

Gate: at 1088px, protection against Klein and Kontext above the matched-PSNR
noise control with CI, LPIPS(original, protected) <= 0.05 and SSIM >= 0.97.
(Revised 2026-11-11: a full-budget 16/255 perturbation is ~24 dB PSNR, so the
earlier PSNR >= 36 dB gate conflicted with the budget; see the runbook.)

### P4. Black-box probe: Nano Banana (after P3, ~1 day + API cost)

- 50 clean + 50 protected images through the Gemini image-edit API with the
  same prompts. Score with the P2 metrics. Report the delta with a CI. Put
  the result, positive or negative, in the README. Do not extrapolate from
  white-box numbers.

### P5. Data scale-up (only if P3 passes)

- 2-5k images at >= 1088px, varied subjects, with person/face masks where
  available. Re-run P3.4 from the 172-image checkpoint.

## 6. Decisions needed from the user

1. GPU budget and card class (48 GB vs 80 GB). This sets whether Kontext dev
   can be a training surrogate (80 GB) or stays eval-only (48 GB).
2. Agree to drop CLIP-H, attention-entropy, flat-minima and noise-target
   from the default configs until an ablation earns them back.
3. Agree that Nano Banana is an evaluation target, not a training claim.
4. Perturbation budget target: eps 16/255 (near-invisible) vs 32/255
   (paper default, visible on smooth skies/skin at 1088px).

## 7. Risks

- Reference-token editors may still be robust: the clean tokens are a small
  minority of the sequence at 1088px (Klein: 4624 reference tokens vs 4624
  generation tokens plus text). If P3.1 shows no signal at eps 32/255 on
  Klein, the approach is unlikely to work on stronger editors, and the honest
  outcome is a negative result.
- Purification (resize + re-encode, diffusion purifiers, upscalers) is known
  to remove pixel perturbations. The social-media transform in P2 covers the
  default pipeline, not a determined adversary. State this limit.
- 172 images is small; a positive P3.1 may not generalise. P5 exists for
  that.
