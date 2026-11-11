# GPU validation runbook (RunPod, 80 GB)

Supersedes the 2026-07 runbook (it targeted the img2img surrogates and
`research_v3.yml`). Everything the CPU suite can prove is proven; this is the
empirical part. Rationale and numbers behind each choice:
`research/evaluation-2026-11-11.md`. Each stage has a go/no-go gate. Do not
start a stage whose predecessor failed.

## 0. Pod and storage

| Item | Choice |
|---|---|
| GPU | one 80 GB card (A100 80GB or H100 80GB), Secure Cloud so it is not preempted |
| Template | RunPod PyTorch (its CUDA build of torch is used as is) |
| Container disk | 50 GB |
| Network volume at `/workspace` | **150 GB** for stages 1-3 (Klein 4B ~24 GB + data + outputs). **350 GB** if you also pre-fetch the full evaluation zoo (about 230 GB of models, sizes from the Hub API on 2026-11-11). |
| Secrets | `HF_TOKEN` with read access to the **private** dataset `sbartlett97/diffvax-high-res` and to any private checkpoints under `sbartlett97/`; licence accepted on SD3.5 Medium and FLUX.1 Kontext dev if you use them. FLUX.2 Klein 4B needs no licence or token. |

```bash
git clone https://github.com/sbartlett97/DiffVax.git && cd DiffVax   # or rsync your working tree
export HF_TOKEN=hf_xxx
bash scripts/runpod_setup.sh            # add --with-zoo for the eval editors
```

The script checks the GPU, builds `/workspace/venv`, installs
`requirements-pod.txt`, runs the test suite, downloads the dataset and
pre-fetches Klein 4B. Read any test failure before continuing: the suite was
only run on CPU/MPS; CUDA is first exercised on this pod.

## 1. Memory and speed profile (15 min)

```bash
python scripts/profile_memory.py --config configs/klein_edit_512.yml \
    --resolutions 512 768 1024 1088 --batch-sizes 1
```

Record peak GB per resolution and seconds per step. These two numbers size the
whole budget: total GPU-hours = steps x seconds/step. A 512px signal run is
about 10,000 steps (30 epochs x ~344 steps). If 1088px peak exceeds ~70 GB,
lower `gradient_timestep_fraction` to 0.25 before anything else.

Gate: 1088px fits in 80 GB at batch 1.

### 1b. Real-weight parity (5 min, once)

```bash
python scripts/parity_check_klein.py --sizes 512 1088
```

The CPU parity test uses tiny random weights. This runs the real
`Flux2KleinPipeline` (is_distilled=true, 4 steps, guidance 1.0 per the model
card) against `FluxKleinEditAttack.attack()` with identical latents. At 512px
nothing is resized, so expect mean abs diff ~1e-3 (bf16); it exits non-zero
above 0.01. At 1088px the editor shrinks the reference to 1024px with LANCZOS
and ours uses antialiased bicubic, so only a loose match is expected.
Gate: "PARITY OK". If it fails, stop: the surrogate is not the editor.

### 1c. Protocol sanity on a case where protection is KNOWN to exist (30 min)

A `NO EFFECT` verdict on Klein only means something if this protocol can see
protection when it is there. The published SD 1.5 checkpoint is that case:

```bash
python scripts/eval_protocol.py --checkpoint checkpoints/diffvax_trained.pth \
    --editors "SD 1.5 Inpainting" --resolution 512 --editor-resolution 512 \
    --no-transform --max-images 20 \
    --output-dir /workspace/outputs/protocol_sanity_notransform
python scripts/eval_protocol.py --checkpoint checkpoints/diffvax_trained.pth \
    --editors "SD 1.5 Inpainting" --resolution 512 --editor-resolution 512 \
    --max-images 20 --output-dir /workspace/outputs/protocol_sanity_transform
```

`--resolution 512 --editor-resolution 512` matter: that checkpoint was trained at
512px and the paper's editor runs at 512, whereas the dataset is 1088px and the
zoo entry defaults to 1024. Without them the test measures a different
resolution than the one the protection was trained for and a NO EFFECT would be
meaningless. That checkpoint is unbounded (no epsilon), so the random-sign control uses each
image's MEASURED L-inf. Gate: verdict `PROTECTED` without the transform. With
the transform it may weaken (that is a real finding about JPEG robustness, not a
protocol failure). If even the no-transform run reads `NO EFFECT`, the protocol
or the metric is broken; fix that before spending GPU time on Klein.

## 2. Loss balance check (30 min)

Run 200 steps with each optional term toggled and compare per-term gradient
norms on the perturbation net (the trainer logs per-term values; add a
temporary print of `unet.final.weight.grad.norm()` per term if needed). Goal: no
single term more than ~10x the gradient norm of loss1. Adjust
`latent_loss.weight` and `hidden_state_loss.weight` in a COPY of the config.

## 3. Signal run, 512px (stage P3.1)

```bash
python scripts/train.py --config configs/klein_edit_512.yml \
    --output-dir /workspace/outputs/klein512 2>&1 | tee /workspace/klein512.log
```

Watch, in order:

1. `[Scaler]` lines should not appear as a problem: the compute dtype is bf16 on
   this card, so GradScaler is disabled. If it is enabled (pre-Ampere card),
   expect the old scale-collapse failure mode and stop.
2. `loss1` trending down (go: >= 10% below its step-200 average by ~2k steps).
3. `HS:` (reference hidden-state cosine) should fall from ~1.0. Flat at 1.0
   means the perturbation is not changing the reference representation.
4. No NaN abort. Every abort writes an emergency checkpoint and a JSON log.

Then run the controlled evaluation on the best checkpoint:

```bash
python scripts/eval_protocol.py \
    --checkpoint /workspace/outputs/klein512/models/1/*_best.pth --epsilon 16 \
    --editors "FLUX.2 Klein" --resolution 512 --editor-resolution 512 \
    --max-images 20 --seeds 0 1 \
    --output-dir /workspace/outputs/protocol_klein512
```

Gate (all three): verdict `PROTECTED` for FLUX.2 Klein against the
matched-PSNR control (CI lower bound of the excess above 0; the same-L-inf
column is a stricter secondary, since saturated +/-16 sign noise is ~24 dB and
much louder than a typical perturbation); perturbation LPIPS(original,
protected) <= 0.05 and SSIM >= 0.97; max L-inf <= 16.

Imperceptibility note: a perturbation that uses the FULL 16-level budget on
every pixel is only ~24 dB PSNR, so a PSNR gate in the mid-30s is not reachable
by the budget alone. The budget caps the worst case; how quiet the average
pixel is comes from `alpha` (soft L1) and `spectral_loss.weight`. If LPIPS/SSIM
fail while the verdict is `PROTECTED`, raise `alpha` (try 2, then 4) before
anything else. PSNR is reported for reference only. If the verdict is `NO EFFECT` at 16/255, stop and report: the
method does not work at this budget on this surrogate. Do not proceed to 1088px
hoping it improves. (One diagnostic is permitted: rerun once at `epsilon: 32`
to see whether any signal exists at all; label it as such.)

## 4. Ablations (stage P3.2), one knob each, 512px, ~half the steps

`hidden_state_loss.enabled`, `latent_loss.enabled`, `spectral_loss.enabled`,
`flux_attack.gradient_timestep_fraction` 0.5 vs 1.0, `eot.enabled` (judged
THROUGH the social-media transform, which eval_protocol applies to all
conditions). Rank by the excess-over-control CI, not the training loss.

## 5. Transfer and ensemble (stage P3.3)

Evaluate the best 512px checkpoint on editors NOT trained against:

```bash
python scripts/eval_protocol.py --checkpoint <best.pth> --epsilon 16 \
    --editors "FLUX.2 Klein" "SD3.5 Medium" "FLUX.1 Kontext" "Qwen-Image-Edit" \
    --resolution 512 --editor-resolution 512 \
    --max-images 50 --output-dir /workspace/outputs/protocol_transfer
```

Kontext and Qwen-Image-Edit use the same reference-token mechanism as the
surrogate, so they are the fair transfer test. SD3.5 img2img is a different
mechanism and is expected to be unaffected. If transfer is absent, adding a
second surrogate (Kontext, 80 GB) is the next experiment, not more epochs.

## 6. 1088px (stage P3.4)

```bash
python scripts/train.py --config configs/klein_edit_1088.yml \
    --checkpoint /workspace/outputs/klein512/models/1/*_best.pth \
    --output-dir /workspace/outputs/klein1088
```

EoT is on (JPEG, resize, blur, noise). Re-run `eval_protocol.py` with
`--resolution 1088` (no `--editor-resolution`, so Klein applies its own
1024*1024 area cap) and `--max-images 50`.

Gate: protection verdict holds at 1088px on Klein and on at least one unseen
reference-token editor; LPIPS(original, protected) <= 0.05 and SSIM >= 0.97 on
the 1088px validation images.

## 7. Black-box probe: Nano Banana (stage P4)

No gradients exist; this is measurement only. Needs a Gemini API key (not set
up yet). Upload N=50 clean and the same 50 protected images with identical edit
prompts, score the returned edits with the same LPIPS/CLIP code against the
clean-image edits. Report the paired difference with a bootstrap CI whatever
its sign. Do not extrapolate white-box results to it.

## Cost control

- Use `--max-images 20` for iteration, 50 for the reported numbers.
- Stop the pod when idle: the network volume keeps `/workspace`, so setup is
  not repeated. Checkpoints are written every 5 epochs.
- Budget the first block of hours from stage 1's seconds/step, then top up.
