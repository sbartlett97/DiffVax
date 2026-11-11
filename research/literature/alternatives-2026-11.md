# Alternatives for one-shot, mobile image immunization against modern editors

**Date:** 2026-11-11 | **Scope:** literature from 2023 to 2026-08 | **Status:** desk review, no new experiments except one FLOP count

**Evidence standard.** Each claim cites a primary source (arXiv abstract or full text, official repo, or vendor page).
"Not verified in this review" means that this review did not open a primary source for the claim.
Numbers come from the papers' own tables unless marked otherwise. No paper here tests our exact setting:
1088 px, social-media JPEG and resize, FLUX.2 Klein plus closed editors, one-shot generator.

---

## 1. Key verdicts

1. **Open DiT editors can be disrupted, but the strongest results are per-image and fragile.**
   VETO transfers to FLUX.2-Klein-4B and reduces edit success there from 70% to 4.3% at ε=4/255.
   On its quantized FLUX.2-dev surrogate, plain JPEG raises edit success from 3% to 70% [VETO]. JPEG was not reported for Klein.
2. **Only one paper measures success against closed editors: MIRAGE.** It does not disrupt the editor.
   It makes the provider's moderation refuse the request. It reports >88% refusal on GPT-Image-2, Gemini-3.1-Flash-Image and Grok.
   It survives JPEG q50 and a 50% downscale. It costs about 15 min per image on an A40 GPU, and no code is public [MIRAGE].
3. **All other protections fail against closed editors or against cheap purification.**
   Off-the-shelf img2img models (FLUX, SD3, GPT-4o) remove protections without special tuning [Pleimling; Purify Once].
4. **Feed-forward generators exist and are small** (DiffVax 70 ms; IDProtector 0.17 s; FastProtect 0.04 s on GPU).
   No paper reports a phone deployment.
5. **The success metric must change.** DiT-era papers measure *identity or content leakage* (ISM, CLIP-I, DINO), not "edit not applied".
   DeContext states that it fails on scene-level prompts where text dominates [DeContext]. This matches our v1 result (style transfers apply fully).

---

## 2. State of the art against diffusion and DiT editing

| Method | Type | Budget | Editors measured | Measured effect | JPEG / resize | Code |
|---|---|---|---|---|---|---|
| PhotoGuard (2023) [PG] | Per-image PGD; encoder or full-pipeline attack | L∞ 16/255 typical | SD 1.5 inpainting / img2img | Edits become unrealistic on SD 1.5 | Weak (see §4) | Yes |
| AdvDM / Mist (2023) [AdvDM; Mist] | Per-image PGD on denoiser loss (+ texture target for Mist) | 8–16/255 | SD 1.x mimicry, img2img | Style mimicry and img2img degrade | Mist defeated by noisy upscaling [Hönig; protection list not verified in this review] | Yes |
| SDS protection (ICLR 2024) [SDS] | Per-image; shows the **encoder** is the weak point of LDMs | 16/255 | SD 1.x | 2× faster, half memory vs AdvDM | Not a focus | Yes |
| Glaze / Nightshade [Glaze; NS] | Per-image optimization | Perceptual budget | Style mimicry / training poisoning | Targets training, **not editing** | Defeated by noisy upscaling [Hönig] | Binaries only |
| EditShield (2023/24) [ES] | Per-image; shifts latent for instruction editors | Not stated in abstract | Instruction-guided editors (InstructPix2Pix-era) | Edits unrealistic, robust to instruction wording | Not verified in this review | Not verified |
| DiffusionGuard (2024/25) [DG] | Per-image; early-timestep objective + mask augmentation | 16/255 | SD inpainting | Beats PhotoGuard on mask robustness | Claims better resilience to noise removal | Yes |
| DiffVax (ICLR 2026) [DV] | **Feed-forward UNet++** | L1-penalized (κ in appendix) | SD 1.5 inpainting; IP2P qualitative | Edit SSIM 0.52 after JPEG vs 0.66 for PhotoGuard-D | JPEG 0.75 and CNN denoiser tested | Yes |
| Posterior Collapse Attack (TIFS 2025, venue inferred from DOI) [PCA] | Per-image; VAE encoder only (<4% of params) | Not stated in abstract | "VAE-based LDM variants" (FLUX/SD3 not named) | Claims better runtime/VRAM and cross-LDM transfer | Not verified | Yes |
| DCT-Shield (ICCV 2025) [DCT] | Per-image; optimizes DCT coefficients inside the JPEG pipeline | Not stated in abstract | Not named in abstract | Claims fewer artifacts, JPEG robustness | Designed for JPEG | Not found |
| BlurGuard (NeurIPS 2025) [BG] | Wrapper; region-adaptive blur of the noise, ~31 s/img A100 | 16/255 | SD 1.4/2.1, IP2P; SD3.5 and FLUX.1-dev in an appendix (not read) | Keeps 92.9% of FID disruption after the worst purifier vs 38–48% for baselines | Worst case over JPEG, noisy upscaling, DiffPure, GrIDPure, PDM-Pure | Yes |
| TDAE (T-PAMI 2026) [TDAE] | Plug-in; flat-minima image term + prompt-diverse text term | — | Cross-model | Claims SOTA cross-model; **fails on commercial editors** per MIRAGE | Not verified | Not verified |
| DeContext (2025) [DC] | Per-image, 800 steps; weakens multimodal attention | η=0.1 (pixel scale not stated) | FLUX.1-Kontext-dev, Step1X-Edit, 512 px | Kontext identity ISM 0.12 vs 0.32 best baseline; Step1X ISM 0.62→0.13 | **Not tested** | Yes |
| VETO (2026-07) [VETO] | Per-image MI-FGSM, 100 steps, ~57 s/img A100; maximizes reference↔canvas attention entropy in the **first double-stream block** | 4/255 chosen | FLUX.2 (quantized dev), Fibo-Edit; transfer to FLUX.2-Klein-4B | Klein-4B edit success 70%→4.3%; PhotoGuard needs ε≈12 for less effect | JPEG raises success to 70% (ε=4) and 63% (ε=12); crop 69% | "Will be released" |
| MIRAGE (2026-06) [MIR] | Per-image PGD, 5,000 steps, ~15 min/img A40; pushes CLIP/SigLIP/DINOv2/moderation embeddings toward an unsafe concept | 16/255 (8/255 ablation) | GPT-Image-2, Gemini-3.1-Flash-Image, Grok-Imagine | >88% refusal; PhotoGuard, BlurGuard, TDAE "very low" on APIs | No loss at JPEG q50, 50% downscale, blur; DiffPure → ~56%; CLIP-embedding removal → 12.5% | Not found |
| Universal semantic injection (2026) [UAP] | **One fixed perturbation** for all images | 10/255 | SD 1.5 white-box; SD 1.4/2.0, IP2P black-box | Beats universal baselines; near image-specific methods | Tested against JPEG, GrIDPure, DiffPure, noisy upscaling | Project page only |

**DiT-specific observations.**
- DeContext's best block set on Kontext is *single blocks 0–25* (ISM 0.13). Attacking all double blocks gives ISM 0.52. Single blocks 12–37 give 0.20. It samples only timesteps 980–1000, the earliest steps [DC].
- VETO hooks only the *first double-stream block* on FLUX.2 and states that the single-stream hook is 6× slower [VETO].
- Both results put the useful signal *early* in denoising. DeContext's abstract says "early-to-mid single blocks", and its ablation favors single blocks 0–25 over 12–37. VETO uses the first double-stream block.
- Our configs use `attention_loss.target_blocks: "middle"` (for example `configs/klein_edit_v2_512.yml`). This hooks the central third of the stack and drops the earliest blocks. `attention_loss.py` tries `transformer_blocks` (double-stream) before `single_transformer_blocks`. This review did not confirm which stack it hooks on Klein. If it hooks middle double-stream blocks, that is near the setting DeContext found weakest (all double blocks, ISM 0.52). This is a check to run, not a confirmed bug.
- `research/literature/survey.md` dates DeContext to December 2024. The arXiv date is December 2025.
- Purify Once re-optimized PhotoGuard, AdvDM and Mist against Step1X-Edit. Unpurified edits then had PSNR 17.8–20.1 dB, so DiT editors *can* be disrupted when the surrogate matches [PO].

**VAE-encoder attacks on 16-channel VAEs.** We found no paper that measures an encoder attack against the FLUX, SD3 or Qwen-Image VAEs. Purify Once reports only partial transfer across versions: SD 1.5-optimized protections still degrade SD 2.0 edits (14–16 dB PSNR), but some methods are slightly weaker there. Its "VAE-Trans" fine-tuned encoder purifies them [PO]. An ensemble encoder attack is therefore an untested hypothesis, not a known result.

---

## 3. Amortized and feed-forward methods

| Method | Generator | Size | Speed | Targets | Notes |
|---|---|---|---|---|---|
| DiffVax [DV] | UNet++ | Not stated in paper | 0.07 s, 5.6 GB GPU | SD 1.5 inpainting | Paper says it is "primarily model-specific" |
| IDProtector (CVPR 2025) [IDP] | ViT-S/8 at 224 px; noise resized to full image | Not stated (ViT-S is about 22M params, not verified for this model) | 0.17 s GPU | ID-encoder generators (InstantID, IP-Adapter, PhotoMaker) | ε=9/255; transfers to Midjourney ISM 0.199→0.098; JPEG q85 nearly lossless |
| FastProtect (CVPR 2025) [FP] | Mixture of pre-trained perturbations, chosen by input latent | Small | 0.04 s GPU, 2.8 s CPU [UAP Table 4] | SD personalization (DreamBooth-type) | 200–3500× faster than PGD baselines |
| Universal semantic injection [UAP] | None: one stored pattern | One image-sized tensor | ~0 | SD 1.5 family | A fixed pattern is easy to estimate and subtract once an attacker gets it (our inference, not tested in the paper) |
| Our NestedUNet (measured here) | UNet++ `[32..512]` | **9.16M params** (≈18 MB fp16, ≈9 MB int8) | **276 GFLOPs at 512 px; 1.25 TFLOPs at 1088 px** | FLUX.2 Klein surrogate | Wide `[64..1024]` variant: 36.6M params (≈73 MB fp16), 4.98 TFLOPs at 1088 px |

The FLOP figures come from `torch.utils.flop_counter` (2 FLOPs per MAC) on the repo model at batch 1.
The `NestedUNet` docstring says "~1.8M params". The real count is 9.16M.

**Mobile.** No paper in this review reports Core ML, TFLite or ONNX deployment of an immunizer.
IDProtector's design (predict at 224 px, resize the noise) is the closest to a phone budget.
A phone estimate for our model: 1.25 TFLOPs at 1088 px is heavy. Full-resolution 32-channel activations are about 76 MB each in fp16. **This is an estimate, not a measurement.** Expect about 1 s or more on a recent Neural Engine, plus memory pressure. The 512 px path (276 GFLOPs) with an upsampled perturbation fits the 1–2 s target more easily.

---

## 4. Robustness and purification

| Study | Finding |
|---|---|
| Hönig et al., ICLR 2025 [HON] | Popular protections (Glaze, Mist and Anti-DreamBooth per our recall; list not verified in this review) fail against low-effort methods such as noisy upscaling. A user study confirms it. A third-party review reports >40% median mimicry success for noisy upscaling (not verified in this review). |
| Tang et al. 2025 [TANG] | PhotoGuard, Mist and Glaze on SD 1.5 can make edits follow the prompt *more* closely. Noise regions get replaced by prompt-guided content. |
| Pleimling et al., SaTML 2026 [PLE] | FLUX.1-dev, SD3 and GPT-Image-1 with a "denoise" prompt remove protections and watermarks. Attack-aware variants with the denoiser in the loop also fail. VINE survives FLUX (TPR 0.878) but not a 0.7% crop. |
| Purify Once, 2026 [PO] | Six protections (PhotoGuard, AdvDM, Mist, SDS, DiffusionGuard, AdvPaint) at 16/255. EditorClean gains 3–6 dB PSNR and 50–70% lower FID. Seedream, Qwen-Image, Step1X, ChatGPT-4o and Gemini "successfully perform semantic edits on protected images" (qualitative only). |
| Real-world transforms, 2026 [RWT] | JPEG q75 plus half-size Lanczos resize moves edit outputs back toward clean outputs for FaceLock, FaceShield, Mist, PhotoGuard and others. DF-RAP survives JPEG alone but not JPEG + resize. |
| VETO [VETO] | JPEG raises FLUX.2 edit success from 3% to 70% at ε=4. Augmentation helps for flips and crops, but not for JPEG. |
| BlurGuard [BG] | Low-pass shaping of the noise keeps most of the disruption under the worst of seven purifiers. This is the best evidence that *spectral shape* matters. |
| MIRAGE [MIR] | Refusal survives JPEG q50, 50% downscale, blur and screenshots. A social-media square crop drops GPT-Image to ~72%. DiffPure drops it to ~56%. A CLIP-embedding removal attack drops it to 12.5%. |

**Implication for social media.** Instagram-class pipelines resize to about 1080 px and recompress (exact settings not verified in this review).
For editor-disruption perturbations, JPEG plus resize is the measured failure point in three independent 2026 papers [RWT; VETO; PO].
The only protection with reported survival is semantic-embedding misalignment (MIRAGE). It is also the easiest for an adaptive attacker to remove.
No method survives a determined attacker with a local img2img model.

---

## 5. Non-adversarial and adjacent alternatives

None of these stop an edit. Each changes what happens *around* the edit.

- **Moderation-triggered refusal (MIRAGE).** It is adversarial, but its target is the provider's safety stack, not the editor. It is the only measured method against closed editors. Product risks are in §8.
- **Provider-side cooperation (GuardDoor).** The provider fine-tunes its image encoder with a backdoor. Owner images carry a trigger, and protected edits come out meaningless. It claims robustness to compression. It needs provider cooperation, so it does not help today [GD].
- **Invisible watermarks.** Watermark Anything (Meta, ICLR 2025) embeds localized 32-bit messages and has open weights [WAM]. VINE (ICLR 2025) survives many edits. W-Bench shows that most of 11 methods fail after editing [VINE]. Stable Signature and SynthID mark *generated* content and are not owner-side tools (not verified in this review).
- **Provenance.** C2PA signing at capture (Truepic Lens SDK, iOS/Android, on-device signing) [TP]. Adobe Content Authenticity adds signed credentials, an invisible watermark, and a "do not train" preference that Adobe calls a preference, not a guarantee [ADB]. Apple Reference Image (press reports, September 2026) signs pixel data at capture on iPhone 18 Pro. It is not C2PA, and it reveals changes but does not block them [APL].
- **Watermark as a protection target.** We found no evidence that any closed editor refuses or degrades an edit because of a watermark or C2PA manifest.

---

## 6. Existing products

| Product | What it offers | Platform / compute | Limits |
|---|---|---|---|
| Glaze / Nightshade [GLZ] | Style cloak / training poison | Desktop; >4 GB NVIDIA GPU or Apple M-series; up to ~60 min/image at top quality on a laptop | FAQ: "there will not be a mobile Android/iOS version". Targets training, not editing. |
| WebGlaze [GLZ] | Glaze on cloud GPUs | Web, invite-only, daily and weekly limits | Artists only |
| Mist | Style-mimicry protection | Desktop GPU (not verified in this review) | Defeated by noisy upscaling [HON] |
| PhotoGuard demo [PGN] | Research demo | Web (Hugging Face) | Works reliably only on Stable Diffusion |
| CloakAI [CLK] | Claims "invisible adversarial cloaking" against training, deepfakes and style theft | Web | Only self-reported figures; no method or evidence published |
| Loti [LOT] | Deepfake detection and takedown | Service | Detection, not prevention |
| Adobe Content Authenticity [ADB] | Content Credentials, watermark, AI-training preference | Free web app | Does not prevent editing |
| Truepic Lens [TP] | C2PA capture SDK | iOS/Android, on-device signing | Provenance only |
| Imatag, Steg.AI, Digimarc [IMT] | Commercial invisible watermarks | SaaS/API (Imatag); mobile SDKs not verified | Tracing, not prevention |
| Apple Reference Image [APL] | Signed "digital negative" | iPhone 18 Pro main camera only | Authenticity only; not retrospective |

**Gap.** We found no consumer mobile app with published evidence of anti-editing protection.
This is a statement about this search, not proof that none exists.
The provenance side (Apple, Adobe, Truepic) is already well served and partly built into the OS.

---

## 7. Feasibility verdict

**(a) Open DiT editors (FLUX.2 Klein, Kontext, Qwen-Image-Edit, SD3.5).**
- *Before JPEG:* strong disruption at ε=4–16 is shown for FLUX.2/Klein and Kontext, per image, in 1–15 min of GPU time [VETO; DC].
- *Amortized:* plausible. DiffVax and IDProtector show that a generator can approximate per-image attacks on UNet-era models. No paper shows this on a DiT editor.
- *After social-media JPEG + resize:* not shown by any paper at an imperceptible budget. VETO measures the JPEG failure on quantized FLUX.2-dev, a close relative of Klein.
- *Qwen-Image-Edit:* no protection paper measures it.
- *Realistic target:* "the edited output does not keep the subject's identity or details", measured with ISM, CLIP-I and DINO. "The edit fails" is not a realistic target. Text-dominant scene edits will still apply.

**(b) Closed services (Gemini, GPT-Image, Midjourney).**
- Editor disruption: no evidence. The measured attempts against closed *instruction editors* fail [MIR]. Purify Once reports the same, but only qualitatively [PO].
- Refusal via moderation: measured at >88%, robust to JPEG, but per-image (15 min GPU), visible at 16/255, and defeated by an adaptive attacker. It also exploits a system that providers change without notice. Grok's rate drifted inside one month [MIR].
- IDProtector's Midjourney result (ISM 0.199→0.098) applies to identity-reference generation, not instruction editing [IDP].

**(c) One-shot on a phone.** A 9M-param generator fits the size target. Latency at 1088 px is the open risk. The protection strength is the bigger open risk.

---

## 8. Ranked candidates

### 1. Gating experiment: per-image PGD on Klein with a VETO-style loss and JPEG/resize EoT, at ε ≤ 8/255
- **Why.** It is the cheapest decisive test. If per-image PGD cannot survive JPEG plus resize at an invisible budget, no amortized generator can.
- **Details.** Maximize attention entropy in the first double-stream block (VETO) and in early single blocks (DeContext). Use the first, noisiest step of the 4-step schedule. Add differentiable JPEG and resize from our `eot.py`. Add a BlurGuard-style spectral constraint. Add the VAE-encoder ensemble (FLUX.2, FLUX.1, SD3, Qwen-Image) as an *auxiliary* loss only. Score with ISM, CLIP-I and DINO against the clean edit, after real JPEG q75 + resize.
- **Expected strength.** Strong before JPEG (VETO). Unknown after JPEG; VETO's evidence points to weak.
- **Cost.** 1–2 days on one A100. Most parts already exist in the repo.
- **Risks.** A negative result is likely. It is still useful because it caps all amortized work.
- **Start from.** VETO [VETO], DeContext code [DC], BlurGuard code [BG].
- **On-device fit.** Not applicable (it is a research upper bound).

### 2. Amortized generator distilled from candidate 1
- **Why.** The only path that meets "one-shot on a phone". DiffVax, IDProtector and FastProtect show that amortization works on older models.
- **Details.** First confirm which stack and how many blocks `attention_loss` hooks on Klein. Then test early double-stream against early single-stream blocks (§2). Optionally distil from PGD targets of candidate 1. Optionally add 1–3 on-device refinement steps through a small proxy (for example a distilled VAE encoder); this is untested.
  Predict at 512 px and upsample. This cuts FLOPs 4.5× and gives a lower-frequency perturbation, which BlurGuard and VINE suggest survives purification better (untested for us).
- **Expected strength.** At most candidate 1's strength, minus an amortization gap.
- **Cost.** 1–2 weeks of GPU training with the current pipeline.
- **Risks.** It inherits the JPEG failure. Model-specific: Kontext and Qwen need their own surrogates.
- **On-device fit.** Good. 9.16M params, about 9–18 MB, convolutions and GroupNorm only, so Core ML and TFLite conversion is direct. 276 GFLOPs at 512 px. 1088 px latency is unmeasured; profile on a device before you commit.
  Battery: one inference of about 1–2 s per photo is a small one-off cost (estimate, unmeasured).

### 3. MIRAGE-style refusal trigger for closed services, amortized
- **Why.** The only measured effect on Gemini and GPT-Image, and it survives JPEG and resize.
- **Expected strength.** High today against casual users. Low against an adaptive attacker or after a provider update.
- **Cost.** The paper has no code. The ensemble (CLIP, SigLIP, DINOv2, ShieldGemma 2, Llama Guard 3 Vision) is all open. A generator version is untested. Image-level embedding attacks are a good fit for amortization because they need no diffusion surrogate (our inference).
- **Risks.** Severe for a product. The default target concept is sexually explicit content. Social platforms run similar classifiers on upload, so the user's own post may be flagged or the account penalized. It attacks provider safety systems, which may breach their terms. Open question: does a *benign* refusal trigger (for example a "real identifiable person" or "copyrighted content" concept) give useful refusal rates?
- **Start from.** MIRAGE [MIR].
- **On-device fit.** A generator would match candidate 2. Per-image PGD with 8 vision encoders is not feasible on a phone at all. A generator would have the same battery cost as candidate 2 (estimate).

### 4. Universal or mixture-of-perturbations (FastProtect / semantic injection)
- **Why.** Near-zero device cost. The UAP paper reports near image-specific strength at 10/255 on SD 1.5.
- **Expected strength.** Weaker than candidate 2. No DiT evidence in the main text.
- **Cost.** Low. Train with the existing surrogate code.
- **Risks.** A fixed pattern is easy to extract and subtract after release. Perceptibility is not image-adaptive.
- **On-device fit.** Best: an image add plus a resize. Use it as a fallback tier for old phones. Battery cost is negligible.

### 5. Provenance and watermark layer (complement, not protection)
- **Why.** It gives an honest, durable claim: "this photo can be shown to be the original". It also supports takedown.
- **Expected strength.** Strong for authenticity. Zero for prevention.
- **Cost.** Low with WAM (open weights) or a C2PA SDK.
- **Risks.** Apple and Adobe already offer this. It is not a differentiator alone.
- **On-device fit.** WAM model size is not stated in the paper. C2PA signing is light. Battery cost is small (estimate, unmeasured).

---

## 9. Product implications

1. **Claim only what is measured.** Use a claim such as "edits made with model X are less likely to keep your face, before recompression". Do not claim "AI can't edit this photo". Over-claiming creates liability, and it repeats the "false sense of security" critique [HON; PLE].
2. **Social-media recompression is the main technical risk.** Protect the file the user posts, then test that same file after the real upload path (download it back from each platform).
3. **Closed editors.** No evidence supports disruption. Refusal works today but is fragile and has a self-harm risk for the user (§8.3).
4. **Adaptive attackers.** Any local img2img pass removes current protections [PLE; PO]. Position the app as raising the cost for casual misuse, not as stopping a determined attacker.
5. **On-device compute.** The 9M generator fits the size budget. Profile 1088 px latency, memory and battery before you design the UX around it.
6. **Moving target.** Editors change monthly. A product needs a pipeline to re-train the generator and an evaluation harness against current editors, not a one-time model.
7. **Positioning.** No mobile anti-editing app with evidence was found (§6). A combined "protect + prove" app (perturbation plus provenance) has a more honest value story than perturbation alone.

---

## References

- [PG] Salman et al., PhotoGuard, ICML 2023. https://arxiv.org/abs/2302.06588
- [PGN] PetaPixel coverage of the PhotoGuard demo. https://petapixel.com/2023/03/02/photoguard-stops-your-pictures-being-manipulated-by-ai/
- [AdvDM] Liang et al., AdvDM, ICML 2023. https://arxiv.org/abs/2302.04578
- [Mist] Liang et al., Mist. https://arxiv.org/abs/2305.12683
- [SDS] Xue et al., ICLR 2024. https://arxiv.org/abs/2311.12832 ; code https://github.com/xavihart/Diff-Protect
- [Glaze] Shan et al., USENIX Sec 2023. https://arxiv.org/abs/2302.04222
- [NS] Shan et al., Nightshade. https://arxiv.org/abs/2310.13828
- [ES] Chen et al., EditShield. https://arxiv.org/abs/2311.12066
- [DG] Choi et al., DiffusionGuard. https://arxiv.org/abs/2410.05694 ; https://github.com/choi403/DiffusionGuard
- [DV] DiffVax, ICLR 2026. https://arxiv.org/abs/2411.17957
- [PCA] Guo et al., Posterior Collapse Attack. https://arxiv.org/abs/2408.10901 ; https://github.com/ZhongliangGuo/PosteriorCollapseAttack
- [DCT] Bala et al., DCT-Shield, ICCV 2025. https://arxiv.org/abs/2504.17894
- [BG] BlurGuard (Korea University / Microsoft Research; first author not verified), NeurIPS 2025. https://arxiv.org/abs/2511.00143 ; https://github.com/jsu-kim/BlurGuard
- [TDAE] Zhang et al., Towards Transferable Defense Against Malicious Image Edits. https://arxiv.org/abs/2512.14341
- [DC] Shen, Cui, Yang, DeContext as Defense. https://arxiv.org/abs/2512.16625 ; https://github.com/LinghuiiShen/DeContext
- [VETO] Grebe et al., VETO: Towards Protecting Images From Frontier AI Editing. https://arxiv.org/abs/2607.27292
- [MIR] Nasery et al., MIRAGE: Protecting against Malicious Image Editing via False Moderation. https://arxiv.org/abs/2606.26199
- [UAP] Lee et al., Universal Image Immunization via Semantic Injection. https://arxiv.org/abs/2602.14679
- [FP] Ahn et al., FastProtect, CVPR 2025. https://arxiv.org/abs/2412.11423
- [IDP] Song et al., IDProtector, CVPR 2025. https://arxiv.org/abs/2412.11638
- [GD] Zeng et al., GuardDoor. https://arxiv.org/abs/2503.03944
- [HON] Hönig, Rando, Carlini, Tramèr, ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/af800ac69aea2fb5c4bcf8c6a5f3c79d-Abstract-Conference.html ; https://arxiv.org/abs/2406.12027
- [TANG] Tang et al., Is Perturbation-Based Image Protection Disruptive to Image Editing? https://arxiv.org/abs/2506.04394
- [PLE] Pleimling et al., Off-The-Shelf Image-to-Image Models Are All You Need, SaTML 2026. https://arxiv.org/abs/2602.22197
- [PO] Zhao et al., Purify Once, Edit Freely. https://arxiv.org/abs/2603.13028
- [RWT] Do Protective Perturbations Really Protect Portrait Privacy under Real-world Image Transformations? https://arxiv.org/abs/2604.23688
- [SURV] Zhang et al., Adversarial Attacks for Good (survey). https://arxiv.org/abs/2608.04314
- [VINE] Lu et al., W-Bench and VINE, ICLR 2025. https://arxiv.org/abs/2410.18775 ; https://github.com/Shilin-LU/VINE
- [WAM] Sander et al., Watermark Anything, ICLR 2025. https://arxiv.org/abs/2411.07231 ; https://github.com/facebookresearch/watermark-anything
- [GLZ] Glaze FAQ and WebGlaze. https://glaze.cs.uchicago.edu/faq.html ; https://glaze.cs.uchicago.edu/webglaze.html ; https://glaze.cs.uchicago.edu/guide.html ; https://nightshade.cs.uchicago.edu/userguide.html
- [CLK] CloakAI. https://app.cloakit.ai/
- [LOT] GeekWire on Loti. https://www.geekwire.com/2025/seattle-startup-loti-lands-16m-to-help-celebs-politicians-and-ceos-detect-deepfakes/
- [ADB] Adobe Content Authenticity app coverage. https://www.maginative.com/article/adobe-launches-free-web-app-for-you-to-add-content-credentials-to-your-work/ ; https://www.dezeen.com/2025/05/15/adobe-content-authenticity-app-empower-creators-protect-work/amp/
- [TP] Truepic Lens / C2PA capture. https://truepic.com/c2pa/capture
- [IMT] Imatag coverage. https://techcrunch.com/2020/03/24/imatag-inserts-invisible-watermarks-to-track-images-around-the-web
- [APL] Apple Reference Image (press). https://www.computerworld.com/article/4222803/apple-just-gave-every-iphone-photo-a-digital-alibi.html
