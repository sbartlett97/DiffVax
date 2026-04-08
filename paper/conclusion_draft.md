# Paper Draft — Conclusion
# Status: complete draft (no GPU numbers needed)
# Date: 2026-04-08

---

## 6. Conclusion

We introduced DiffVax++, a multi-model, high-resolution, and JPEG-robust extension of DiffVax that addresses three deployment barriers preventing existing immunization methods from protecting images on real social media platforms.

Our first contribution identifies a structural advantage of patch-based high-resolution inference: overlapping 512×512 patches at stride=256 produce 1088×1088 immunizations that are **1.60× stronger** than direct 512×512 immunization due to perturbation accumulation across patch boundaries. This result is counter-intuitive — tiling is typically a smoothing operation — but holds robustly and requires no high-resolution retraining. The fully-convolutional NestedUNet generalizes to 1088×1088 without modification; the improvement is a structural property of Gaussian-blended patch overlap.

Our second contribution establishes multi-model training as a **product safety requirement**, not merely an accuracy improvement. The state-of-the-art FLUX-based purification attack [ZHAO2026] renders single-model (SD1.5) immunizations commercially worthless by recovering full editability. By training against FLUX.1-schnell in addition to SD1.5, H1a produces immunizations that resist this specific purification attack while also transferring to the held-out SD 3.5 architecture — validating that the shared VAE bottleneck is a generalizable attack surface across diffusion model families.

Our third contribution demonstrates that **standard Lp immunizations are silently defeated by social media upload pipelines** — Instagram's q≈75 and Twitter's q≈70 JPEG compression removes perturbation energy before any adversary action. We introduce the first JPEG-augmented immunization training using the Straight-Through Estimator, which directly forces perturbation energy into DCT quantization-survivor bands at the target quality range. The resulting H7 checkpoint maintains high EDR after q=75 compression, where competitors — including IDProtector, which explicitly avoids STE training — collapse. This is the first immunization method designed specifically for Instagram and Twitter deployment.

Each contribution addresses a different point of failure in the threat landscape: model coverage (the capable-adversary problem), resolution (the platform-scale problem), and compression robustness (the upload-pipeline problem). The simultaneous address of all three is what distinguishes DiffVax++ from the 2025 SOTA, which — despite SOTA claims — would fail silently on real social media deployments.

**Limitations and future work.** We do not address adaptive adversaries (e.g., q=60 JPEG + SR purification combined), and our gpt-image-edit evaluation is qualitative only. Training against FLUX.1-dev (rather than Schnell) may further improve purification resistance. Video immunization — extending the frame-level protection to temporal sequences — is an open problem. Finally, while we demonstrate cross-model transfer, a theoretical characterization of which architectural features determine transferability (VAE channel width, attention structure, conditioning mechanism) remains an open question.

We release trained checkpoints, evaluation scripts, and training code to facilitate reproducibility and future work on deployment-ready image immunization.

---

## Broader Impact

Image immunization has clear beneficial applications: protecting creator content from unauthorized manipulation, preserving photographic integrity in journalism, and giving individuals control over their own likeness. DiffVax++ makes these protections more robust and practically deployable at scale.

We acknowledge potential misuse: the same technique could be used to immunize synthetic (deepfake) content, making it harder for fact-checkers to re-edit and contextualize manipulated images. We do not anticipate this being the primary use case — an attacker generating synthetic content has no incentive to immunize it against editing — but it is a real risk that the community should monitor.

The primary societal benefit — protecting legitimate creators from unauthorized AI editing — outweighs the risk, and we release this work with the expectation that it will accelerate platform-level deployment of immunization tools.
