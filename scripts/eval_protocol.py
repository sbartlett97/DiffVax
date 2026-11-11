#!/usr/bin/env python3
"""Controlled protection evaluation (see src/diffvax/protocol.py for the why).

For each editor, each validation image x prompts x seeds:
  clean / protected / Gaussian-noise@matched-PSNR / random-sign-noise@same-L-inf
are each passed through a social-media transform (resize 1080 + JPEG q80),
edited with the SAME seed, and compared to the clean edit with LPIPS.
Protection = d(protected) - d(noise control) with image-level bootstrap CIs.

Example (80 GB GPU):
    python scripts/eval_protocol.py --checkpoint outputs/models/1/x_best.pth \\
        --epsilon 16 --editors "FLUX.2 Klein" "SD3.5 Medium" \\
        --max-images 50 --output-dir outputs/protocol_klein512

Unseen transfer targets (opt in; large downloads):
    --editors "FLUX.1 Kontext" "Qwen-Image-Edit"

--editors "STUB" runs a CPU blur "editor" for pipeline smoke tests only.
"""
import argparse
import json
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, "src"))
sys.path.insert(0, _script_dir)

import numpy as np
from PIL import Image, ImageFilter

from diffvax import protocol as P
from diffvax.utils import (
    immunize_image_pil, load_perturbation_net, set_seed_lib,
)


def _make_divergence(kind: str):
    if kind == "l1":
        return lambda a, b: float(
            np.abs(np.asarray(a.convert("RGB"), np.float32)
                   - np.asarray(b.convert("RGB"), np.float32)).mean() / 255.0)
    try:
        import lpips
    except ImportError as exc:
        raise SystemExit("LPIPS needed: pip install lpips (or --divergence l1)") from exc
    import torch
    net = lpips.LPIPS(net="alex", verbose=False).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev)

    def to_t(im):
        a = torch.from_numpy(np.asarray(im.convert("RGB"), np.float32) / 127.5 - 1.0)
        return a.permute(2, 0, 1)[None].to(dev)

    def div(a, b):
        if a.size != b.size:
            b = b.resize(a.size, Image.LANCZOS)
        with torch.no_grad():
            return float(net(to_t(a), to_t(b)).item())
    return div


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help=".pth, save_pretrained dir, or Hub id")
    ap.add_argument("--epsilon", type=float, default=None,
                    help="pixel-unit bound the checkpoint was trained with (raw .pth only; "
                         "Hub/save_pretrained checkpoints carry it in config.json)")
    ap.add_argument("--nb-filter", type=int, nargs="+", default=None)
    ap.add_argument("--data-dir", default=os.path.join(_project_root, "data"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--mask-type", default="person")
    ap.add_argument("--max-images", type=int, default=50)
    ap.add_argument("--prompts-per-image", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--editors", nargs="+", default=["FLUX.2 Klein"])
    ap.add_argument("--social-max-side", type=int, default=1080)
    ap.add_argument("--social-quality", type=int, default=80)
    ap.add_argument("--resolution", type=int, default=None,
                    help="LANCZOS-resize each original to NxN BEFORE protecting. Match the "
                         "resolution the checkpoint was trained at (512 for the signal run and "
                         "the legacy SD1.5 checkpoint; 1088 for the production run)")
    ap.add_argument("--editor-resolution", type=int, default=None,
                    help="override the zoo entry's native_resolution so run_model does not "
                         "rescale (512 for the legacy SD1.5 sanity run and the 512px gate; "
                         "leave unset at 1088 so Klein applies its own area cap)")
    ap.add_argument("--mask-gate-perturbation", action="store_true",
                    help="keep the perturbation only in the protected subject (mask==0; dataset "
                         "convention 1=background) and restore the original elsewhere, as the "
                         "ORIGINAL DiffVax app does. Controls are gated to the same region.")
    ap.add_argument("--bn-train-mode", action="store_true",
                    help="run the perturbation net in train mode (per-image BatchNorm "
                         "statistics). The original app never calls .eval(), so this is how "
                         "the published BatchNorm checkpoint was actually used")
    ap.add_argument("--no-transform", action="store_true",
                    help="skip the social-media transform (sanity runs on the legacy SD1.5 checkpoint)")
    ap.add_argument("--control-eps", type=int, default=None,
                    help="L-inf (uint8) of the random-sign control; default = the MEASURED "
                         "L-inf of each protected image (right for unbounded legacy checkpoints)")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="success = d(protected) - d(control) > margin")
    ap.add_argument("--divergence", choices=["lpips", "l1"], default="lpips")
    ap.add_argument("--n-samples", type=int, default=3, help="rows in the saved samples_<editor>.png")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--output-dir", default=os.path.join(_project_root, "outputs", "protocol"))
    args = ap.parse_args()

    set_seed_lib(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    eps_px = args.control_eps   # None -> measured per image

    import eval_multimodel as EM
    entries, _res = EM.load_eval_entries(
        args.data_dir, args.mask_type, args.max_images, split=args.split)
    net = load_perturbation_net(
        args.checkpoint, nb_filter=args.nb_filter, epsilon=args.epsilon)
    transform = (lambda im: im.convert("RGB")) if args.no_transform else (
        lambda im: P.social_media_transform(im, args.social_max_side, args.social_quality))
    divergence = _make_divergence(args.divergence)

    # ---- protect once per image; build all conditions (post-transform) ----
    if args.bn_train_mode:
        net.train()
    from skimage.metrics import structural_similarity as _ssim
    prepared, pert = [], {"psnr": [], "ssim": [], "lpips": [], "linf": [], "psnr_after_transform": []}
    for e in entries:
        orig = Image.open(e.image_path).convert("RGB")
        if args.resolution:
            orig = orig.resize((args.resolution, args.resolution), Image.LANCZOS)
        mask_img = Image.open(e.mask_path).convert("L") if os.path.exists(e.mask_path) else None
        if mask_img is not None:
            mask_img = mask_img.resize(orig.size, Image.NEAREST)
        gate = None
        if args.mask_gate_perturbation and mask_img is not None:
            gate = np.asarray(mask_img) < 128          # subject = mask 0
        prot = immunize_image_pil(net, orig, mask_pil=mask_img if gate is not None else None)
        conds = P.build_conditions(orig, prot, eps_px, rng, transform, gate=gate)
        pert["psnr"].append(P.psnr_u8(orig, prot))
        pert["ssim"].append(float(_ssim(
            np.asarray(orig), np.asarray(prot), channel_axis=2, data_range=255)))
        pert["lpips"].append(divergence(prot, orig))
        pert["linf"].append(P.linf_u8(orig, prot))
        pert["psnr_after_transform"].append(
            P.psnr_u8(conds["clean"], conds["protected"]))
        mask = Image.open(e.mask_path).convert("L") if os.path.exists(e.mask_path) else None
        prepared.append((e, conds, mask, orig.size))
    pert_summary = {k: float(np.mean(v)) for k, v in pert.items()}
    pert_summary["linf_max"] = float(np.max(pert["linf"]))
    pert_summary["image_size"] = float(prepared[0][1]["protected"].size[0]) if prepared else 0.0
    pert_summary["resolution_arg"] = float(args.resolution or 0)
    print("Perturbation:", pert_summary)

    all_records, zoo = [], {m.name.lower(): m for m in EM.DEFAULT_ZOO}
    clip_metric = None
    for name in args.editors:
        if name.upper() == "STUB":
            cfg, pipe = None, None
            edit_fn = lambda img, mask, prompt, seed: img.filter(
                ImageFilter.GaussianBlur(2 + (seed % 2)))
            prompt_key = "prompts"
            clip_score = None
        else:
            cfg = zoo.get(name.lower())
            if cfg is None:
                raise SystemExit(f"Unknown editor {name!r}. Available: {[m.name for m in EM.DEFAULT_ZOO]}")
            if args.editor_resolution:
                import dataclasses
                cfg = dataclasses.replace(cfg, native_resolution=args.editor_resolution)
            print(f"== Loading {cfg.name} (native_resolution={cfg.native_resolution})")
            pipe = EM.create_pipeline(cfg)
            prompt_key = cfg.prompt_key
            if clip_metric is None:
                clip_metric = EM.create_metrics()["clip"]

            def edit_fn(img, mask, prompt, seed, _p=pipe, _c=cfg):
                mm = mask.resize(img.size, Image.NEAREST) if mask is not None else None
                return EM.run_model(_p, _c, img, mm, prompt, seed, img.size[0])

            clip_score = lambda im, pr: float(clip_metric([im], [pr])[0])

        items = []
        for e, conds, mask, size in prepared:
            prompts = (e.flux_prompts if prompt_key == "flux_prompts" else e.sd_prompts)
            items.append({"name": e.name, "mask": mask, "conditions": conds,
                          "restrict_to_mask": bool(cfg and cfg.uses_mask),
                          "prompts": prompts[: args.prompts_per_image]})
        label = cfg.name if cfg else "STUB"
        samples = []
        all_records += P.run_editor(
            label, edit_fn, items, args.seeds, divergence, clip_score,
            samples=samples, max_samples=args.n_samples)
        if samples:
            safe = "".join(ch if ch.isalnum() else "_" for ch in label)
            P.make_sample_grid(samples).save(os.path.join(args.output_dir, f"samples_{safe}.png"))
        if pipe is not None:
            EM.unload_pipeline(pipe)

    summary = P.summarize(all_records, margin=args.margin, n_boot=args.n_boot)
    md = P.to_markdown(summary, pert_summary, args.margin)
    print("\n" + md)
    with open(os.path.join(args.output_dir, "protocol_report.json"), "w") as fh:
        json.dump({
            "args": vars(args), "perturbation": pert_summary,
            "summary": summary,
            "verdicts": {k: P.verdict(v) for k, v in summary.items()},
            "records": [r.__dict__ for r in all_records],
        }, fh, indent=1, default=float)
    with open(os.path.join(args.output_dir, "protocol_report.md"), "w") as fh:
        fh.write(md + "\n")
    print(f"\nWrote {args.output_dir}/protocol_report.{{json,md}}")


if __name__ == "__main__":
    main()
