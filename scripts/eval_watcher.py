#!/usr/bin/env python3
"""Evaluate each new training checkpoint with eval_protocol.py and post to Discord.

    export DISCORD_WEBHOOK=...
    python scripts/eval_watcher.py --models-dir /workspace/outputs/klein512/models/1 \
        --epsilon 16 --editors "FLUX.2 Klein" --resolution 512 --editor-resolution 512

Watches for *_epoch<N>.pth and *_final.pth, evaluates each once (oldest first),
appends a line to <out>/summary.tsv, and posts the headline numbers. Exits after
the final checkpoint has been evaluated.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import requests

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def post(url, text, image_path=None):
    try:
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as fh:
                requests.post(url, data={"payload_json": json.dumps({"content": text[:1900]})},
                              files={"file": (os.path.basename(image_path), fh, "image/png")}, timeout=60)
        else:
            requests.post(url, json={"content": text[:1900]}, timeout=15)
    except Exception as exc:  # type only: the message can contain the webhook URL
        print("discord post failed:", type(exc).__name__, file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--out", default="/workspace/outputs/ckpt_evals")
    ap.add_argument("--epsilon", default="16")
    ap.add_argument("--nb-filter", nargs="+", default=["32", "64", "128", "256", "512"])
    ap.add_argument("--editors", nargs="+", default=["FLUX.2 Klein"])
    ap.add_argument("--resolution", default="512")
    ap.add_argument("--editor-resolution", default="512")
    ap.add_argument("--max-images", default="10")
    ap.add_argument("--extra", nargs="*", default=[], help="extra args for eval_protocol.py")
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()
    url = os.environ.get("DISCORD_WEBHOOK")
    os.makedirs(args.out, exist_ok=True)
    done = set()
    while True:
        cks = sorted(glob.glob(os.path.join(args.models_dir, "*_epoch*.pth")),
                     key=lambda p: int(re.search(r"_epoch(\d+)\.pth$", p).group(1)))
        finals = glob.glob(os.path.join(args.models_dir, "*_final.pth"))
        todo = [c for c in cks + finals if c not in done]
        for ck in todo:
            time.sleep(5)  # let the file finish writing
            tag = os.path.basename(ck).split("_")[-1].replace(".pth", "")
            out = os.path.join(args.out, tag)
            cmd = [sys.executable, os.path.join(_root, "scripts", "eval_protocol.py"),
                   "--checkpoint", ck, "--epsilon", args.epsilon, "--nb-filter", *args.nb_filter,
                   "--editors", *args.editors, "--resolution", args.resolution,
                   "--editor-resolution", args.editor_resolution,
                   "--max-images", args.max_images, "--seeds", "0", "--output-dir", out, *args.extra]
            r = subprocess.run(cmd, capture_output=True, text=True)
            done.add(ck)
            rp = os.path.join(out, "protocol_report.json")
            if r.returncode != 0 or not os.path.exists(rp):
                if url:
                    post(url, f"**eval {tag}** failed:\n```{r.stderr[-700:]}```")
                continue
            rep = json.load(open(rp))
            p = rep["perturbation"]
            for ed, s in rep["summary"].items():
                line = (f"**eval {tag}** {ed}: d(prot)={s['d_protected'][0]:+.3f} "
                        f"noise={s.get('d_noise_psnr', [float('nan')])[0]:+.3f} "
                        f"excess={s['excess_noise_psnr'][0]:+.3f} "
                        f"[{s['excess_noise_psnr'][1]:+.3f},{s['excess_noise_psnr'][2]:+.3f}] "
                        f"=> **{rep['verdicts'][ed]}** | SSIM={p['ssim']:.3f} LPIPS={p['lpips']:.3f} "
                        f"PSNR={p['psnr']:.1f} Linf={p['linf_max']:.0f}")
                print(line, flush=True)
                with open(os.path.join(args.out, "summary.tsv"), "a") as fh:
                    fh.write(f"{tag}\t{ed}\t{s['d_protected'][0]:.4f}\t{s['excess_noise_psnr'][0]:.4f}\t"
                             f"{rep['verdicts'][ed]}\t{p['ssim']:.4f}\t{p['lpips']:.4f}\t{p['psnr']:.2f}\n")
                if url:
                    safe = "".join(ch if ch.isalnum() else "_" for ch in ed)
                    post(url, line, os.path.join(out, f"samples_{safe}.png"))
        if finals and all(f in done for f in finals):
            if url:
                post(url, "@here **final checkpoint evaluated** - training and evals are finished.")
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
