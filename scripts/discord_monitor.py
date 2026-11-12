#!/usr/bin/env python3
"""Tail a DiffVax training log and post periodic updates to Discord.

Works on an already-running job (it only reads the log), so no restart needed.
Posts: a start message, a progress ping every --every minutes with the latest
per-term losses, step speed, ETA and a loss chart, an alert if the log stalls,
and a final message when the training process exits.

    export DISCORD_WEBHOOK=https://discord.com/api/webhooks/...   # never commit
    python scripts/discord_monitor.py --log /workspace/klein512.log --label klein512 --every 10
"""
import argparse
import io
import json
import os
import re
import sys
import time

import requests

STEP = re.compile(
    r"AVG Loss: ([\d.]+) Loss1: ([\d.]+) Loss2: ([\d.]+)(.*?):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.]+)s/it")
EPOCH = re.compile(r"^Epoch (\d+)\s+avg=([\d.]+)")


def parse(path):
    text = open(path, errors="ignore").read().replace("\r", "\n")
    steps, epochs = [], []
    for line in text.split("\n"):
        m = STEP.search(line)
        if m:
            extra = dict(re.findall(r"(\w+): (-?[\d.]+)", m.group(4)))
            steps.append(dict(loss=float(m[1]), loss1=float(m[2]), loss2=float(m[3]), i=int(m[5]),
                              n=int(m[6]), eta=m[8], spit=float(m[9]), **{k: float(v) for k, v in extra.items()}))
        e = EPOCH.match(line)
        if e:
            epochs.append((int(e[1]), float(e[2])))
    return steps, epochs, text


def chart(steps, epochs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    if steps:
        k = max(1, len(steps) // 400)
        for key in ("loss", "loss1", "HS", "Lat"):
            ys = [s[key] for s in steps if key in s][::k]
            if ys:
                ax[0].plot(ys, label=key, lw=1)
        ax[0].set_title("recent per-step terms"); ax[0].legend(fontsize=7)
    if epochs:
        ax[1].plot([e[0] for e in epochs], [e[1] for e in epochs], "o-")
        ax[1].set_title("epoch average loss")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110); plt.close(fig); buf.seek(0)
    return buf


def post(url, text, img=None):
    try:
        if img is not None:
            requests.post(url, data={"payload_json": json.dumps({"content": text[:1900]})},
                          files={"file": ("loss.png", img, "image/png")}, timeout=20)
        else:
            requests.post(url, json={"content": text[:1900]}, timeout=20)
    except Exception as exc:  # never let a bad webhook kill monitoring
        # Type only: requests errors embed the full URL, which holds the webhook token.
        print("discord post failed:", type(exc).__name__, file=sys.stderr)


def alive(pattern):
    return os.system(f"pgrep -f '{pattern}' >/dev/null") == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--every", type=float, default=10, help="minutes between pings")
    ap.add_argument("--stall", type=float, default=10, help="minutes without log growth = alert")
    ap.add_argument("--proc", default="scripts/train.py", help="pgrep pattern of the training process")
    ap.add_argument("--price", type=float, default=1.79, help="$/hour for the cost line")
    args = ap.parse_args()
    url = os.environ.get("DISCORD_WEBHOOK") or os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        sys.exit("Set DISCORD_WEBHOOK")
    t0 = time.time()
    post(url, f"**{args.label}** monitoring started.")
    last_size, last_growth, last_post, stalled = -1, time.time(), 0.0, False
    while True:
        size = os.path.getsize(args.log) if os.path.exists(args.log) else 0
        if size != last_size:
            last_size, last_growth, stalled = size, time.time(), False
        elif not stalled and time.time() - last_growth > args.stall * 60 and alive(args.proc):
            stalled = True
            post(url, f"@here **{args.label}**: no log output for {args.stall:.0f} min while the process is alive. Possible stall.")
        if not alive(args.proc):
            steps, epochs, text = parse(args.log) if os.path.exists(args.log) else ([], [], "")
            tail = "\n".join(text.strip().split("\n")[-4:])[-900:]
            post(url, f"@here **{args.label}** training process exited. Epochs done: {len(epochs)}.\n```{tail}```", chart(steps, epochs))
            return
        if time.time() - last_post >= args.every * 60:
            steps, epochs, _ = parse(args.log)
            if steps:
                s = steps[-1]
                terms = ", ".join(f"{k}={s[k]:.3f}" for k in ("loss", "loss1", "loss2", "Spec", "Lat", "HS") if k in s)
                hrs = (time.time() - t0) / 3600
                msg = (f"**{args.label}** epoch {len(epochs)} step {s['i']}/{s['n']} | {s['spit']:.2f} s/it | "
                       f"epoch ETA {s['eta']}\n{terms}\n"
                       f"last epoch avgs: {', '.join(f'{e[1]:.3f}' for e in epochs[-4:]) or 'n/a'} | "
                       f"monitored {hrs:.1f} h = ${hrs * args.price:.2f}")
                post(url, msg, chart(steps, epochs))
            last_post = time.time()
        time.sleep(30)


if __name__ == "__main__":
    main()
