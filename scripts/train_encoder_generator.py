#!/usr/bin/env python3
"""Train the one-shot perturbation generator on the VAE-ENCODER objective.

Why: per-image PGD showed that pushing the editor's VAE latent of the protected
image toward a flat-gray latent makes FLUX.2 Klein edits stop depicting the
original person, and that it survives JPEG when optimised through a JPEG/resize
EoT. This trains the same objective AMORTIZED, so one forward pass of the small
NestedUNet (mobile-sized) replaces per-image optimisation. No denoising loop is
needed in training (encoder only), so it is ~10-50x cheaper per step than the
edit-mode trainer and scales to 1088px.

    python scripts/train_encoder_generator.py --resolution 512 --steps 3000 --epsilon 16 \
        --out /workspace/outputs/enc_gen_512

Loss = enc_weight * ||E(T(x+d)) - E(gray)||^2 / ||E(x) - E(gray)||^2
     + lpips_weight * LPIPS(x, x+d) + alpha * |d|_1,   ||d||_inf <= epsilon
where E is each configured VAE encoder (mean), T is the EoT (JPEG/resize).
Checkpoints are named *_epoch<k>.pth / *_final.pth for scripts/eval_watcher.py.
"""
import argparse
import glob
import os
import random
import sys
import time

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffvax.eot import DifferentiableEoT
from diffvax.model import NestedUNet
from diffvax.reporter import TrainingReporter
from diffvax.utils import resolve_device, resolve_dtype


class Images(torch.utils.data.Dataset):
    """Images at res; with teacher_dir, also the per-image teacher perturbation
    (make_teacher.py <split>/deltas/<name>.pt, same resolution), flipped together."""

    def __init__(self, files, res, teacher_dir=None):
        self.res, self.teacher_dir = res, teacher_dir
        if teacher_dir:
            files = [f for f in files if os.path.exists(self._tpath(f))]
        self.files = files

    def _tpath(self, f):
        return os.path.join(self.teacher_dir, os.path.splitext(os.path.basename(f))[0] + ".pt")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB").resize((self.res, self.res), Image.LANCZOS)
        a = torch.from_numpy(np.asarray(img, np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        t = torch.load(self._tpath(self.files[i]))[0].float() if self.teacher_dir else torch.zeros(0)
        if random.random() < 0.5:
            a = a.flip(-1)
            t = t.flip(-1) if t.numel() else t
        return a, t


def load_vaes(specs, device, dtype):
    """specs: 'repo[:subfolder]' strings. Uses AutoencoderKLFlux2 for FLUX.2
    repos, AutoencoderKLQwenImage for Qwen, else AutoencoderKL."""
    from diffusers import AutoencoderKL, AutoencoderKLFlux2
    vaes = []
    for spec in specs:
        repo, _, sub = spec.partition(":")
        sub = sub or "vae"
        if "FLUX.2" in repo:
            cls = AutoencoderKLFlux2
        elif "Qwen" in repo:
            from diffusers import AutoencoderKLQwenImage as cls
        else:
            cls = AutoencoderKL
        v = cls.from_pretrained(repo, subfolder=sub, torch_dtype=dtype).to(device).eval()
        v.requires_grad_(False)
        vaes.append((spec, v))
    return vaes


def encode(vae, x, dtype):
    out = vae.encode(x.to(dtype))
    z = out.latent_dist.mode() if hasattr(out, "latent_dist") else out[0]
    return z.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(_root, "data"))
    ap.add_argument("--vae", nargs="+", default=["black-forest-labs/FLUX.2-klein-4B:vae"])
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epsilon", type=float, default=16)
    ap.add_argument("--enc-weight", type=float, default=1.0)
    ap.add_argument("--lpips-weight", type=float, default=1.0)
    ap.add_argument("--lpips-target", type=float, default=None,
                    help="hinge: penalise LPIPS only ABOVE this value (e.g. 0.05). A plain LPIPS "
                         "penalty drove the perturbation to ~0 (LPIPS 0.001, no effect) in run 1")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--eot", choices=["on", "off"], default="on")
    ap.add_argument("--nb-filter", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    ap.add_argument("--teacher-dir", default=None,
                    help="make_teacher.py deltas dir (e.g. <out>/train/deltas) to distil from")
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument("--init", default=None, help="warm-start .pth")
    ap.add_argument("--bound", choices=["tanh", "ste"], default="ste",
                    help="epsilon bound: straight-through hard clip (default) or tanh")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--label", default="enc-gen")
    ap.add_argument("--price", type=float, default=1.79)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = resolve_device(); dtype = resolve_dtype(device)
    torch.manual_seed(0); random.seed(0)
    os.makedirs(os.path.join(args.out, "models"), exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data_dir, "train", "images", "*.png")))
    ds = Images(files, args.resolution, args.teacher_dir)
    print(f"{len(ds)} training images" + (f" with teacher perturbations" if args.teacher_dir else ""))
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch,
                                     shuffle=True, drop_last=True, num_workers=4)
    net = NestedUNet(nb_filter=args.nb_filter, epsilon=args.epsilon, bound=args.bound).to(device)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=device, weights_only=True))
    else:
        net.init_small_output()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    vaes = load_vaes(args.vae, device, dtype)
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)
    eot = DifferentiableEoT({"eot": {"enabled": True, "jpeg_quality_range": [70, 90],
                                     "resize_range": [0.75, 1.0], "p_jpeg": 1.0, "p_resize": 0.5,
                                     "p_blur": 0.0, "p_noise": 0.0}}) if args.eot == "on" else None

    rep = TrainingReporter({"project_name": args.label, "reporting": {
        "progress_every_minutes": 10, "price_per_hour": args.price, "label": args.label,
        "trackio": {"enabled": True, "project": "diffvax-enc-gen", "log_every_n_steps": 10}}}, args.out)
    rep.start_run({"objective": "VAE encoder -> gray latent", "vaes": ", ".join(args.vae),
                   "resolution": args.resolution, "batch": args.batch, "steps": args.steps,
                   "epsilon": f"{args.epsilon}/255", "lpips_weight": args.lpips_weight,
                   "lpips_target": args.lpips_target, "lr": args.lr,
                   "alpha": args.alpha, "eot": args.eot,
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)})

    gray_cache = {}
    step, save_k, t_prev = 0, 0, time.perf_counter()
    stem = os.path.join(args.out, "models", f"encgen_eps{int(args.epsilon)}_{args.resolution}")
    while step < args.steps:
        for x, d_t in dl:
            x = x.to(device)
            d = net(x)
            x_adv = (x + d).clamp(-1, 1)
            inp = eot(x_adv) if eot is not None else x_adv
            enc_terms = []
            for spec, vae in vaes:
                key = (spec, tuple(x.shape))
                if key not in gray_cache:
                    with torch.no_grad():
                        gray_cache[key] = encode(vae, torch.zeros_like(x), dtype)
                zg = gray_cache[key]
                with torch.no_grad():
                    base = F.mse_loss(encode(vae, x, dtype), zg)
                enc_terms.append(F.mse_loss(encode(vae, inp, dtype), zg) / base.clamp(min=1e-8))
            l_enc = torch.stack(enc_terms).mean()
            l_lp_raw = lp(x, x_adv).flatten()
            l_lp = (F.relu(l_lp_raw - args.lpips_target) if args.lpips_target is not None
                    else l_lp_raw).mean()
            l_l1 = d.abs().mean()
            l_dist = torch.zeros((), device=device)
            if args.teacher_dir:
                eps_t = 2.0 * args.epsilon / 255.0
                l_dist = F.mse_loss(d, d_t.to(device)) / eps_t ** 2
            loss = (args.enc_weight * l_enc + args.lpips_weight * l_lp + args.alpha * l_l1
                    + args.distill_weight * l_dist)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            step += 1
            now = time.perf_counter(); st = now - t_prev; t_prev = now
            terms = {"loss": loss.item(), "enc": l_enc.item(), "lpips": l_lp_raw.mean().item(),
                     "l1": l_l1.item(), "distill": l_dist.item()}
            if step % 20 == 0:
                print(f"step {step}/{args.steps} " + " ".join(f"{k}={v:.4f}" for k, v in terms.items())
                      + f" {st:.2f}s/it", flush=True)
            rep.log_step(step, terms, epoch=step // max(1, len(dl)), batch=step, n_batches=args.steps,
                         step_seconds=st)
            if step % args.save_every == 0 or step == args.steps:
                path = f"{stem}_final.pth" if step == args.steps else f"{stem}_epoch{save_k}.pth"
                torch.save(net.state_dict(), path)
                save_k += 1
                rep.report_checkpoint(step, loss.item(), path, is_best=False)
            if step >= args.steps:
                break
    rep.report_complete(step, loss.item(), f"{stem}_final.pth")


if __name__ == "__main__":
    main()
