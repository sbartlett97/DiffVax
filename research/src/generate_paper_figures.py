#!/usr/bin/env python3
"""Generate publication-ready figures for the DiffVax++ ICLR paper.

Produces:
  fig1_overview.pdf         — 3-panel overview (H2 bar, H1 heatmap, JPEG paradox)
  fig2_jpeg_paradox.pdf     — JPEG paradox detail: EDR vs compression per checkpoint/model
  fig3_patch_accumulation.pdf — Perturbation accumulation diagram (conceptual + H2 data)

Usage:
    python research/src/generate_paper_figures.py --out paper/latex/figures/
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
H1_CSV = PROJECT_ROOT / "research/experiments/H1-multimodel-transfer/results/transfer_edr_metrics.csv"
H2_CSV = PROJECT_ROOT / "research/experiments/H2-patch-inference/results/patch_edr_metrics.csv"

def setup():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.gridspec as gridspec
        import numpy as np
        import pandas as pd
        return plt, np, pd, mpatches, gridspec
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)


def fig1_overview(out_dir: Path):
    """3-panel overview figure — the paper's Figure 1."""
    plt, np, pd, mpatches, gridspec = setup()

    df1 = pd.read_csv(H1_CSV)
    df2 = pd.read_csv(H2_CSV)

    fig = plt.figure(figsize=(14, 4.5))
    fig.suptitle(
        r"\textbf{DiffVax}$^{++}$: One Confirmed Contribution and Three Surprising Discoveries",
        fontsize=13, fontweight="bold", y=1.01
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ─── Panel A: H2 Patch Inference ──────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0])
    order = ["baseline_512", "no_overlap", "25pct_overlap", "50pct_overlap"]
    labels = ["512px\nbaseline", "0% overlap\n(stride 512)", "25% overlap\n(stride 384)", "50% overlap\n(stride 256)"]
    colors_a = ["#c0392b", "#95a5a6", "#7f8c8d", "#27ae60"]
    edrs_a = []
    for c in order:
        sub = df2[df2["condition"] == c]
        edrs_a.append(sub["disrupted"].mean() if len(sub) else 0)

    bars = ax_a.bar(range(len(order)), edrs_a, color=colors_a, edgecolor="white",
                    linewidth=1.5, width=0.6)
    ax_a.set_xticks(range(len(order)))
    ax_a.set_xticklabels(labels, fontsize=8)
    ax_a.set_ylabel("Edit Disruption Rate (EDR)", fontsize=9)
    ax_a.set_ylim(0, 0.6)
    ax_a.set_title("(a) Patch-Based High-Res Inference\nMore overlap $\\rightarrow$ stronger (×1.60 baseline)",
                   fontsize=9, fontweight="bold")

    baseline_edr = edrs_a[0] if edrs_a[0] > 0 else 1
    for i, (bar, val) in enumerate(zip(bars, edrs_a)):
        label = f"{val:.3f}"
        if order[i] == "50pct_overlap" and baseline_edr:
            label = f"{val:.3f}\n(×{val/baseline_edr:.2f}↑)"
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                  label, ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ─── Panel B: H1 Cross-Architecture Transfer Heatmap ──────────────────────
    ax_b = fig.add_subplot(gs[1])
    df1_clean = df1[df1["jpeg_quality"].astype(str) == "none"]
    ckpt_order = ["sd15_only", "multimodel_h1a"]
    ckpt_labels = ["SD1.5 only\n(published)", "Multi-model\n(H1a)"]
    model_order = ["sd15", "flux_schnell", "sd35"]
    model_labels = ["SD 1.5", "FLUX.1-schnell", "SD 3.5"]

    ckpts_present = [c for c in ckpt_order if c in df1_clean["checkpoint"].unique()]
    models_present = [m for m in model_order if m in df1_clean["eval_model"].unique()]

    matrix = np.zeros((len(ckpts_present), len(models_present)))
    for i, ckpt in enumerate(ckpts_present):
        for j, mdl in enumerate(models_present):
            sub = df1_clean[(df1_clean["checkpoint"] == ckpt) & (df1_clean["eval_model"] == mdl)]
            if len(sub):
                matrix[i, j] = sub["disrupted"].mean()

    im = ax_b.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=0.5, aspect="auto")
    ax_b.set_xticks(range(len(models_present)))
    ax_b.set_xticklabels([model_labels[model_order.index(m)] for m in models_present],
                          fontsize=8.5, rotation=15, ha="right")
    ax_b.set_yticks(range(len(ckpts_present)))
    ax_b.set_yticklabels([ckpt_labels[ckpt_order.index(c)] for c in ckpts_present], fontsize=9)
    ax_b.set_xlabel("Evaluation Model", fontsize=9)
    ax_b.set_title("(b) Cross-Architecture Transfer\nSD1.5-only transfers to FLUX & SD3.5 (no retraining)",
                   fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04, label="EDR")

    for i in range(len(ckpts_present)):
        for j in range(len(models_present)):
            val = matrix[i, j]
            tc = "white" if val < 0.12 or val > 0.38 else "black"
            ax_b.text(j, i, f"{val:.2f}", ha="center", va="center",
                      fontsize=11, fontweight="bold", color=tc)

    # ─── Panel C: JPEG Paradox ────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[2])
    # sd15_only on flux_schnell: clean=0.200, q75=0.300, q70=0.260
    # multimodel_h1a on flux_schnell: clean=0.140, q75=0.150, q70=0.150
    # h7_jpeg on flux_schnell: clean=0.090, q75=0.080, q70=0.090
    # Use actual CSV data if h7 is there, otherwise hard-coded from findings.md
    jpeg_modes = ["none", "75", "70"]
    jpeg_labels_c = ["Clean", "JPEG q=75\n(Instagram)", "JPEG q=70\n(Twitter/X)"]
    ckpts_c = [c for c in ["sd15_only", "multimodel_h1a", "h7_jpeg_robust"] if c in df1["checkpoint"].unique()]
    # Also try h7_jpeg
    if "h7_jpeg" in df1["checkpoint"].unique():
        ckpts_c = [c for c in ["sd15_only", "multimodel_h1a", "h7_jpeg"] if c in df1["checkpoint"].unique()]

    palette_c = {
        "sd15_only": "#2980b9",
        "multimodel_h1a": "#e67e22",
        "h7_jpeg_robust": "#27ae60",
        "h7_jpeg": "#27ae60",
    }
    label_c = {
        "sd15_only": "SD1.5 only (baseline)",
        "multimodel_h1a": "Multi-model H1a",
        "h7_jpeg_robust": "JPEG-augmented H7",
        "h7_jpeg": "JPEG-augmented H7",
    }

    x_c = np.arange(len(jpeg_modes))
    bw = 0.22
    df1_flux = df1[df1["eval_model"] == "flux_schnell"] if "flux_schnell" in df1["eval_model"].unique() else df1

    any_plotted = False
    for i, ckpt in enumerate(ckpts_c):
        edrs_c = []
        for jq in jpeg_modes:
            sub = df1_flux[(df1_flux["checkpoint"] == ckpt) &
                           (df1_flux["jpeg_quality"].astype(str) == str(jq))]
            edrs_c.append(sub["disrupted"].mean() if len(sub) else 0)
        offset = (i - len(ckpts_c) / 2 + 0.5) * bw
        color = palette_c.get(ckpt, "#9b59b6")
        bars_c = ax_c.bar(x_c + offset, edrs_c, bw * 0.9,
                          label=label_c.get(ckpt, ckpt), color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars_c, edrs_c):
            if val > 0.02:
                ax_c.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                          f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        any_plotted = True

    if not any_plotted:
        # Fallback: hard-coded values from findings.md
        data = {
            "SD1.5 only": [0.200, 0.300, 0.260],
            "Multi-model": [0.140, 0.150, 0.150],
            "JPEG-aug H7": [0.090, 0.080, 0.090],
        }
        for i, (name, edrs_c) in enumerate(data.items()):
            offset = (i - 1.0) * bw
            color = ["#2980b9", "#e67e22", "#27ae60"][i]
            bars_c = ax_c.bar(x_c + offset, edrs_c, bw * 0.9,
                              label=name, color=color, alpha=0.85, edgecolor="white")
            for bar, val in zip(bars_c, edrs_c):
                ax_c.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                          f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Annotate paradox arrow
    ax_c.annotate("", xy=(1.0, 0.305), xytext=(0.0, 0.205),
                  arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2))
    ax_c.text(0.4, 0.315, "+50%\nparadox", ha="center", fontsize=7.5,
              color="#c0392b", fontweight="bold")

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(jpeg_labels_c, fontsize=8.5)
    ax_c.set_ylabel("FLUX EDR", fontsize=9)
    ax_c.set_ylim(0, 0.45)
    ax_c.set_title("(c) JPEG Paradox on FLUX.1-schnell\nJPEG q=75 increases sd15\\_only EDR by 50%",
                   fontsize=9, fontweight="bold")
    ax_c.legend(fontsize=7.5, loc="upper right")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig1_overview.pdf"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", format="pdf")
    plt.savefig(str(out_path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def fig2_jpeg_paradox(out_dir: Path):
    """Detailed JPEG paradox figure — all 3 models × 3 checkpoints × 3 compression levels."""
    plt, np, pd, mpatches, gridspec = setup()

    df1 = pd.read_csv(H1_CSV)
    eval_models = [m for m in ["sd15", "flux_schnell", "sd35"] if m in df1["eval_model"].unique()]
    model_titles = {"sd15": "SD 1.5 (UNet, 4-ch VAE)", "flux_schnell": "FLUX.1-schnell (MM-DiT, 16-ch VAE)",
                    "sd35": "SD 3.5 (MM-DiT, 16-ch VAE)"}

    n = len(eval_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle("JPEG Paradox: Compression Effect on Edit Disruption Rate by Evaluation Model",
                 fontsize=12, fontweight="bold")

    jpeg_modes = ["none", "75", "70"]
    jpeg_labels = ["Clean", "q=75\n(Instagram)", "q=70\n(Twitter/X)"]
    ckpts_all = df1["checkpoint"].unique().tolist()
    ckpt_order = ["sd15_only", "multimodel_h1a", "h7_jpeg_robust", "h7_jpeg"]
    ckpts = [c for c in ckpt_order if c in ckpts_all] or ckpts_all
    palette = {"sd15_only": "#2980b9", "multimodel_h1a": "#e67e22",
               "h7_jpeg_robust": "#27ae60", "h7_jpeg": "#27ae60"}
    short_names = {"sd15_only": "SD1.5 only", "multimodel_h1a": "Multi-model",
                   "h7_jpeg_robust": "JPEG-aug", "h7_jpeg": "JPEG-aug"}

    bw = 0.22
    x = np.arange(len(jpeg_modes))

    for ax, mdl in zip(axes, eval_models):
        df_m = df1[df1["eval_model"] == mdl]
        for i, ckpt in enumerate(ckpts):
            edrs = []
            for jq in jpeg_modes:
                sub = df_m[(df_m["checkpoint"] == ckpt) &
                           (df_m["jpeg_quality"].astype(str) == str(jq))]
                edrs.append(sub["disrupted"].mean() if len(sub) else 0)
            offset = (i - len(ckpts) / 2 + 0.5) * bw
            color = palette.get(ckpt, "#9b59b6")
            bars = ax.bar(x + offset, edrs, bw * 0.9,
                          label=short_names.get(ckpt, ckpt), color=color, alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, edrs):
                if val > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(jpeg_labels, fontsize=9)
        ax.set_title(model_titles.get(mdl, mdl), fontsize=9, fontweight="bold")
        ax.set_ylabel("Edit Disruption Rate (EDR)", fontsize=9)
        ax.set_ylim(0, 0.5)
        ax.legend(fontsize=7.5, loc="upper left")

        # Annotate JPEG paradox for FLUX only
        if mdl == "flux_schnell":
            ax.annotate("JPEG paradox\n(+50%)", xy=(1.0, 0.305), xytext=(0.6, 0.38),
                        fontsize=8, color="#c0392b", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5))

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig2_jpeg_paradox.pdf"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", format="pdf")
    plt.savefig(str(out_path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def fig3_patch_accumulation(out_dir: Path):
    """Conceptual patch accumulation figure with H2 data."""
    plt, np, pd, mpatches, gridspec = setup()
    df2 = pd.read_csv(H2_CSV)

    fig = plt.figure(figsize=(12, 4))
    fig.suptitle("Perturbation Accumulation: Why Overlapping 512px Patches Immunize 1088px Images Better",
                 fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35, width_ratios=[1.2, 1])

    # ─── Panel A: conceptual coverage heatmap ─────────────────────────────────
    ax_a = fig.add_subplot(gs[0])
    # Simulate overlap count for 1088px with stride=256, patch=512
    img_size = 1088
    patch = 512
    stride = 256
    starts = list(range(0, img_size - patch + 1, stride))
    count_map = np.zeros((img_size, img_size), dtype=float)
    for sy in starts:
        for sx in starts:
            count_map[sy:sy + patch, sx:sx + patch] += 1.0

    im = ax_a.imshow(count_map, cmap="YlOrRd", interpolation="nearest", aspect="equal")
    ax_a.set_title("Patch overlap count at 1088px, stride=256\n"
                   "Center: 4× perturbation accumulation",
                   fontsize=9)
    ax_a.set_xlabel("x (pixels)", fontsize=8)
    ax_a.set_ylabel("y (pixels)", fontsize=8)
    ax_a.tick_params(labelsize=7)
    cbar = plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
    cbar.set_label("# overlapping patches", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Mark center
    cx = img_size // 2
    ax_a.plot(cx, cx, "w*", markersize=10)
    ax_a.text(cx + 20, cx - 40, "4×", color="white", fontsize=12, fontweight="bold")

    # ─── Panel B: EDR vs overlap strategy ─────────────────────────────────────
    ax_b = fig.add_subplot(gs[1])
    order = ["baseline_512", "no_overlap", "25pct_overlap", "50pct_overlap"]
    labels_b = ["512px\nbaseline", "0%\noverlap", "25%\noverlap", "50%\noverlap\n(×1.60↑)"]
    colors_b = ["#c0392b", "#95a5a6", "#7f8c8d", "#27ae60"]
    edrs_b = []
    for c in order:
        sub = df2[df2["condition"] == c]
        edrs_b.append(sub["disrupted"].mean() if len(sub) else 0)
    psnr_b = []
    for c in order:
        sub = df2[df2["condition"] == c]
        psnr_b.append(sub["psnr_immunized"].mean() if len(sub) else 0)

    x_b = np.arange(len(order))
    bars_b = ax_b.bar(x_b - 0.2, edrs_b, 0.35, label="EDR (↑ better)", color=colors_b,
                      edgecolor="white", linewidth=1.2, alpha=0.9)
    ax_b.set_ylabel("Edit Disruption Rate (EDR)", fontsize=9, color="#2c3e50")
    ax_b.set_ylim(0, 0.6)

    ax_b2 = ax_b.twinx()
    line = ax_b2.plot(x_b + 0.2, psnr_b, "D--", color="#8e44ad", linewidth=1.5,
                      markersize=7, label="PSNR (↑ more imperceptible)")
    ax_b2.set_ylabel("PSNR (dB)", fontsize=9, color="#8e44ad")
    ax_b2.set_ylim(25, 38)
    ax_b2.axhline(28, color="#8e44ad", linestyle=":", alpha=0.5)
    ax_b2.tick_params(axis="y", labelcolor="#8e44ad")

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(labels_b, fontsize=8)
    ax_b.set_title("EDR (effectiveness) vs PSNR (imperceptibility)\nMore overlap = stronger AND perceptually acceptable",
                   fontsize=9)

    for bar, val in zip(bars_b, edrs_b):
        ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                  f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Combine legends
    handles1, labels1 = ax_b.get_legend_handles_labels()
    handles2, labels2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(handles1 + handles2, labels1 + labels2, fontsize=7.5, loc="upper left")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig3_patch_accumulation.pdf"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", format="pdf")
    plt.savefig(str(out_path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="paper/latex/figures/")
    parser.add_argument("--figures", nargs="+", choices=["fig1", "fig2", "fig3", "all"],
                        default=["all"])
    args = parser.parse_args()
    out_dir = PROJECT_ROOT / args.out

    targets = args.figures
    if "all" in targets or "fig1" in targets:
        fig1_overview(out_dir)
    if "all" in targets or "fig2" in targets:
        fig2_jpeg_paradox(out_dir)
    if "all" in targets or "fig3" in targets:
        fig3_patch_accumulation(out_dir)
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
