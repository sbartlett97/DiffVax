#!/usr/bin/env python3
"""Generate publication-ready plots from DiffVax experiment CSVs.

Usage:
    # H2: patch inference comparison
    python plot_results.py h2 \
        --csv research/experiments/H2-patch-inference/results/patch_edr_metrics.csv \
        --out research/to_human/figures/

    # H1: cross-model transfer heatmap
    python plot_results.py h1 \
        --csv research/experiments/H1-multimodel-transfer/results/transfer_edr_metrics.csv \
        --out research/to_human/figures/

    # H7: JPEG robustness bar chart
    python plot_results.py h7 \
        --csv research/experiments/H1-multimodel-transfer/results/transfer_edr_metrics.csv \
        --out research/to_human/figures/

    # All: generate all plots from all available CSVs
    python plot_results.py all --out research/to_human/figures/
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        return plt, np, mpatches
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)


def _require_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        print("pandas not installed. Run: pip install pandas")
        sys.exit(1)


def plot_h2(csv_path: Path, out_dir: Path):
    """H2: EDR and PSNR comparison across patch overlap conditions."""
    plt, np, mpatches = _require_matplotlib()
    pd = _require_pandas()

    df = pd.read_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Order conditions meaningfully
    cond_order = ["baseline_512", "no_overlap", "25pct_overlap", "50pct_overlap"]
    cond_labels = {
        "baseline_512": "Baseline\n(512px)",
        "no_overlap": "No overlap\n(stride=512)",
        "25pct_overlap": "25% overlap\n(stride=384)",
        "50pct_overlap": "50% overlap\n(stride=256)\n★ recommended",
    }

    present = [c for c in cond_order if c in df["condition"].unique()]
    edr_vals = [df[df["condition"] == c]["disrupted"].mean() for c in present]
    psnr_vals = [df[df["condition"] == c]["psnr_immunized"].mean() for c in present]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("H2: Patch-Based 1088px Immunization — EDR vs Overlap Strategy",
                 fontsize=14, fontweight="bold")

    colors = ["#4CAF50" if v >= 0.8 else "#FF9800" if v >= 0.6 else "#F44336"
              for v in edr_vals]

    # EDR
    bars = axes[0].bar([cond_labels.get(c, c) for c in present], edr_vals,
                       color=colors, edgecolor="white", linewidth=1.5)
    axes[0].axhline(0.8, color="green", linestyle="--", alpha=0.7, label="Target EDR (0.8)")
    axes[0].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random baseline (0.5)")
    axes[0].set_ylabel("Edit Disruption Rate (EDR)", fontsize=11)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=9)
    axes[0].set_title("Edit Disruption Rate", fontsize=12)
    for bar, val in zip(bars, edr_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", va="bottom", fontweight="bold")

    # PSNR
    bars2 = axes[1].bar([cond_labels.get(c, c) for c in present], psnr_vals,
                        color=colors, edgecolor="white", linewidth=1.5)
    axes[1].axhline(30, color="green", linestyle="--", alpha=0.7, label="Imperceptible threshold (30 dB)")
    axes[1].set_ylabel("PSNR (dB) — higher is more imperceptible", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].set_title("Imperceptibility (PSNR)", fontsize=12)
    for bar, val in zip(bars2, psnr_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    out_path = out_dir / "h2_patch_inference.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_h1(csv_path: Path, out_dir: Path):
    """H1: Cross-model transfer heatmap (checkpoint × eval model)."""
    plt, np, mpatches = _require_matplotlib()
    pd = _require_pandas()

    df = pd.read_csv(csv_path)
    # Use only clean (no JPEG) results
    if "jpeg_quality" in df.columns:
        df = df[df["jpeg_quality"] == "none"]

    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = df["checkpoint"].unique().tolist()
    eval_models = df["eval_model"].unique().tolist()

    edr_matrix = np.zeros((len(checkpoints), len(eval_models)))
    for i, ckpt in enumerate(checkpoints):
        for j, mdl in enumerate(eval_models):
            sub = df[(df["checkpoint"] == ckpt) & (df["eval_model"] == mdl)]
            if len(sub):
                edr_matrix[i, j] = sub["disrupted"].mean()

    fig, ax = plt.subplots(figsize=(max(7, len(eval_models) * 2), max(5, len(checkpoints) * 1.2)))
    im = ax.imshow(edr_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(eval_models)))
    ax.set_xticklabels(eval_models, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(len(checkpoints)))
    ax.set_yticklabels(checkpoints, fontsize=11)
    ax.set_xlabel("Evaluation Model (unseen during training)", fontsize=12)
    ax.set_ylabel("Training Checkpoint", fontsize=12)
    ax.set_title("H1: Edit Disruption Rate — Cross-Model Transfer\n"
                 "(Green = high EDR = better immunization)", fontsize=13, fontweight="bold")

    for i in range(len(checkpoints)):
        for j in range(len(eval_models)):
            val = edr_matrix[i, j]
            color = "white" if val < 0.4 or val > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="Edit Disruption Rate")
    plt.tight_layout()
    out_path = out_dir / "h1_transfer_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_h7(csv_path: Path, out_dir: Path):
    """H7: JPEG robustness — EDR before vs after JPEG compression."""
    plt, np, mpatches = _require_matplotlib()
    pd = _require_pandas()

    df = pd.read_csv(csv_path)
    if "jpeg_quality" not in df.columns:
        print("No jpeg_quality column. Re-run eval with --jpeg-qualities 75 70")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = df["checkpoint"].unique().tolist()
    jpeg_modes = sorted(df["jpeg_quality"].unique().tolist(), key=lambda x: (x == "none", x))
    jpeg_labels = {q: ("No compression\n(baseline)" if q == "none" else f"JPEG q={q}")
                   for q in jpeg_modes}

    eval_models = df["eval_model"].unique().tolist()

    # One subplot per eval model
    n_models = len(eval_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6), sharey=True)
    if n_models == 1:
        axes = [axes]

    fig.suptitle("H7: JPEG Robustness — Does Immunization Survive Social Media Upload Compression?",
                 fontsize=13, fontweight="bold")

    bar_width = 0.8 / len(checkpoints)
    x = np.arange(len(jpeg_modes))
    palette = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    for ax, eval_model in zip(axes, eval_models):
        for i, (ckpt, color) in enumerate(zip(checkpoints, palette)):
            edrs = []
            for jq in jpeg_modes:
                sub = df[(df["checkpoint"] == ckpt) &
                         (df["eval_model"] == eval_model) &
                         (df["jpeg_quality"] == jq)]
                edrs.append(sub["disrupted"].mean() if len(sub) else 0.0)

            offset = (i - len(checkpoints) / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, edrs, bar_width * 0.9,
                          label=ckpt, color=color, alpha=0.85)
            for bar, val in zip(bars, edrs):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([jpeg_labels[q] for q in jpeg_modes], fontsize=10)
        ax.set_xlabel("Compression Applied to Immunized Image", fontsize=10)
        ax.set_title(f"Eval: {eval_model}", fontsize=11)
        ax.axhline(0.7, color="green", linestyle="--", alpha=0.6, label="Target (0.7)")
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.4, label="Chance (0.5)")
        ax.set_ylim(0, 1.1)
        if ax == axes[0]:
            ax.set_ylabel("Edit Disruption Rate (EDR)", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path = out_dir / "h7_jpeg_robustness.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_h6(csv_path: Path, out_dir: Path):
    """H6: Purification robustness — EDR before vs after FLUX purification."""
    plt, np, _ = _require_matplotlib()
    pd = _require_pandas()

    df = pd.read_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = df["checkpoint"].unique().tolist()
    palette = ["#2196F3", "#FF5722", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("H6: Does DiffVax-FLUX Immunization Resist FLUX Purification?\n"
                 "(Higher bars = purification fails = better product security)",
                 fontsize=12, fontweight="bold")

    x = np.arange(len(checkpoints))
    width = 0.35

    direct_edrs = [df[df["checkpoint"] == c]["direct_disrupted"].mean() for c in checkpoints]
    purified_edrs = [df[df["checkpoint"] == c]["purified_disrupted"].mean() for c in checkpoints]

    bars1 = ax.bar(x - width / 2, direct_edrs, width, label="Direct EDR (no purification)",
                   color="#4CAF50", alpha=0.85)
    bars2 = ax.bar(x + width / 2, purified_edrs, width,
                   label="Post-Purification EDR (H6 key metric)",
                   color="#FF5722", alpha=0.85)

    ax.axhline(0.7, color="green", linestyle="--", alpha=0.5, label="Target EDR (0.7)")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.4, label="Chance baseline (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints, fontsize=11)
    ax.set_ylabel("Edit Disruption Rate (EDR)", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom",
                    fontweight="bold", fontsize=10)

    plt.tight_layout()
    out_path = out_dir / "h6_purification_robustness.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["h1", "h2", "h6", "h7", "all"])
    parser.add_argument("--csv", help="Path to results CSV (required for single experiment)")
    parser.add_argument("--out", default="research/to_human/figures/",
                        help="Output directory for figures")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)

    if args.experiment == "all":
        # Try to find all CSVs automatically
        results_glob = {
            "h2": PROJECT_ROOT / "research/experiments/H2-patch-inference/results/patch_edr_metrics.csv",
            "h1": PROJECT_ROOT / "research/experiments/H1-multimodel-transfer/results/transfer_edr_metrics.csv",
            "h6": PROJECT_ROOT / "research/experiments/H6-purification-robustness/results/purification_robustness.csv",
        }
        for exp, csv_path in results_glob.items():
            if csv_path.exists():
                print(f"\nGenerating {exp} plot from {csv_path}")
                globals()[f"plot_{exp}"](csv_path, out_dir)
                # Also generate h7 from h1 CSV if it has jpeg_quality column
                if exp == "h1":
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    if "jpeg_quality" in df.columns and len(df["jpeg_quality"].unique()) > 1:
                        print("  Also generating H7 JPEG robustness plot")
                        plot_h7(csv_path, out_dir)
            else:
                print(f"No results yet for {exp} (expected: {csv_path})")
    else:
        if not args.csv:
            parser.error(f"--csv required for {args.experiment}")
        csv_path = Path(args.csv)
        globals()[f"plot_{args.experiment}"](csv_path, out_dir)


if __name__ == "__main__":
    main()
