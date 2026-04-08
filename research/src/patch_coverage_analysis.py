#!/usr/bin/env python3
"""Compute and visualise the patch coverage density map for 1088x1088 tiling.

This is the core mechanistic evidence for H2: the 'perturbation accumulation'
hypothesis predicts that pixels covered by more patches receive stronger
immunization. This script computes the exact coverage count and Gaussian-
weighted density for each pixel, generating the Figure 2 heatmap for the paper.

No GPU or model required — pure analytical computation.
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "research" / "to_human" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def gaussian_window(size: int, sigma_ratio: float = 0.125) -> np.ndarray:
    """Create 2D Gaussian window matching SD upscaler blending (scipy.signal.windows.gaussian)."""
    from scipy.signal.windows import gaussian
    window_1d = gaussian(size, std=size * sigma_ratio)
    window_2d = np.outer(window_1d, window_1d)
    return window_2d


def compute_patch_coverage(img_size: int, patch_size: int, stride: int):
    """Compute per-pixel patch coverage count and Gaussian-weighted sum.

    Returns:
        coverage_count: (H, W) int — number of patches covering each pixel
        gaussian_sum: (H, W) float — sum of Gaussian weights across covering patches
        offsets: list of (row, col) top-left corners of each patch
    """
    H = W = img_size
    offsets = []
    r = 0
    while True:
        c = 0
        row_added = False
        while True:
            offsets.append((r, c))
            row_added = True
            if c + patch_size >= W:
                break
            c = min(c + stride, W - patch_size)
        if r + patch_size >= H:
            break
        r = min(r + stride, H - patch_size)

    coverage_count = np.zeros((H, W), dtype=int)
    gaussian_sum = np.zeros((H, W), dtype=float)
    gwin = gaussian_window(patch_size)

    for (pr, pc) in offsets:
        coverage_count[pr:pr + patch_size, pc:pc + patch_size] += 1
        gaussian_sum[pr:pr + patch_size, pc:pc + patch_size] += gwin

    return coverage_count, gaussian_sum, offsets


def plot_coverage_heatmap(img_size: int = 1088, patch_size: int = 512):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print("matplotlib/scipy required. Run: pip install matplotlib scipy")
        return

    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.3)

    configs = [
        (512, "No overlap\n(stride=512)\nEDR=0.300", "Reds"),
        (384, "25% overlap\n(stride=384)\nEDR=0.330", "Oranges"),
        (256, "50% overlap ★\n(stride=256)\nEDR=0.400", "Greens"),
    ]

    for col, (stride, label, cmap) in enumerate(configs):
        count, gsum, offsets = compute_patch_coverage(img_size, patch_size, stride)

        # Normalised density: Gaussian-weighted overlap relative to max
        # This is what each pixel contributes as perturbation density
        norm_density = gsum / gsum.max()

        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(norm_density, cmap=cmap, vmin=0, vmax=1, origin="upper")

        # Draw patch borders for first config
        if stride == 512:
            for (pr, pc) in offsets:
                rect = plt.Rectangle((pc - 0.5, pr - 0.5), patch_size, patch_size,
                                      fill=False, edgecolor="white", linewidth=1.5, linestyle="--")
                ax.add_patch(rect)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Norm. perturbation density")

        # Annotate max coverage count
        max_count = count.max()
        center_count = count[img_size // 2, img_size // 2]
        ax.set_title(f"{label}\n{len(offsets)} patches, max coverage={max_count}",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Pixel column", fontsize=8)
        if col == 0:
            ax.set_ylabel("Pixel row", fontsize=8)

        # Mark the center pixel coverage count
        cx, cy = img_size // 2, img_size // 2
        ax.plot(cx, cy, "wx", markersize=10, linewidth=2)
        ax.text(cx + 20, cy - 40, f"center:\n{center_count} patches",
                color="white", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))

    # Panel 4: EDR vs overlap, annotated with mechanism
    ax4 = fig.add_subplot(gs[0, 3])
    strides = [512, 384, 256]
    edrs = [0.300, 0.330, 0.400]
    overlap_pcts = [0, 25, 50]
    max_coverages = []
    center_coverages = []
    for s in strides:
        cnt, _, _ = compute_patch_coverage(img_size, patch_size, s)
        max_coverages.append(int(cnt.max()))
        center_coverages.append(int(cnt[img_size // 2, img_size // 2]))

    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    bars = ax4.bar(overlap_pcts, edrs, color=colors, edgecolor="white", linewidth=1.5, width=18)
    ax4.axhline(0.250, color="gray", linestyle=":", alpha=0.8, label="Baseline 512px (EDR=0.250)")

    for bar, edr, cc in zip(bars, edrs, center_coverages):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"EDR={edr:.3f}\ncenter={cc}p", ha="center", va="bottom",
                 fontsize=8, fontweight="bold")

    ax4.set_xticks(overlap_pcts)
    ax4.set_xticklabels([f"{p}%\noverlap" for p in overlap_pcts], fontsize=9)
    ax4.set_ylabel("Edit Disruption Rate (EDR)", fontsize=9)
    ax4.set_title("H2: EDR vs Overlap\n(center patch count predicts EDR)",
                  fontsize=9, fontweight="bold")
    ax4.set_ylim(0, 0.55)
    ax4.legend(fontsize=8)

    fig.suptitle(
        f"Patch Coverage Density: 1088×1088 image, 512×512 patches\n"
        "Brighter = more Gaussian-weighted patch overlap = stronger immunization",
        fontsize=11, fontweight="bold", y=1.03,
    )

    out_path = OUT_DIR / "patch_coverage_density.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def print_coverage_stats(img_size: int = 1088, patch_size: int = 512):
    """Print key stats for paper claims verification."""
    print(f"\n{'='*60}")
    print(f"Patch coverage stats: {img_size}×{img_size} image, {patch_size}×{patch_size} patches")
    print(f"{'='*60}")
    print(f"{'Stride':8s} | {'Overlap':7s} | {'N patches':9s} | {'Max cov':7s} | {'Center cov':10s} | {'EDR (obs)':9s}")
    print("-" * 60)
    observed_edrs = {512: 0.300, 384: 0.330, 256: 0.400}
    for stride in [512, 384, 256]:
        cnt, gsum, offsets = compute_patch_coverage(img_size, patch_size, stride)
        overlap_pct = int(100 * (1 - stride / patch_size))
        max_cov = cnt.max()
        center_cov = cnt[img_size // 2, img_size // 2]
        n_patches = len(offsets)
        edr = observed_edrs.get(stride, "?")
        print(f"{stride:8d} | {overlap_pct:6d}% | {n_patches:9d} | {max_cov:7d} | {center_cov:10d} | {edr:9.3f}")

    print()
    # Verify the paper claim "~4 overlapping patches at center"
    cnt_256, _, _ = compute_patch_coverage(img_size, patch_size, 256)
    print(f"Paper claim verification (stride=256):")
    print(f"  Center pixel ({img_size//2}, {img_size//2}) covered by: "
          f"{cnt_256[img_size//2, img_size//2]} patches")
    print(f"  Mean coverage across all pixels: {cnt_256.mean():.2f}")
    print(f"  Pixels with coverage >= 4: "
          f"{(cnt_256 >= 4).sum()} ({100*(cnt_256 >= 4).mean():.1f}% of image)")
    print(f"  Pixels with coverage == 1 (edges/corners): "
          f"{(cnt_256 == 1).sum()} ({100*(cnt_256 == 1).mean():.1f}% of image)")

    # Verify EDR correlation with center coverage
    edrs_list = [0.300, 0.330, 0.400]
    center_covs = []
    for stride in [512, 384, 256]:
        cnt, _, _ = compute_patch_coverage(img_size, patch_size, stride)
        center_covs.append(cnt[img_size // 2, img_size // 2])
    corr = np.corrcoef(center_covs, edrs_list)[0, 1]
    print(f"\n  Center coverage vs EDR correlation: r={corr:.4f}")
    print(f"  Center coverages: {center_covs} (stride=512,384,256)")
    print(f"  EDRs:             {edrs_list}")
    print(f"  → {'SUPPORTS' if corr > 0.9 else 'DOES NOT SUPPORT'} perturbation accumulation hypothesis")


if __name__ == "__main__":
    print_coverage_stats()
    print("\nGenerating heatmap figure...")
    plot_coverage_heatmap()
    print("\nDone. Figure saved to research/to_human/figures/patch_coverage_density.png")
