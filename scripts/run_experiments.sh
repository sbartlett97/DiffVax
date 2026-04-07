#!/usr/bin/env bash
# DiffVax experiment launcher — run all research experiments in sequence.
#
# Requires: Python env with all dependencies, a GPU, and data in data/
#
# Usage:
#   bash scripts/run_experiments.sh [experiment]
#
# experiments:
#   h2       — Patch inference eval (no training needed, ~1h)
#   h1a      — Multi-model SD+FLUX training (~8h) + transfer eval
#   h1b      — Multi-model SD+FLUX+SD3 training (~10h) + transfer eval
#   h3       — 1088px fine-tuning (~12h)
#   h4       — VAE feature loss training (~8h) + transfer eval
#   h6       — Purification robustness eval (needs h1a checkpoint)
#   all      — Run h2, h1a, h6 in sequence

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs}"
EXPERIMENT="${1:-all}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

check_gpu() {
    python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU found'" || {
        log "ERROR: No GPU available. Set CUDA_VISIBLE_DEVICES or use cloud."
        exit 1
    }
    log "GPU: $(python3 -c "import torch; print(torch.cuda.get_device_name(0))")"
}

run_h2() {
    log "=== H2: Patch inference at 1088x1088 (no training) ==="
    CKPT="$PROJECT_DIR/checkpoints/diffvax_trained.pth"
    if [ ! -f "$CKPT" ]; then
        log "WARN: Checkpoint not found at $CKPT — run training first or download from HuggingFace"
        return
    fi
    python3 research/experiments/H2-patch-inference/code/run_patch_eval.py \
        --checkpoint "$CKPT" \
        --data-dir "$DATA_DIR" \
        --output-dir research/experiments/H2-patch-inference/results/ \
        --n-images 50
    log "H2 done. Results: research/experiments/H2-patch-inference/results/patch_edr_metrics.csv"
}

run_h1a() {
    log "=== H1a: Multi-model training (SD 25% + FLUX 75%) ==="
    python3 scripts/train.py \
        --config configs/train_multimodel.yml \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR/h1a"
    # Find the final checkpoint
    H1A_CKPT=$(find "$OUTPUT_DIR/h1a" -name "*_final.pth" | sort | tail -1)
    log "H1a checkpoint: $H1A_CKPT"
    log "=== H1a: Transfer evaluation ==="
    python3 research/experiments/H1-multimodel-transfer/code/eval_transfer.py \
        --checkpoints "sd15_only=$PROJECT_DIR/checkpoints/diffvax_trained.pth" "multimodel_flux=$H1A_CKPT" \
        --eval-models sd15 flux_schnell sd35 \
        --data-dir "$DATA_DIR" \
        --output-dir research/experiments/H1-multimodel-transfer/results/ \
        --n-images 50
    log "H1a done."
    # Generate transfer + JPEG robustness plots if results exist
    python3 research/src/plot_results.py all --out research/to_human/figures/ 2>/dev/null || true
}

run_h1b() {
    log "=== H1b: Multi-model training (SD + FLUX + SD3.5) ==="
    python3 scripts/train.py \
        --config configs/train_multimodel_sd3.yml \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR/h1b"
    log "H1b training done."
}

run_h3() {
    log "=== H3: 1088px fine-tuning ==="
    # First generate 1088px training data if not present
    if [ ! -d "$DATA_DIR/train_1088/images" ]; then
        log "Generating 1088px training data..."
        python3 scripts/generate_masks.py \
            --src "$DATA_DIR/train/images" \
            --dst "$DATA_DIR/train_1088" \
            --size 1088
    fi
    python3 scripts/train.py \
        --config configs/train_1088.yml \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR/h3"
    log "H3 done."
}

run_h4() {
    log "=== H4: VAE feature loss training ==="
    python3 scripts/train.py \
        --config configs/train_multimodel_h4.yml \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR/h4"
    H4_CKPT=$(find "$OUTPUT_DIR/h4" -name "*_final.pth" | sort | tail -1)
    H1A_CKPT=$(find "$OUTPUT_DIR/h1a" -name "*_final.pth" | sort | tail -1 || echo "")
    if [ -n "$H1A_CKPT" ]; then
        log "=== H4: Transfer comparison vs H1a ==="
        python3 research/experiments/H1-multimodel-transfer/code/eval_transfer.py \
            --checkpoints "h1a_no_vae_loss=$H1A_CKPT" "h4_vae_loss=$H4_CKPT" \
            --eval-models flux_schnell sd35 \
            --data-dir "$DATA_DIR" \
            --output-dir research/experiments/H4-vae-feature-loss/results/ \
            --n-images 50
    fi
    log "H4 done."
}

run_h7() {
    log "=== H7: JPEG-robust training (social media compression) ==="
    python3 scripts/train.py \
        --config configs/train_multimodel_h7.yml \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR/h7"
    H7_CKPT=$(find "$OUTPUT_DIR/h7" -name "*_final.pth" | sort | tail -1)
    H1A_CKPT=$(find "$OUTPUT_DIR/h1a" -name "*_final.pth" | sort | tail -1 || echo "")
    if [ -n "$H1A_CKPT" ]; then
        log "=== H7: Transfer comparison vs H1a (no JPEG aug) ==="
        python3 research/experiments/H1-multimodel-transfer/code/eval_transfer.py \
            --checkpoints "h1a_no_jpeg=$H1A_CKPT" "h7_jpeg_robust=$H7_CKPT" \
            --eval-models sd15 flux_schnell sd35 \
            --data-dir "$DATA_DIR" \
            --output-dir research/experiments/H7-jpeg-robust/results/ \
            --n-images 50
    fi
    log "H7 done."
}

run_h6() {
    log "=== H6: Purification robustness ==="
    SD15_CKPT="$PROJECT_DIR/checkpoints/diffvax_trained.pth"
    FLUX_CKPT=$(find "$OUTPUT_DIR/h1a" -name "*_final.pth" 2>/dev/null | sort | tail -1 || echo "")
    if [ -z "$FLUX_CKPT" ]; then
        log "WARN: H1a checkpoint not found. Run h1a first."
        return
    fi
    python3 research/experiments/H6-purification-robustness/code/eval_purification_robustness.py \
        --checkpoint-sd15 "$SD15_CKPT" \
        --checkpoint-flux "$FLUX_CKPT" \
        --data-dir "$DATA_DIR" \
        --output-dir research/experiments/H6-purification-robustness/results/ \
        --n-images 30
    log "H6 done."
}

# Main
check_gpu

case "$EXPERIMENT" in
    h2)   run_h2 ;;
    h1a)  run_h1a ;;
    h1b)  run_h1b ;;
    h3)   run_h3 ;;
    h4)   run_h4 ;;
    h6)   run_h6 ;;
    h7)   run_h7 ;;
    all)
        run_h2
        run_h1a
        run_h6
        run_h7
        ;;
    *)
        echo "Usage: $0 [h2|h1a|h1b|h3|h4|h6|h7|all]"
        exit 1
        ;;
esac

log "All requested experiments complete."
