#!/usr/bin/env bash
# run_post_h1a.sh — Run all evaluations immediately after H1a checkpoint is ready.
#
# Usage (on GPU instance, after git pull):
#   bash scripts/run_post_h1a.sh \
#       --h1a-checkpoint checkpoints/diffvax_multimodel.pth \
#       --sd15-checkpoint checkpoints/diffvax_trained.pth \
#       [--data-dir data] [--output-dir research/experiments]
#
# This script runs in sequence:
#   1. H1 transfer evaluation (SD1.5 / FLUX / SD3.5, with JPEG robustness baselines)
#   2. H6 purification robustness evaluation (purify-strengths 0.3 0.5 0.7)
#   3. Launch H7 JPEG-robust training in background
#
# Expected wall-clock: H1 eval ~3h + H6 eval ~4h + H7 training ~44h (background)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

H1A_CKPT=""
SD15_CKPT=""
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${PROJECT_ROOT}/research/experiments"
H7_CONFIG="${PROJECT_ROOT}/configs/train_multimodel_h7.yml"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --h1a-checkpoint) H1A_CKPT="$2"; shift 2;;
        --sd15-checkpoint) SD15_CKPT="$2"; shift 2;;
        --data-dir) DATA_DIR="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ -z "$H1A_CKPT" || -z "$SD15_CKPT" ]]; then
    echo "Usage: $0 --h1a-checkpoint <path> --sd15-checkpoint <path> [--data-dir <path>]"
    echo ""
    echo "Checkpoints are typically in: checkpoints/"
    echo "  SD15 baseline: checkpoints/diffvax_trained.pth  (original DiffVax checkpoint)"
    echo "  H1a result:    checkpoints/diffvax_multimodel.pth  (new multi-model checkpoint)"
    exit 1
fi

echo "=========================================="
echo "DiffVax++ Post-H1a Evaluation Suite"
echo "=========================================="
echo "H1a checkpoint: $H1A_CKPT"
echo "SD15 checkpoint: $SD15_CKPT"
echo "Data dir: $DATA_DIR"
echo ""

cd "$PROJECT_ROOT"

# ---- Step 1: H1 Transfer Evaluation ----
echo ">>> Step 1/3: H1 Transfer Evaluation"
echo "    Tests: SD1.5, FLUX.1-schnell, SD3.5"
echo "    Also records JPEG baselines at q=75, q=70 (for H7 comparison)"
echo ""

H1_OUT="${OUTPUT_DIR}/H1-multimodel-transfer/results"
mkdir -p "$H1_OUT"

python research/experiments/H1-multimodel-transfer/code/eval_transfer.py \
    --checkpoints \
        sd15_only="${SD15_CKPT}" \
        multimodel_h1a="${H1A_CKPT}" \
    --eval-models sd15 flux_schnell sd35 \
    --jpeg-qualities 75 70 \
    --data-dir "${DATA_DIR}" \
    --output-dir "${H1_OUT}" \
    --n-images 50

echo ">>> H1 eval complete. Results in: ${H1_OUT}/transfer_edr_metrics.csv"
echo ""

# ---- Step 2: H6 Purification Robustness ----
echo ">>> Step 2/3: H6 Purification Robustness"
echo "    Tests EditorClean purification at strengths 0.3, 0.5, 0.7"
echo ""

H6_OUT="${OUTPUT_DIR}/H6-purification-robustness/results"
mkdir -p "$H6_OUT"

python research/experiments/H6-purification-robustness/code/eval_purification_robustness.py \
    --checkpoint-sd15 "${SD15_CKPT}" \
    --checkpoint-flux "${H1A_CKPT}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${H6_OUT}" \
    --n-images 30 \
    --purify-strengths 0.3 0.5 0.7

echo ">>> H6 eval complete. Results in: ${H6_OUT}/purification_edr.csv"
echo ""

# ---- Step 3: H7 JPEG-robust training (background) ----
echo ">>> Step 3/3: Starting H7 JPEG-robust training in background"
echo "    Config: ${H7_CONFIG}"
echo "    Training SD(25%) + FLUX(75%) + JPEG aug (q=70-85, p=0.5)"
echo "    Expected: ~44h with max_steps=8000"
echo ""

H7_OUT="${PROJECT_ROOT}/outputs/h7"
mkdir -p "$H7_OUT"

nohup python scripts/train.py \
    --config "${H7_CONFIG}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${H7_OUT}" \
    > "${H7_OUT}/train_h7.log" 2>&1 &

H7_PID=$!
echo ">>> H7 training started (PID: ${H7_PID})"
echo "    Log: ${H7_OUT}/train_h7.log"
echo "    Monitor: tail -f ${H7_OUT}/train_h7.log"
echo ""

echo "=========================================="
echo "All evaluations complete. H7 training running."
echo ""
echo "Next steps:"
echo "  1. Review H1 results: ${H1_OUT}/transfer_edr_metrics.csv"
echo "  2. Review H6 results: ${H6_OUT}/purification_edr.csv"
echo "  3. Run plot_results.py after results are in:"
echo "     python research/src/plot_results.py all"
echo "  4. H7 checkpoint will be at: ${H7_OUT}/"
echo "     After training: eval H7 checkpoint with --jpeg-qualities 75 70"
echo "=========================================="
