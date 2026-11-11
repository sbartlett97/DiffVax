#!/usr/bin/env bash
# One-shot setup for a RunPod pod (80 GB GPU, network volume at /workspace).
#
#   export HF_TOKEN=hf_xxx          # REQUIRED: the dataset sbartlett97/diffvax-high-res is private
#   bash scripts/runpod_setup.sh [--with-zoo]
#
# --with-zoo also pre-downloads the unseen evaluation editors (SD3.5 Medium,
# FLUX.1 Kontext, Qwen-Image-Edit; ~165 GB, two of them gated).
# Everything persistent (venv, HF cache, outputs) lives under /workspace so a
# stopped/restarted pod keeps it.
set -euo pipefail

WITH_ZOO=0
[[ "${1:-}" == "--with-zoo" ]] && WITH_ZOO=1

: "${HF_TOKEN:?Set HF_TOKEN first (private dataset + gated models need it)}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_TOKEN
mkdir -p "$HF_HOME"

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'PY'
import torch, sys
assert torch.cuda.is_available(), "no CUDA device visible"
free, total = torch.cuda.mem_get_info()
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
      f"bf16={torch.cuda.is_bf16_supported()}  total={total/2**30:.0f} GiB")
if total / 2**30 < 70:
    sys.exit("Need an 80 GB card for the 1088px runs (found <70 GiB).")
PY

echo "== Disk"
df -h /workspace | tail -1

echo "== Python env (uv)"
VENV=/workspace/venv
# Ubuntu 24.04 images block `pip install` into the system Python (PEP 668);
# the RunPod image already ships uv, so only install it if it is missing.
command -v uv >/dev/null || python -m pip install -q --break-system-packages uv
# --system-site-packages keeps the template's CUDA torch. Pin it with a
# constraints file so no dependency can swap the CUDA build (a plain `pip
# install` once replaced torch 2.9.1+cu128 with 2.15.0+cu132).
[[ -d "$VENV" ]] || uv venv --system-site-packages "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python - <<'PY' > /workspace/torch-constraints.txt
import importlib.metadata as m
for p in ("torch", "torchvision", "torchaudio"):
    try:
        # Drop the local "+cu128" suffix: PyPI does not serve local versions, and
        # its Linux wheel for the same release is the matching CUDA build.
        print(f"{p}=={m.version(p).split('+')[0]}")
    except m.PackageNotFoundError:
        pass
PY
cat /workspace/torch-constraints.txt
uv pip install -q -r requirements-pod.txt -c /workspace/torch-constraints.txt
uv pip install -q -e . -c /workspace/torch-constraints.txt
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
uv pip check || echo "WARNING: uv pip check reported conflicts above - review before training"

echo "== Tests (CPU suite; the 2-rank DDP tests are skipped on CUDA hosts)"
python -m pytest tests/ -q || echo "WARNING: some tests failed on this host - read the output before training"

echo "== Dataset (private repo)"
python scripts/download_dataset.py --data-dir data

echo "== Pre-fetching models"
python - <<PY
from huggingface_hub import snapshot_download
ids = ["black-forest-labs/FLUX.2-klein-4B"]          # ungated, Apache-2.0, ~24 GB
if $WITH_ZOO:
    ids += ["stabilityai/stable-diffusion-3.5-medium",   # gated: accept licence
            "black-forest-labs/FLUX.1-Kontext-dev",      # gated: accept licence
            "Qwen/Qwen-Image-Edit-2509",
            "runwayml/stable-diffusion-inpainting"]
for i in ids:
    print("downloading", i, flush=True)
    snapshot_download(i, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model"])
PY

cat <<'MSG'

Setup complete. Next (see research/gpu-validation-runbook.md):

  source /workspace/venv/bin/activate
  export HF_HOME=/workspace/hf-cache HF_TOKEN=...
  python scripts/profile_memory.py --config configs/klein_edit_512.yml --resolutions 512 1088 --batch-sizes 1
  python scripts/train.py --config configs/klein_edit_512.yml --output-dir /workspace/outputs/klein512
MSG
