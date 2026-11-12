"""End-to-end smoke of scripts/eval_protocol.py with the CPU STUB editor:
checkpoint load (epsilon-bounded), protect, conditions, social transform,
records, bootstrap summary, report files."""
import json
import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from diffvax.model import NestedUNet  # noqa: E402


def test_eval_protocol_cli_with_stub_editor(tmp_path):
    split = tmp_path / "data" / "validation"
    (split / "images").mkdir(parents=True); (split / "masks").mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(4):
        n = f"im{i}"
        Image.fromarray(rng.integers(40, 200, (64, 64, 3), dtype=np.uint8)).save(split / "images" / f"{n}.png")
        Image.fromarray(np.full((64, 64), 255, np.uint8)).save(split / "masks" / f"person_{n}.png")
        rows.append({"file_name": f"images/{n}.png", "mask": f"masks/person_{n}.png",
                     "masks": {"person": f"masks/person_{n}.png"},
                     "prompts": ["a", "b"], "flux_prompts": ["a", "b"]})
    (split / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    torch.manual_seed(0)
    net = NestedUNet(nb_filter=[4, 8, 16, 32, 64], epsilon=16)
    ck = tmp_path / "net.pth"
    torch.save(net.state_dict(), ck)

    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, "-I", os.path.join(ROOT, "scripts", "eval_protocol.py"),
         "--checkpoint", str(ck), "--epsilon", "16", "--nb-filter", "4", "8", "16", "32", "64",
         "--data-dir", str(tmp_path / "data"), "--editors", "STUB", "--divergence", os.environ.get("DIV", "l1"),
         "--max-images", "4", "--resolution", "32", "--seeds", "0", "1", "--n-boot", "100",
         "--output-dir", str(out)],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert r.returncode == 0, r.stderr[-2000:]
    rep = json.loads((out / "protocol_report.json").read_text())
    assert rep["perturbation"]["linf_max"] <= 16
    assert rep["perturbation"]["image_size"] == 32      # --resolution honoured (inputs are 64px)
    assert len(rep["records"]) == 4 * 2 * 2
    assert rep["verdicts"]["STUB"] in {"PROTECTED", "NO EFFECT", "WORSE THAN CONTROL"}
    assert (out / "protocol_report.md").exists()
