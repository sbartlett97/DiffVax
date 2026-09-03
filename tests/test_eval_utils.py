"""Unit tests for the checkpoint-loading utilities shared by scripts/evaluate.py
and scripts/eval_multimodel.py (diffvax.utils.load_perturbation_net,
immunize_image_pil), and evaluate.py's small _aggregate() helper.

Covers both checkpoint formats load_perturbation_net must accept: a raw
.pth state_dict (legacy format, requires the caller to know nb_filter) and a
save_pretrained() directory (recovers nb_filter automatically) — the same
code path a Hugging Face Hub repo id takes, since NestedUNet.from_pretrained()
handles local directories and Hub repo ids identically.
"""

import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from diffvax.model import NestedUNet  # noqa: E402
from diffvax.utils import load_perturbation_net, immunize_image_pil  # noqa: E402


def test_load_perturbation_net_from_raw_pth(tmp_path):
    net = NestedUNet(num_classes=3, nb_filter=[4, 8, 16, 32, 64])
    pth_path = tmp_path / "ckpt.pth"
    torch.save(net.state_dict(), pth_path)

    loaded = load_perturbation_net(
        str(pth_path), nb_filter=[4, 8, 16, 32, 64], device=torch.device("cpu")
    )
    assert isinstance(loaded, NestedUNet)
    assert not loaded.training, "loaded net should be in eval() mode"
    assert all(not p.requires_grad for p in loaded.parameters())


def test_load_perturbation_net_from_save_pretrained_dir(tmp_path):
    """Same code path a Hub repo id takes — NestedUNet.from_pretrained()
    handles a local directory and a Hub repo id identically, and correctly
    recovers a non-default nb_filter from config.json without the caller
    having to specify it."""
    net = NestedUNet(num_classes=3, nb_filter=[4, 8, 16, 32, 64])
    out_dir = tmp_path / "saved_model"
    net.save_pretrained(str(out_dir))

    loaded = load_perturbation_net(str(out_dir), device=torch.device("cpu"))
    assert loaded.nb_filter == [4, 8, 16, 32, 64]
    assert all(not p.requires_grad for p in loaded.parameters())

    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(net.eval()(x), loaded(x))


def test_immunize_image_pil_output_shape_and_type():
    net = NestedUNet(num_classes=3, nb_filter=[4, 8, 16, 32, 64]).eval()
    img = Image.new("RGB", (64, 64), color=(50, 100, 150))

    out = immunize_image_pil(net, img, device=torch.device("cpu"), dtype=torch.float32)

    assert isinstance(out, Image.Image)
    assert out.size == (64, 64)
    assert out.mode == "RGB"


def test_immunize_image_pil_mask_gating_confines_perturbation_to_subject():
    """mask_pil, when passed, must zero the perturbation wherever mask==1
    (dataset convention: 1=background, 0=subject) — mirrors the training
    loop's perturbation_mask_gating so eval of a mask-gated checkpoint
    actually reflects how it was trained.
    """
    import numpy as np

    net = NestedUNet(num_classes=3, nb_filter=[4, 8, 16, 32, 64]).eval()
    rng = np.random.default_rng(0)
    img_np = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(img_np, "RGB")

    mask_np = np.full((64, 64), 255, dtype=np.uint8)  # background=1 (white)
    mask_np[16:48, 16:48] = 0  # subject=0 (black) center square
    mask = Image.fromarray(mask_np, "L")

    out = immunize_image_pil(
        net, img, device=torch.device("cpu"), dtype=torch.float32, mask_pil=mask,
    )
    out_np = np.array(out)

    background = mask_np == 255
    subject = ~background
    assert np.abs(out_np[background].astype(int) - img_np[background].astype(int)).max() <= 1, (
        "Perturbation leaked into the background (mask=1) region"
    )
    assert np.abs(out_np[subject].astype(int) - img_np[subject].astype(int)).max() > 1, (
        "No perturbation detected in the subject region — gating may be inverted or a no-op"
    )


def test_aggregate_helper_skips_missing_keys_and_empty_list():
    from evaluate import _aggregate

    assert _aggregate([{"psnr": 10.0}, {"psnr": 20.0}], "psnr") == 15.0
    assert _aggregate([{"psnr": 10.0}, {}], "psnr") == 10.0
    assert _aggregate([], "psnr") is None
