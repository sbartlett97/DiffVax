"""CPU smoke test for the REAL DiffVaxImmunization training loop.

Runs ``train_immunization_all_images_batch`` end to end — dataset streaming
from disk, NestedUNet perturbation, clamp, attack forward, loss1/loss2,
GradScaler (passthrough on CPU), optimizer step, checkpointing, reporter —
against a tiny differentiable stub attack model, and asserts:

  S1: the loop completes and saves a final checkpoint;
  S2: the epoch-average loss DECREASES, i.e. the training method produces a
      usable learning signal through its own plumbing (not a reimplementation);
  S3: NestedUNet parameters actually changed (optimizer steps were applied,
      no silent step-skipping).
"""

import json
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffvax.attack_base import BaseAttack  # noqa: E402


class StubAttack(BaseAttack):
    """Differentiable stand-in for a diffusion surrogate.

    Output = 0.9 * gaussian_blur(input): content-dependent and differentiable,
    so loss1 (push output toward zero) has a well-defined descent direction
    (darken the adversarial image), balanced against loss2 (perturbation
    magnitude). The training loop must reduce the combined loss.
    """

    def __init__(self):
        k = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
        ) / 16.0
        self._kernel = k.expand(3, 1, 3, 3).clone()

    def attack(self, prompt, image, mask=None, height=64, width=64,
               num_inference_steps=4, batch_size=1, strength=1.0):
        img = image.float()
        blurred = F.conv2d(img, self._kernel.to(img.device), padding=1, groups=3)
        return (blurred * 0.9).to(image.dtype)

    def to_device(self, device):
        pass

    def to_cpu(self):
        pass

    @property
    def loss_uses_mask_weighting(self):
        return False

    @property
    def is_inpainting(self):
        return False

    @property
    def native_resolution(self):
        return 512


def _write_dataset(root, n=2):
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "masks"), exist_ok=True)
    rng = np.random.default_rng(0)
    entries = []
    for i in range(n):
        name = f"img{i}"
        arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(os.path.join(root, "images", f"{name}.png"))
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 255
        Image.fromarray(mask, "L").save(
            os.path.join(root, "masks", f"mask_{name}.png")
        )
        entries.append(
            {"image_name": name, "prompt": "a photo", "flux_prompt": "a photo"}
        )
    return entries


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-path smoke test")
def test_training_loop_smoke_cpu(tmp_path):
    from diffvax.attack_manager import AttackModelManager
    from diffvax.immunization.diffvax_immunization import DiffVaxImmunization

    data_dir = str(tmp_path / "data")
    out_dir = str(tmp_path / "out")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    entries = _write_dataset(data_dir)

    config = {
        "learning_rate": 2e-3,
        "resolution": 64,
        "batch_size": 1,
        "num_inference_steps": 2,
        "nb_filter": [4, 8, 16, 32, 64],  # tiny H6 variant for CPU speed
        "dataloader": {"num_workers": 0},
    }

    manager = AttackModelManager(
        models={"sd_stub": StubAttack()},
        probabilities={"sd_stub": 1.0},
    )
    immunizer = DiffVaxImmunization(
        attack_manager=manager, config=config, output_dir=out_dir
    )
    params_before = [p.detach().clone() for p in immunizer.unetmodel.parameters()]

    result = immunizer.train_immunization_all_images_batch(
        entries,
        data_dir,
        "images",
        "masks",
        size=(64, 64),
        iter_num=8,
        SEED=5,
        batch_size=1,
        alpha=1,
        strength_range=[0.6, 0.9],
    )

    # S1: completed and saved a final checkpoint
    assert result is not None, "Training aborted (NaN path returns None)"
    _, final_path = result
    assert os.path.exists(final_path), f"Final checkpoint missing: {final_path}"

    # S2: epoch-average loss decreased (learning signal flows end to end)
    log_path = os.path.join(out_dir, "training_log.json")
    with open(log_path) as fh:
        events = json.load(fh)
    epoch_losses = [e["avg_loss"] for e in events if e["type"] == "epoch"]
    assert len(epoch_losses) == 8
    assert all(np.isfinite(epoch_losses)), f"Non-finite losses: {epoch_losses}"
    first2 = float(np.mean(epoch_losses[:2]))
    last2 = float(np.mean(epoch_losses[-2:]))
    assert last2 < first2, (
        f"Epoch loss did not decrease through the real training loop "
        f"(first2={first2:.5f}, last2={last2:.5f}): {epoch_losses}"
    )

    # S3: optimizer steps were applied (no silent step-skipping)
    changed = any(
        not torch.equal(pb, pa.detach())
        for pb, pa in zip(params_before, immunizer.unetmodel.parameters())
    )
    assert changed, "NestedUNet parameters unchanged — steps were skipped"


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-path smoke test")
def test_perturbation_mask_gating_confines_perturbation_to_subject_region(tmp_path):
    """perturbation_mask_gating=true must zero the perturbation network's
    output wherever mask==1 (dataset convention: 1=background, 0=subject),
    verified end to end by intercepting the exact tensor the attack surrogate
    receives — not by re-deriving the arithmetic in the test.
    """
    from diffvax.attack_manager import AttackModelManager
    from diffvax.immunization.diffvax_immunization import (
        DiffVaxImmunization,
        ImmunizationDataset,
    )

    data_dir = str(tmp_path / "data")
    out_dir = str(tmp_path / "out")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    entries = _write_dataset(data_dir, n=1)

    captured = {}

    class RecordingStubAttack(StubAttack):
        def attack(self, prompt, image, mask=None, height=64, width=64,
                   num_inference_steps=4, batch_size=1, strength=1.0):
            captured["image"] = image.detach().clone()
            return super().attack(
                prompt, image, mask, height, width,
                num_inference_steps, batch_size, strength,
            )

    config = {
        "learning_rate": 2e-3,
        "resolution": 64,
        "batch_size": 1,
        "num_inference_steps": 2,
        "nb_filter": [4, 8, 16, 32, 64],
        "dataloader": {"num_workers": 0},
        "perturbation_mask_gating": True,
    }
    manager = AttackModelManager(
        models={"sd_stub": RecordingStubAttack()},
        probabilities={"sd_stub": 1.0},
    )
    immunizer = DiffVaxImmunization(
        attack_manager=manager, config=config, output_dir=out_dir
    )
    result = immunizer.train_immunization_all_images_batch(
        entries, data_dir, "images", "masks", size=(64, 64),
        iter_num=1, SEED=5, batch_size=1, alpha=1, strength_range=[0.6, 0.9],
    )
    assert result is not None
    assert "image" in captured, "Stub attack was never called"

    # Load the exact same clean image/mask tensors the loop trained on, via
    # the same dataset class, so normalization can't drift from the loop's.
    dataset = ImmunizationDataset(
        entries, data_dir, "images", "masks", (64, 64),
        dtype=captured["image"].dtype,
    )
    img_t, mask_t, _, _ = dataset[0]
    received = captured["image"][0].detach().cpu().float()  # (3, H, W)
    img_t = img_t.float().cpu()
    background = mask_t[0].bool().cpu()  # mask==1 -> background per dataset convention

    for c in range(3):
        assert torch.allclose(received[c][background], img_t[c][background], atol=1e-3), (
            "Perturbation leaked into the background (mask=1) region"
        )
    subject = ~background
    assert not torch.allclose(
        received[:, subject], img_t[:, subject], atol=1e-3
    ), "No perturbation detected in the subject region — gating may be inverted or a no-op"


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-path smoke test")
def test_perturbation_mask_gating_survives_all_background_mask(tmp_path):
    """A real face mask can come back all-background (face detection found
    nothing) — subject_region.sum() is then exactly 0, which would divide by
    zero in loss2's weighted normalizer if not guarded. Must fall back to
    whole-image gating for that batch instead of aborting the run with NaN.
    """
    from diffvax.attack_manager import AttackModelManager
    from diffvax.immunization.diffvax_immunization import DiffVaxImmunization

    data_dir = str(tmp_path / "data")
    out_dir = str(tmp_path / "out")
    os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "masks"), exist_ok=True)
    rng = np.random.default_rng(0)
    name = "img0"
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(os.path.join(data_dir, "images", f"{name}.png"))
    # All-background mask: every pixel is 1 (background), zero subject pixels.
    mask = np.full((64, 64), 255, dtype=np.uint8)
    Image.fromarray(mask, "L").save(os.path.join(data_dir, "masks", f"mask_{name}.png"))
    entries = [{"image_name": name, "prompt": "a photo", "flux_prompt": "a photo"}]

    config = {
        "learning_rate": 2e-3,
        "resolution": 64,
        "batch_size": 1,
        "num_inference_steps": 2,
        "nb_filter": [4, 8, 16, 32, 64],
        "dataloader": {"num_workers": 0},
        "perturbation_mask_gating": True,
    }
    manager = AttackModelManager(
        models={"sd_stub": StubAttack()}, probabilities={"sd_stub": 1.0}
    )
    immunizer = DiffVaxImmunization(
        attack_manager=manager, config=config, output_dir=out_dir
    )
    result = immunizer.train_immunization_all_images_batch(
        entries, data_dir, "images", "masks", size=(64, 64),
        iter_num=1, SEED=5, batch_size=1, alpha=1, strength_range=[0.6, 0.9],
    )
    assert result is not None, "Training aborted (NaN path returns None) on an all-background mask"
