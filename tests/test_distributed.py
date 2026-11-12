"""Multi-GPU (DDP) tests — run entirely on CPU via the gloo backend.

The headline test spawns 2 real ranks and drives the REAL
``train_immunization_all_images_batch`` against a stub surrogate, asserting
that both ranks converge to bit-identical weights. That is the property that
actually matters: it can only hold if DDP's gradient all-reduce genuinely
fired, which in turn requires that the training loop calls the DDP wrapper
(not ``.forward()``, which bypasses the sync hooks) and that no rank
silently skipped a step.

These use gloo/CPU, so they verify the *wiring* — process group setup,
sampler partitioning, gradient synchronisation, collective-safe early exits,
rank-0-only IO. They do NOT verify NCCL behaviour or real multi-GPU memory
characteristics; no multi-GPU hardware was available. See
research/gpu-validation-runbook.md.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffvax.distributed import (  # noqa: E402
    all_reduce_mean, any_rank_true, get_local_rank, get_rank, get_world_size,
    is_distributed, is_main_process,
)

# Reuse the differentiable stub surrogate from the single-process smoke test
# so this exercises the same training path, just with 2 ranks.
from test_training_smoke import StubAttack, _write_dataset  # noqa: E402


# ---------------------------------------------------------------------------
# Single-process behaviour of the helpers (no process group initialised)
# ---------------------------------------------------------------------------

def test_helpers_degrade_to_single_process():
    """Every helper must be a safe identity when not running under torchrun,
    so the exact same code path runs single-process unchanged."""
    assert not is_distributed()
    assert get_rank() == 0
    assert get_world_size() == 1
    assert is_main_process() is True
    assert all_reduce_mean(2.5) == pytest.approx(2.5)
    assert any_rank_true(True) is True
    assert any_rank_true(False) is False


def test_get_local_rank_reads_env(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("OMPI_COMM_WORLD_LOCAL_RANK", raising=False)
    monkeypatch.delenv("SLURM_LOCALID", raising=False)
    assert get_local_rank() == 0
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert get_local_rank() == 3


def test_init_distributed_noop_without_env(monkeypatch):
    """Without RANK/WORLD_SIZE set, init must decline rather than hang trying
    to reach a nonexistent rendezvous."""
    from diffvax.distributed import init_distributed

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert init_distributed() is False


# ---------------------------------------------------------------------------
# Real 2-rank gloo workers
# ---------------------------------------------------------------------------

def _dist_setup(rank: int, world_size: int, port: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.distributed.init_process_group(backend="gloo")


def _collectives_worker(rank, world_size, port, result_path):
    try:
        _dist_setup(rank, world_size, port)
        out = {
            "rank": get_rank(),
            "world_size": get_world_size(),
            "is_distributed": is_distributed(),
            "is_main": is_main_process(),
            # rank 0 contributes 0.0, rank 1 contributes 2.0 -> mean 1.0
            "mean": all_reduce_mean(float(rank) * 2.0),
            # Only rank 1 raises the flag; BOTH ranks must observe True.
            "any_true": any_rank_true(rank == 1),
            "any_false": any_rank_true(False),
        }
        with open(f"{result_path}.{rank}", "w") as fh:
            json.dump(out, fh)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_collectives_across_two_gloo_ranks(tmp_path):
    world_size = 2
    result_path = str(tmp_path / "res")
    mp.spawn(
        _collectives_worker,
        args=(world_size, "29571", result_path),
        nprocs=world_size,
        join=True,
    )

    results = []
    for r in range(world_size):
        with open(f"{result_path}.{r}") as fh:
            results.append(json.load(fh))

    assert [r["rank"] for r in results] == [0, 1]
    assert all(r["world_size"] == 2 for r in results)
    assert all(r["is_distributed"] for r in results)
    assert [r["is_main"] for r in results] == [True, False]
    # Mean of {0.0, 2.0} is 1.0 on BOTH ranks.
    assert all(r["mean"] == pytest.approx(1.0) for r in results)
    # A flag raised on rank 1 alone must be observed by rank 0 too — this is
    # what makes the OOM-skip / NaN-abort paths deadlock-free.
    assert all(r["any_true"] for r in results)
    assert not any(r["any_false"] for r in results)


def _training_worker(rank, world_size, port, data_dir, out_dir, result_path):
    try:
        _dist_setup(rank, world_size, port)

        from diffvax.attack_manager import AttackModelManager
        from diffvax.immunization.diffvax_immunization import DiffVaxImmunization

        entries = [
            {"image_name": f"img{i}", "prompt": "a photo", "flux_prompt": "a photo"}
            for i in range(4)
        ]
        config = {
            "learning_rate": 2e-3,
            "resolution": 64,
            "batch_size": 1,
            "num_inference_steps": 2,
            "nb_filter": [4, 8, 16, 32, 64],
            "dataloader": {"num_workers": 0},
        }
        # Each rank pins its OWN surrogate instance — the production topology.
        manager = AttackModelManager(
            models={f"stub_rank{rank}": StubAttack()},
            probabilities={f"stub_rank{rank}": 1.0},
        )
        immunizer = DiffVaxImmunization(
            attack_manager=manager,
            config=config,
            output_dir=os.path.join(out_dir, f"rank{rank}"),
        )

        result = immunizer.train_immunization_all_images_batch(
            entries, data_dir, "images", "masks",
            size=(64, 64), iter_num=4, SEED=5, batch_size=1, alpha=1,
            strength_range=[0.6, 0.9],
        )

        # Flatten final weights so the parent can compare ranks bit-for-bit.
        flat = torch.cat([
            p.detach().flatten() for p in immunizer._unet_module.parameters()
        ])
        with open(f"{result_path}.{rank}", "w") as fh:
            json.dump(
                {
                    "rank": rank,
                    "completed": result is not None,
                    "is_ddp": immunizer.is_distributed,
                    "weights": flat.tolist(),
                },
                fh,
            )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.is_available(), reason="gloo/CPU-path test")
def test_two_rank_ddp_training_synchronises_weights(tmp_path):
    """The load-bearing test: 2 ranks train the real loop and must end with
    IDENTICAL NestedUNet weights.

    Identical weights are only possible if DDP's gradient all-reduce actually
    fired on every step. If the loop called ``.forward()`` (bypassing DDP's
    hooks), or a rank skipped a step out of lockstep, the ranks would drift
    apart and this assertion fails.
    """
    data_dir = str(tmp_path / "data")
    out_dir = str(tmp_path / "out")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    _write_dataset(data_dir, n=4)

    world_size = 2
    result_path = str(tmp_path / "train")
    mp.spawn(
        _training_worker,
        args=(world_size, "29573", data_dir, out_dir, result_path),
        nprocs=world_size,
        join=True,
    )

    results = []
    for r in range(world_size):
        with open(f"{result_path}.{r}") as fh:
            results.append(json.load(fh))

    assert all(r["completed"] for r in results), "A rank aborted training"
    assert all(r["is_ddp"] for r in results), "DDP was not actually engaged"

    w0 = torch.tensor(results[0]["weights"])
    w1 = torch.tensor(results[1]["weights"])
    assert w0.numel() > 0
    assert torch.equal(w0, w1), (
        "Ranks ended with different weights — DDP gradient all-reduce did not "
        f"synchronise (max abs diff {(w0 - w1).abs().max().item():.3e}). "
        "Check that the loop calls self.unetmodel(...) and not .forward()."
    )


@pytest.mark.skipif(torch.cuda.is_available(), reason="gloo/CPU-path test")
def test_two_rank_ddp_writes_only_rank0_checkpoints(tmp_path):
    """Only rank 0 may write checkpoints; a non-zero rank writing to the same
    paths would race and interleave partial files."""
    data_dir = str(tmp_path / "data")
    out_dir = str(tmp_path / "out")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    _write_dataset(data_dir, n=4)

    world_size = 2
    result_path = str(tmp_path / "ckpt")
    mp.spawn(
        _training_worker,
        args=(world_size, "29575", data_dir, out_dir, result_path),
        nprocs=world_size,
        join=True,
    )

    rank0_ckpts = list((tmp_path / "out" / "rank0").rglob("*.pth"))
    rank1_ckpts = list((tmp_path / "out" / "rank1").rglob("*.pth"))
    assert rank0_ckpts, "Rank 0 wrote no checkpoint"
    assert not rank1_ckpts, f"Non-zero rank wrote checkpoints: {rank1_ckpts}"
