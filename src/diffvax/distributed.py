"""Distributed (multi-GPU) training helpers.

DiffVax's multi-GPU strategy is *ensemble sharding*, not model sharding:
each rank pins ONE frozen attack surrogate for the whole run and holds its
own replica of the (tiny, ~9M-param) NestedUNet under DDP. Every rank
pushes its own images through its own surrogate; DDP all-reduces only the
NestedUNet gradients (~36 MB fp32), so the perturbation network learns from
the whole surrogate ensemble every step.

Why not the usual answers:
  - FSDP / DeepSpeed-ZeRO shard optimizer state, gradients, and parameters
    of the model being *trained*. Here the trained model is ~9M params with
    no meaningful optimizer state to shard; the memory is spent on a FROZEN
    surrogate's weights plus activations retained for a backward pass that
    only yields a gradient w.r.t. its *input*. ZeRO has no lever on either.
  - Tensor/context parallelism inside a surrogate would put the gradient
    path across device boundaries (the one thing this codebase has
    repeatedly had silently break) and needs fast interconnect. The
    all-reduce here is small enough to be fine over plain PCIe.

Consequence: the only hard requirement is that the LARGEST single surrogate
fits on ONE card.

Collective-safety note: every rank must execute the same number of
DDP-synchronised backward passes. Any per-rank early exit (an OOM skip, a
NaN abort) must therefore be agreed across ranks first — see
``any_rank_true``, which the training loop uses to turn a local "skip this
batch" into a global one.
"""

import os
from typing import Optional

import torch

try:  # torch.distributed is unavailable in some minimal builds
    import torch.distributed as dist

    _DIST_IMPORTABLE = True
except ImportError:  # pragma: no cover - depends on the torch build
    dist = None
    _DIST_IMPORTABLE = False


def is_distributed() -> bool:
    """True when a process group is initialised and holds more than one rank."""
    return (
        _DIST_IMPORTABLE
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )


def get_rank() -> int:
    """Global rank of this process (0 when not distributed)."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Total number of ranks (1 when not distributed)."""
    return dist.get_world_size() if is_distributed() else 1


def get_local_rank() -> int:
    """Rank within this node — selects which local GPU this process owns.

    Read from the environment (``torchrun`` sets ``LOCAL_RANK``) so it is
    available *before* the process group is initialised, which is when the
    CUDA device must be selected.
    """
    for env_var in ("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"):
        val = os.environ.get(env_var)
        if val is not None:
            return int(val)
    return 0


def is_main_process() -> bool:
    """True on rank 0 — the only rank that writes checkpoints, logs, and Hub uploads."""
    return get_rank() == 0


def init_distributed(backend: Optional[str] = None) -> bool:
    """Initialise the default process group if launched under torchrun.

    No-ops (returning False) when the required environment variables are
    absent, so the same entry point runs single-process unchanged.

    Backend defaults to ``nccl`` when CUDA is available and ``gloo``
    otherwise. gloo is what the CPU test suite uses; it is also the only
    option on Apple Silicon, where multi-GPU is moot anyway (one integrated
    GPU per machine).

    Returns:
        True if this call initialised a multi-rank process group.
    """
    if not _DIST_IMPORTABLE or not dist.is_available():
        return False
    if dist.is_initialized():
        return dist.get_world_size() > 1
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False
    if int(os.environ["WORLD_SIZE"]) <= 1:
        return False

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Bind this process to its own GPU before creating the process group so
    # NCCL picks the right device and later .cuda() calls land correctly.
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())

    dist.init_process_group(backend=backend)
    return dist.get_world_size() > 1


def cleanup_distributed() -> None:
    """Destroy the process group if one is active. Safe to call unconditionally."""
    if _DIST_IMPORTABLE and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """Block until every rank reaches this point. No-op when not distributed."""
    if is_distributed():
        dist.barrier()


def _collective_device(device: Optional[torch.device] = None) -> torch.device:
    """Device the collective's scratch tensor must live on for the active backend.

    NCCL operates on CUDA tensors; gloo on CPU tensors. Getting this wrong
    is a hang or a hard error, not a warning.
    """
    if is_distributed() and dist.get_backend() == "nccl":
        if device is not None and device.type == "cuda":
            return device
        return torch.device(f"cuda:{get_local_rank()}")
    return torch.device("cpu")


def all_reduce_mean(value: float, device: Optional[torch.device] = None) -> float:
    """Average a Python scalar across all ranks (identity when not distributed).

    Used so every rank agrees on the epoch loss, and therefore agrees on
    whether this epoch produced a new best model — otherwise ranks would
    disagree about whether to checkpoint.
    """
    if not is_distributed():
        return float(value)
    t = torch.tensor([float(value)], device=_collective_device(device))
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / get_world_size())


def any_rank_true(flag: bool, device: Optional[torch.device] = None) -> bool:
    """True on every rank if *flag* is True on any rank (identity when not distributed).

    This is what makes per-rank early exits collective-safe. If one rank hit
    an OOM and skipped its backward while the others proceeded, the others
    would block forever in DDP's gradient all-reduce waiting for a peer that
    is already on the next batch. Agreeing first means all ranks skip
    together and stay in lockstep.
    """
    if not is_distributed():
        return bool(flag)
    t = torch.tensor([1.0 if flag else 0.0], device=_collective_device(device))
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item() > 0.0)
