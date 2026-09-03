"""Device/dtype resolution tests (CUDA > MPS > CPU).

These test the *selection logic* in src/diffvax/utils.py by monkeypatching
torch's availability checks — they do not require real CUDA or MPS hardware,
so they run identically in CI and on a real Apple Silicon machine. What they
do NOT verify (and cannot, without physical hardware) is MPS kernel
correctness for the actual training ops (attention backward, FFT in
spectral_loss, kornia JPEG) — see research/findings.md.
"""

import torch
import pytest

from diffvax.utils import resolve_device, resolve_dtype, empty_cache, make_generator


def _patch_accelerators(monkeypatch, cuda: bool, mps: bool):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None:
        monkeypatch.setattr(mps_backend, "is_available", lambda: mps)


def test_resolve_device_prefers_cuda_over_mps(monkeypatch):
    _patch_accelerators(monkeypatch, cuda=True, mps=True)
    assert resolve_device().type == "cuda"


def test_resolve_device_prefers_mps_over_cpu(monkeypatch):
    _patch_accelerators(monkeypatch, cuda=False, mps=True)
    assert resolve_device().type == "mps"


def test_resolve_device_falls_back_to_cpu(monkeypatch):
    _patch_accelerators(monkeypatch, cuda=False, mps=False)
    assert resolve_device().type == "cpu"


@pytest.mark.parametrize("device_type,expected_dtype", [
    ("cuda", torch.float16),
    ("mps", torch.bfloat16),
    ("cpu", torch.float32),
])
def test_resolve_dtype_per_backend(device_type, expected_dtype):
    assert resolve_dtype(torch.device(device_type)) is expected_dtype


def test_empty_cache_is_noop_on_cpu():
    # Must not raise regardless of which accelerator (if any) is present.
    empty_cache(torch.device("cpu"))


def test_make_generator_cpu_fallback_for_mps():
    """diffusers documents that torch.Generator(device='mps') does not
    reproduce seeded results reliably — make_generator must always fall
    back to a CPU generator for MPS, regardless of hardware availability.
    """
    gen = make_generator(torch.device("mps"), seed=5)
    assert gen.device.type == "cpu"


def test_make_generator_matches_device_for_cpu():
    gen = make_generator(torch.device("cpu"), seed=5)
    assert gen.device.type == "cpu"


def test_make_generator_seed_is_deterministic():
    gen_a = make_generator(torch.device("cpu"), seed=42)
    gen_b = make_generator(torch.device("cpu"), seed=42)
    a = torch.randn(4, generator=gen_a)
    b = torch.randn(4, generator=gen_b)
    assert torch.equal(a, b)


def test_make_generator_no_seed_leaves_default_state():
    # Must not raise when seed is omitted (None).
    gen = make_generator(torch.device("cpu"))
    assert gen is not None
