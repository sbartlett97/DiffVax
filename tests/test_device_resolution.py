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
    ("mps", torch.bfloat16),
    ("cpu", torch.float32),
])
def test_resolve_dtype_per_backend(device_type, expected_dtype):
    assert resolve_dtype(torch.device(device_type)) is expected_dtype


@pytest.mark.parametrize("bf16_ok,expected", [
    (True, torch.bfloat16),    # Ampere+: no GradScaler needed
    (False, torch.float16),    # pre-Ampere fallback
])
def test_resolve_dtype_cuda_prefers_bf16(monkeypatch, bf16_ok, expected):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: bf16_ok)
    assert resolve_dtype(torch.device("cuda")) is expected


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


def test_grad_scaler_disabled_when_cuda_dtype_is_bf16():
    """The module-level scaler must be a passthrough whenever the compute
    dtype is not fp16 (bf16 needs no loss scaling)."""
    from diffvax.immunization import diffvax_immunization as d
    expected = torch.cuda.is_available() and resolve_dtype(torch.device("cuda")) == torch.float16
    assert d._SCALER_ENABLED == expected
