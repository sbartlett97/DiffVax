"""Hard epsilon bound (NestedUNet) and surrogate hold_batches (manager)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffvax.model import NestedUNet                      # noqa: E402
from diffvax.attack_manager import AttackModelManager     # noqa: E402


def _net(**kw):
    torch.manual_seed(0)
    return NestedUNet(nb_filter=[4, 8, 16, 32, 64], **kw).eval()


def test_unbounded_by_default_matches_legacy():
    """epsilon=None must not change the forward pass at all."""
    a, b = _net(), _net(epsilon=None)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(a(x), b(x))


def test_epsilon_is_bounded_in_pixel_units():
    """16/255 pixel budget: after quantising to uint8 the max difference <=16.
    (Tensors are in [-1,1], so the tensor bound is 2*16/255 — a literal
    16/255 on the tensor would be only 8/255 in pixels.)"""
    net = _net(epsilon=16)
    # Blow the final layer up so the raw output is far outside the budget.
    with torch.no_grad():
        net.final.weight.mul_(1000.0)
    img = torch.rand(1, 3, 32, 32) * 1.6 - 0.8          # leave clamp headroom
    with torch.no_grad():
        delta = net(img)
    assert delta.abs().max().item() <= 2 * 16 / 255 + 1e-6
    to_u8 = lambda t: np.round((t.clamp(-1, 1) / 2 + 0.5).numpy() * 255).astype(np.int32)
    diff = np.abs(to_u8(img + delta) - to_u8(img)).max()
    assert diff <= 16, diff
    assert delta.abs().max().item() > 0.9 * 2 * 16 / 255   # actually saturates


def test_epsilon_is_near_identity_for_small_outputs_and_keeps_gradient():
    bounded, raw = _net(epsilon=16), _net()
    for n in (bounded, raw):
        with torch.no_grad():
            n.final.weight.mul_(1e-3)       # raw output << eps (0.125)
            n.final.bias.zero_()
    x = torch.randn(1, 3, 32, 32)
    out, ref = bounded(x), raw(x)
    assert ref.abs().max() < 0.01
    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-5)
    out.abs().mean().backward()
    assert sum(p.grad.abs().sum() for p in bounded.parameters() if p.grad is not None) > 0


def test_epsilon_round_trips_through_hub_config():
    net = NestedUNet(nb_filter=[4, 8, 16, 32, 64], epsilon=16)
    assert net._hub_mixin_config["epsilon"] == 16


class _Stub:
    def __init__(self): self.moves = 0
    def to_device(self, d): self.moves += 1
    def to_cpu(self): pass


def test_hold_batches_limits_swaps():
    import random
    random.seed(0)
    models = {"a": _Stub(), "b": _Stub(), "c": _Stub()}
    m = AttackModelManager(models, {"a": 1/3, "b": 1/3, "c": 1/3}, hold_batches=8)
    seq = [m.select_and_load()[0] for _ in range(64)]
    # selection can change at most every 8 calls
    changes = sum(1 for i in range(1, 64) if seq[i] != seq[i - 1])
    assert changes <= 7
    for i in range(0, 64, 8):
        assert len(set(seq[i:i + 8])) == 1


def test_hold_batches_default_resamples_every_call():
    import random
    random.seed(1)
    models = {"a": _Stub(), "b": _Stub()}
    m = AttackModelManager(models, {"a": .5, "b": .5})
    seq = [m.select_and_load()[0] for _ in range(60)]
    assert sum(1 for i in range(1, 60) if seq[i] != seq[i - 1]) > 15
