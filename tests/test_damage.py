"""Tests for damage operators.

The important property is that damage erases *every* channel, not just RGB:
clearing alpha is what marks the region dead so the alive mask can act on it.
"""

from __future__ import annotations

import torch

from nca.damage import damage_center_square, damage_circles, damage_half, damage_rect

C, S = 16, 24


def _full(batch: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(batch, C, S, S) + 0.5


def test_n_zero_is_identity():
    x = _full()
    assert torch.equal(damage_circles(x, n=0), x)
    assert torch.equal(damage_rect(x, n=0), x)


def test_shape_preserved():
    x = _full()
    for out in (damage_circles(x, n=3), damage_rect(x, n=2), damage_half(x), damage_center_square(x)):
        assert out.shape == x.shape


def test_circles_erase_every_channel():
    x = _full()
    out = damage_circles(x, n=2)
    erased = (out.abs().sum(dim=1) == 0) & (x.abs().sum(dim=1) > 0)
    assert erased.any(), "expected some cells to be erased"
    # Every erased cell must be zero across all channels, not merely in RGB.
    assert torch.all(out.permute(0, 2, 3, 1)[erased] == 0)


def test_rect_erases_every_channel():
    x = _full()
    out = damage_rect(x, n=2)
    erased = (out.abs().sum(dim=1) == 0) & (x.abs().sum(dim=1) > 0)
    assert erased.any()
    assert torch.all(out.permute(0, 2, 3, 1)[erased] == 0)


def test_circles_are_local_lesions():
    """A single circle must leave most of the pattern intact.

    This is why the paper uses circles for training: the model learns to heal a
    wound rather than to re-grow from nothing.
    """
    x = _full(batch=1)
    out = damage_circles(x, n=1, max_radius_frac=0.2)
    survivors = (out.abs().sum(dim=1) > 0).float().mean()
    assert survivors > 0.5, f"damage destroyed too much: {survivors:.2f}"


def test_damage_is_random_across_calls():
    x = _full(batch=1)
    a, b = damage_circles(x, n=1), damage_circles(x, n=1)
    assert not torch.equal(a, b), "damage should vary between calls"


def test_generator_makes_damage_reproducible():
    x = _full(batch=2)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    assert torch.equal(damage_circles(x, n=2, generator=g1), damage_circles(x, n=2, generator=g2))


def test_half_damage_is_exact():
    x = _full(batch=1)
    left = damage_half(x, "left")
    assert torch.all(left[:, :, :, : S // 2] == 0)
    assert torch.equal(left[:, :, :, S // 2 :], x[:, :, :, S // 2 :])

    top = damage_half(x, "top")
    assert torch.all(top[:, :, : S // 2, :] == 0)
    assert torch.equal(top[:, :, S // 2 :, :], x[:, :, S // 2 :, :])


def test_half_damage_rejects_bad_side():
    try:
        damage_half(_full(), "diagonal")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown side")


def test_center_square_hits_the_middle():
    x = _full(batch=1)
    out = damage_center_square(x, frac=0.5)
    mid = S // 2
    assert torch.all(out[:, :, mid, mid] == 0)
    assert torch.equal(out[:, :, 0, 0], x[:, :, 0, 0])
