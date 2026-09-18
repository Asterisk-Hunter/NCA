"""Tests for procedural targets and the seed state."""

from __future__ import annotations

import torch

from nca.model import ALPHA_CHANNEL, RGBA
from nca.targets import SHAPES, make_seed, make_target

SIZE = 32


def test_all_shapes_render():
    for shape in SHAPES:
        t = make_target(shape, size=SIZE)
        assert t.shape == (1, RGBA, SIZE, SIZE), shape
        assert t.min() >= 0.0 and t.max() <= 1.0, shape
        # Every shape should occupy a sensible fraction of the canvas.
        coverage = float(t[0, ALPHA_CHANNEL].mean())
        assert 0.05 < coverage < 0.95, f"{shape} coverage {coverage:.3f}"


def test_target_is_deterministic():
    assert torch.equal(make_target("star", size=SIZE), make_target("star", size=SIZE))


def test_alpha_spans_background_and_foreground():
    t = make_target("circle", size=SIZE)
    alpha = t[0, ALPHA_CHANNEL]
    assert float(alpha.max()) > 0.9, "foreground should be opaque"
    assert float(alpha.min()) < 0.1, "background should be transparent"


def test_rgb_is_premultiplied_by_alpha():
    """Premultiplication means transparent regions carry no colour information.

    Without it the network is asked to reproduce an arbitrary colour in empty
    space, which wastes capacity and muddies the loss.
    """
    t = make_target("heart", size=SIZE)
    alpha = t[0, ALPHA_CHANNEL]
    assert torch.all(t[0, :3] <= alpha.unsqueeze(0) + 1e-6)
    assert torch.allclose(t[0, :3][:, alpha < 1e-6], torch.zeros_like(t[0, :3][:, alpha < 1e-6]))


def test_antialiasing_produces_intermediate_alphas():
    t = make_target("circle", size=SIZE)
    alpha = t[0, ALPHA_CHANNEL]
    partial = ((alpha > 0.01) & (alpha < 0.99)).sum()
    assert partial > 0, "expected antialiased edge pixels"


def test_seed_is_a_single_cell():
    for channels in (16, 12):
        seed = make_seed(SIZE, channels=channels)
        assert seed.shape == (1, channels, SIZE, SIZE)
        assert float(seed[0, :3].abs().sum()) == 0.0, "seed RGB must be black"
        # Alpha and hidden channels are all 1.0 at exactly one location.
        live = (seed != 0).any(dim=1)[0]
        assert int(live.sum()) == 1, "seed must be exactly one cell"
        i, j = (int(v) for v in live.nonzero()[0])
        assert (i, j) == (SIZE // 2, SIZE // 2)
        assert float(seed[0, ALPHA_CHANNEL, i, j]) == 1.0
        assert float(seed[0, RGBA:, i, j].min()) == 1.0


def test_rejects_tiny_size():
    try:
        make_target("circle", size=4)
    except ValueError:
        return
    raise AssertionError("expected ValueError for size < 8")


def test_rejects_unknown_shape():
    try:
        make_target("dodecahedron", size=SIZE)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown shape")
