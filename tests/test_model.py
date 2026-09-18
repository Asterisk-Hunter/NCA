"""Invariants of the NCA update rule.

These are the properties that make an NCA an NCA. If any of them breaks, the
model may still train but it is no longer a cellular automaton.
"""

from __future__ import annotations

import math

import torch

from nca.model import ALPHA_CHANNEL, NCA, Perception

C = 16
DEFAULT_PARAMS = 8320


def _randomized(channels: int = C) -> NCA:
    """An NCA whose final layer is *not* zeroed, so updates are non-trivial."""
    torch.manual_seed(0)
    model = NCA(channels=channels)
    with torch.no_grad():
        model.update[-1].weight.normal_(0, 0.1)
    return model


def _live_state(batch: int = 1, size: int = 9, channels: int = C) -> torch.Tensor:
    x = torch.randn(batch, channels, size, size)
    x[:, ALPHA_CHANNEL] = 1.0  # opaque everywhere, so alive masking is a no-op
    return x


def test_parameter_count_matches_paper():
    model = NCA()
    # 48*128 + 128 + 128*16 = 8320, matching "approximately 8000 parameters".
    assert model.num_parameters == DEFAULT_PARAMS, model.num_parameters


def test_output_shape_preserved():
    model = _randomized()
    x = torch.zeros(2, C, 11, 13)
    out = model(x, steps=3)
    assert out.shape == x.shape


def test_initial_update_is_identity():
    """The zeroed final layer must make the untrained model a fixed point."""
    model = NCA()
    x = _live_state()
    out = model.step(x, fire_rate=1.0)
    assert torch.allclose(out, x), "untrained model must not change the state"


def test_locality_is_exactly_3x3():
    """Changing one cell must only affect its immediate 3x3 neighbourhood."""
    model = _randomized()
    size = 9
    x = torch.zeros(1, C, size, size)
    perturbed = x.clone()
    i, j = 4, 4
    perturbed[0, 0, i, j] = 1.0

    d0 = model.update(model.perceive(x))
    d1 = model.update(model.perceive(perturbed))
    changed = (d0 - d1).abs().sum(dim=1)[0] > 0

    expected = torch.zeros(size, size, dtype=torch.bool)
    expected[max(0, i - 1) : i + 2, max(0, j - 1) : j + 2] = True
    assert torch.equal(changed, expected), "receptive field is not exactly 3x3"


def test_fire_rate_zero_freezes_the_grid():
    model = _randomized()
    x = _live_state()
    out = model.step(x, fire_rate=0.0)
    assert torch.allclose(out, x)


def test_fire_rate_one_matches_full_update():
    model = _randomized()
    x = _live_state()
    out = model.step(x, fire_rate=1.0)
    updated = x + model.update(model.perceive(x))
    expected = updated * model.alive_mask(updated)
    assert torch.allclose(out, expected, atol=1e-5)


def test_alive_mask_threshold_and_neighbourhood():
    """A sub-threshold cell with no mature neighbour dies; one next to a mature cell lives."""
    model = NCA()
    x = torch.zeros(1, C, 9, 9)
    x[0, ALPHA_CHANNEL, 2, 2] = 1.0  # mature
    x[0, ALPHA_CHANNEL, 6, 6] = 0.05  # below the 0.1 threshold, isolated

    mask = model.alive_mask(x)[0, 0]
    assert mask[2, 2] == 1.0, "mature cell must be alive"
    assert mask[1, 1] == 1.0 and mask[3, 3] == 1.0, "neighbours of a mature cell must be alive"
    assert mask[6, 6] == 0.0, "sub-threshold isolated cell must be dead"
    assert mask[4, 4] == 0.0, "empty cells must be dead"


def test_alive_masking_zeroes_dead_cells_entirely():
    model = _randomized()
    x = torch.zeros(1, C, 9, 9)
    x[0, ALPHA_CHANNEL, 2, 2] = 1.0
    x[0, 0, 8, 8] = 5.0  # a stray bright pixel far from any mature cell
    out = model.step(x, fire_rate=1.0)
    assert torch.all(out[0, :, 8, 8] == 0), "stray detached pixel must be deleted"


def test_rotation_kernel_math():
    """theta=90deg must swap the sensing axes: Kx=-dy, Ky=dx."""
    base = Perception(C, theta=0.0)
    rot = Perception(C, theta=math.pi / 2)
    _, kx, ky = rot.kernels()
    assert torch.allclose(kx, -base.k_dy, atol=1e-6)
    assert torch.allclose(ky, base.k_dx, atol=1e-6)


def test_rotation_by_2pi_is_identity():
    base = Perception(C, theta=0.0)
    rot = Perception(C, theta=2 * math.pi)
    for a, b in zip(base.kernels(), rot.kernels()):
        assert torch.allclose(a, b, atol=1e-6)


def test_circular_padding_is_translation_equivariant():
    """With circular padding the update commutes with translation.

    This is the cleanest statement of what "local rule" buys you: no cell knows
    where it is, so shifting the input shifts the output identically.
    """
    torch.manual_seed(1)
    model = NCA(pad_mode="circular", theta=0.0)
    base = NCA(pad_mode="circular", theta=0.0)
    with torch.no_grad():
        model.update[-1].weight.normal_(0, 0.1)
        base.update[-1].weight.normal_(0, 0.1)
        model.update.load_state_dict(base.update.state_dict())

    x = torch.randn(1, C, 12, 12)
    shifts = (3, 5)
    shifted = torch.roll(x, shifts=shifts, dims=(2, 3))

    d0 = model.update(model.perceive(x))
    d1 = model.update(model.perceive(shifted))
    assert torch.allclose(d1, torch.roll(d0, shifts=shifts, dims=(2, 3)), atol=1e-5)


def test_gradients_reach_all_parameters():
    model = _randomized()
    out = model(_live_state(batch=2, size=9), steps=2)
    out.sum().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads), "every parameter must receive a gradient"
    assert any(float(g.abs().sum()) > 0 for g in grads), "gradients must be non-zero"


def test_perception_channels():
    model = NCA(channels=C)
    x = torch.zeros(1, C, 7, 7)
    assert model.perceive(x).shape[1] == 3 * C


def test_rollout_records_requested_frames():
    model = NCA()
    _, frames = model.rollout(_live_state(), steps=10, fire_rate=1.0, record_every=5)
    assert len(frames) == 2
    assert all(not f.requires_grad for f in frames)


def test_rejects_too_few_channels():
    try:
        NCA(channels=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError for channels < 4")
