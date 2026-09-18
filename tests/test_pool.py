"""Tests for the sample pool."""

from __future__ import annotations

import torch

from nca.pool import SamplePool
from nca.targets import make_seed

C, S = 16, 12


def _seed() -> torch.Tensor:
    return make_seed(S, channels=C)


def test_pool_starts_full_of_seeds():
    pool = SamplePool(_seed(), size=16)
    assert len(pool) == 16
    assert torch.equal(pool.states[0], pool.states[15])


def test_sample_returns_distinct_indices():
    pool = SamplePool(_seed(), size=32)
    idx, batch = pool.sample(8)
    assert batch.shape == (8, C, S, S)
    assert len(set(idx.tolist())) == 8, "sampled slots must be distinct"


def test_commit_writes_back_to_the_sampled_slots():
    pool = SamplePool(_seed(), size=16)
    idx, _ = pool.sample(4)
    values = torch.full((4, C, S, S), 0.25)
    pool.commit(idx, values)
    for slot in idx.tolist():
        assert torch.allclose(pool.states[slot], values[0])


def test_commit_detaches_from_the_graph():
    """Pool writes must not keep autograd graphs alive, or memory grows unbounded."""
    pool = SamplePool(_seed(), size=8)
    idx, batch = pool.sample(2)
    values = batch * 2.0
    assert values.requires_grad is False
    pool.commit(idx, values)
    assert pool.states.grad_fn is None


def test_reseed_forces_slots_back_to_the_seed():
    pool = SamplePool(_seed(), size=8)
    idx = torch.tensor([1, 3, 5])
    pool.states[:] = 9.0
    pool.reseed(idx, _seed())
    assert torch.equal(pool.states[1], _seed()[0])
    assert float(pool.states[0].mean()) == 9.0


def test_batch_larger_than_pool_is_rejected():
    pool = SamplePool(_seed(), size=4)
    try:
        pool.sample(5)
    except ValueError:
        return
    raise AssertionError("expected ValueError when batch exceeds pool size")


def test_invalid_pool_size_is_rejected():
    try:
        SamplePool(_seed(), size=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for pool size 0")


def test_stats_report_something_useful():
    pool = SamplePool(_seed(), size=8)
    stats = pool.stats()
    assert set(stats) == {"pool_mean", "pool_std", "pool_nonzero_frac"}
    # Mostly empty at the start: one live cell per state.
    assert stats["pool_nonzero_frac"] < 0.1
