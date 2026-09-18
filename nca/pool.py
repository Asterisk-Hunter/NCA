"""The sample pool (Experiment 2).

A naive training loop always starts from the seed, so the rule only ever learns
one trajectory: seed -> target. It never learns to *hold* the pattern, and the
resulting dynamics drift, die out, or explode when run past training length.

The pool fixes this. We keep a bank of intermediate states, sample a batch from
it, train on those, and write the results back. Because the bank accumulates
states the model has actually produced, the rule is progressively trained to
recover from its own mistakes -- which both stabilises the target as an attractor
and, as a side effect, produces regeneration.
"""

from __future__ import annotations

import torch


class SamplePool:
    """A fixed-size bank of grid states."""

    def __init__(self, seed: torch.Tensor, size: int = 1024) -> None:
        if size < 1:
            raise ValueError("pool size must be >= 1")
        self.size = size
        # Every slot starts as the same single-cell seed.
        self.states = seed.detach().clone().repeat(size, 1, 1, 1)
        self.cursor = 0

    def __len__(self) -> int:
        return self.size

    @property
    def shape(self) -> torch.Size:
        return self.states.shape

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Random distinct slots, returned with their indices so we can commit back."""
        if batch_size > self.size:
            raise ValueError(f"batch_size {batch_size} exceeds pool size {self.size}")
        idx = torch.randperm(self.size, generator=generator, device=self.states.device)[:batch_size]
        return idx, self.states[idx].clone()

    def commit(self, idx: torch.Tensor, values: torch.Tensor) -> None:
        """Write states back into the slots they were sampled from."""
        self.states[idx] = values.detach().to(self.states.device)

    def reseed(self, idx: torch.Tensor, seed: torch.Tensor) -> None:
        """Force specific slots back to the seed."""
        self.states[idx] = seed.detach().to(self.states.device).expand(len(idx), *seed.shape[1:])

    def stats(self) -> dict[str, float]:
        return {
            "pool_mean": float(self.states.mean()),
            "pool_std": float(self.states.std()),
            "pool_nonzero_frac": float((self.states != 0).float().mean()),
        }
