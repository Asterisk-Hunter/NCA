"""Evaluation metrics.

Training loss alone is a poor summary of an NCA. Two models can share a loss
while one holds its pattern indefinitely and the other melts after 200 steps. So
we measure three things separately: how well the pattern was grown, whether it
persists, and whether it recovers from damage.
"""

from __future__ import annotations

import torch

from .model import ALPHA_CHANNEL, rgba


def per_sample_mse(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE against the target for each item in the batch, shape [B]."""
    diff = (rgba(x) - target) ** 2
    return diff.flatten(1).mean(dim=1)


def rgb_mae(x: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error over RGB inside the target's opaque region.

    Restricted to the foreground, because that is the part anyone can see; errors
    in empty space are already captured by the alpha term.
    """
    pred, tgt = rgba(x), target
    mask = (tgt[:, ALPHA_CHANNEL : ALPHA_CHANNEL + 1] > 0.5).to(x.dtype)
    denom = mask.sum() * 3
    if denom == 0:
        return float("nan")
    return float(((pred[:, :3] - tgt[:, :3]).abs() * mask).sum() / denom)


def alpha_iou(x: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Intersection-over-union of the predicted and target silhouettes.

    This is the most legible single number for "did it grow the right shape".
    """
    pred = rgba(x)[:, ALPHA_CHANNEL] > threshold
    tgt = target[:, ALPHA_CHANNEL] > threshold
    union = (pred | tgt).sum()
    if union == 0:
        return 1.0
    return float((pred & tgt).sum() / union)


def evaluate(
    model,
    seed: torch.Tensor,
    target: torch.Tensor,
    steps: int = 96,
    persist_steps: int = 0,
    fire_rate: float | None = 0.5,
) -> dict[str, float]:
    """Grow from the seed, then optionally keep running to test stability."""
    model.eval()
    x = seed.clone()
    x = model.rollout(x, steps=steps, fire_rate=fire_rate)[0]

    out = {
        "mse": float(per_sample_mse(x, target).mean()),
        "rgb_mae": rgb_mae(x, target),
        "alpha_iou": alpha_iou(x, target),
        "steps": float(steps),
    }
    if persist_steps > 0:
        y = model.rollout(x, steps=persist_steps, fire_rate=fire_rate)[0]
        out["persist_alpha_iou"] = alpha_iou(y, target)
        out["persist_mse"] = float(per_sample_mse(y, target).mean())
        # Did the pattern survive, or grow without bound / collapse to nothing?
        out["persist_mass_ratio"] = float(
            rgba(y)[:, ALPHA_CHANNEL].sum() / rgba(x)[:, ALPHA_CHANNEL].sum().clamp_min(1e-6)
        )
    model.train()
    return out
