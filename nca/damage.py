"""Damage operators.

Damage serves two distinct roles:

* **In training** (Experiment 3) we erase parts of pool states before each
  rollout. The rule then has to learn to regenerate, not just grow, which widens
  the basin of attraction around the target pattern.
* **In evaluation** we apply damage types the model was never trained on -- halves
  and squares rather than circles -- to test whether regeneration *generalises*
  beyond the training distribution. That generalisation is the interesting claim.

All operators zero the erased region across every channel, which sets alpha to 0
and therefore marks those cells dead for the alive-masking step.
"""

from __future__ import annotations

import torch

# Damage geometry, as fractions of the grid size.
DEFAULT_MAX_RADIUS_FRAC = 0.23


def damage_circles(
    x: torch.Tensor,
    n: int = 3,
    max_radius_frac: float = DEFAULT_MAX_RADIUS_FRAC,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Zero out ``n`` random circular regions per sample -- the training damage.

    The paper erases "a random circular region within the pattern". Circles are a
    gentler, more *biological* lesion than rectangles: they remove a blob without
    destroying long-range structure, so the model can learn to heal rather than
    merely re-grow.

    Fully vectorised over the batch: one mask is built per circle with broadcast
    comparisons instead of a Python loop over samples.
    """
    if n <= 0:
        return x
    out = x.clone()
    b, _, h, w = out.shape
    max_r = max(1, int(max_radius_frac * min(h, w)))

    ys = torch.arange(h, device=x.device, dtype=x.dtype)[None, :, None]  # [1,H,1]
    xs = torch.arange(w, device=x.device, dtype=x.dtype)[None, None, :]  # [1,1,W]

    for _ in range(n):
        cy = torch.randint(0, h, (b, 1, 1), device=x.device, generator=generator).to(x.dtype)
        cx = torch.randint(0, w, (b, 1, 1), device=x.device, generator=generator).to(x.dtype)
        r = torch.randint(1, max_r + 1, (b, 1, 1), device=x.device, generator=generator).to(x.dtype)
        inside = ((ys - cy) ** 2 + (xs - cx) ** 2) <= r**2  # [B,H,W]
        out = out * (~inside).unsqueeze(1)

    return out


def damage_rect(
    x: torch.Tensor,
    n: int = 3,
    max_size_frac: float = 0.3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Zero out ``n`` random axis-aligned rectangles per sample."""
    if n <= 0:
        return x
    out = x.clone()
    b, _, h, w = out.shape
    max_h = max(1, int(max_size_frac * h))
    max_w = max(1, int(max_size_frac * w))

    rows = torch.arange(h, device=x.device)[None, :, None]
    cols = torch.arange(w, device=x.device)[None, None, :]

    for _ in range(n):
        ph = torch.randint(1, max_h + 1, (b, 1, 1), device=x.device, generator=generator)
        pw = torch.randint(1, max_w + 1, (b, 1, 1), device=x.device, generator=generator)
        top = (torch.rand((b, 1, 1), device=x.device, generator=generator) * (h - ph + 1)).long()
        left = (torch.rand((b, 1, 1), device=x.device, generator=generator) * (w - pw + 1)).long()
        in_rows = (rows >= top) & (rows < top + ph)
        in_cols = (cols >= left) & (cols < left + pw)
        inside = in_rows & in_cols  # [B,H,W]
        out = out * (~inside).unsqueeze(1)

    return out


def damage_half(x: torch.Tensor, side: str = "left") -> torch.Tensor:
    """Erase one half of the grid. Never seen in training -- a generalisation test."""
    b, _, h, w = x.shape
    out = x.clone()
    if side in ("left", "right"):
        cut = w // 2
        if side == "left":
            out[:, :, :, :cut] = 0.0
        else:
            out[:, :, :, cut:] = 0.0
    elif side in ("top", "bottom"):
        cut = h // 2
        if side == "top":
            out[:, :, :cut, :] = 0.0
        else:
            out[:, :, cut:, :] = 0.0
    else:
        raise ValueError(f"unknown side {side!r}")
    return out


def damage_center_square(x: torch.Tensor, frac: float = 0.5) -> torch.Tensor:
    """Punch a square hole out of the middle. Never seen in training."""
    _, _, h, w = x.shape
    sh, sw = max(1, int(frac * h)), max(1, int(frac * w))
    top, left = (h - sh) // 2, (w - sw) // 2
    out = x.clone()
    out[:, :, top : top + sh, left : left + sw] = 0.0
    return out


def damage_seed_only(seed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Replace every live cell with the seed: a total restart."""
    return seed.expand_as(x).clone()
