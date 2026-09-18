"""Neural Cellular Automata: a learned, local, iterative update rule.

Each cell holds a vector of channels (the first four are visible RGBA, the rest
are hidden "genome" channels). One shared network is applied to every cell using
only a 3x3 neighbourhood, and the grid is iterated many times. A small local rule
iterated long enough produces global structure -- and because the update depends
only on nearby cells, that structure self-repairs when damaged.

Reference: Mordvintsev, Randazzo, Niklasson & Levin, "Growing Neural Cellular
Automata", Distill, 2020. https://distill.pub/2020/growing-ca/
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel layout: [0:3] = RGB, [3] = alpha, [4:] = hidden state.
RGBA = 4
ALPHA_CHANNEL = 3

# A cell is "mature" if its own alpha exceeds this. Cells with no mature cell in
# their 3x3 neighbourhood are dead and get zeroed every step.
ALIVE_THRESHOLD = 0.1

# Fixed (non-learned) perception kernels. A cell sees its own state plus the
# gradient of every channel along x and y.
_IDENTITY = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
_SOBEL_X = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
_SOBEL_Y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]


class Perception(nn.Module):
    """Depthwise 3x3 filters (identity, dx, dy) producing 3x the channels.

    These weights are fixed rather than learned: giving the update network
    explicit access to local gradients is what lets a cell detect the edge of its
    own body. The Sobel kernels are divided by 8 so the gradient channels are on
    the same scale as the identity channel, which keeps optimisation well
    conditioned. The original paper's pseudocode omits that normalisation but its
    reference implementation includes it.

    ``theta`` rotates the sensing axes (Experiment 4). A model trained at
    theta=0 grows rotated copies of its pattern at other angles with no
    retraining, because the update rule never sees the absolute axes, only the
    gradients along them.
    """

    def __init__(self, channels: int, pad_mode: str = "zeros", theta: float = 0.0) -> None:
        super().__init__()
        if channels < RGBA:
            raise ValueError(f"need at least {RGBA} channels (RGBA), got {channels}")
        self.channels = channels
        self.pad_mode = pad_mode
        self.theta = float(theta)

        bases = {
            "k_identity": torch.tensor(_IDENTITY, dtype=torch.float32),
            "k_dx": torch.tensor(_SOBEL_X, dtype=torch.float32) / 8.0,
            "k_dy": torch.tensor(_SOBEL_Y, dtype=torch.float32) / 8.0,
        }
        for name, kernel in bases.items():
            # [C, 1, 3, 3] for a depthwise conv over the C state channels.
            # Non-persistent: constant, so it does not belong in the state dict.
            self.register_buffer(name, kernel.repeat(channels, 1, 1).unsqueeze(1), persistent=False)

    def kernels(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.theta == 0.0:
            return self.k_identity, self.k_dx, self.k_dy
        cos, sin = math.cos(self.theta), math.sin(self.theta)
        # [Kx; Ky] = [[cos, -sin], [sin, cos]] @ [Sobel_x; Sobel_y]
        kx = cos * self.k_dx - sin * self.k_dy
        ky = sin * self.k_dx + cos * self.k_dy
        return self.k_identity, kx, ky

    def _conv(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        if self.pad_mode == "zeros":
            return F.conv2d(x, kernel, padding=1, groups=self.channels)
        x = F.pad(x, (1, 1, 1, 1), mode=self.pad_mode)
        return F.conv2d(x, kernel, groups=self.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity, kx, ky = self.kernels()
        return torch.cat(
            [self._conv(x, identity), self._conv(x, kx), self._conv(x, ky)],
            dim=1,
        )


class NCA(nn.Module):
    """A single shared update rule applied to every cell, repeatedly.

    Despite the recurrent appearance this is a tiny model: with the default
    settings it has 8,320 parameters, all of them in the 1x1 convolutions.
    """

    def __init__(
        self,
        channels: int = 16,
        hidden: int = 128,
        fire_rate: float = 0.5,
        alive_masking: bool = True,
        pad_mode: str = "zeros",
        theta: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.fire_rate = fire_rate
        self.alive_masking = alive_masking

        self.perceive = Perception(channels, pad_mode=pad_mode, theta=theta)
        # 1x1 convolutions are exactly an MLP applied independently at each cell.
        self.update = nn.Sequential(
            nn.Conv2d(3 * channels, hidden, kernel_size=1),
            nn.ReLU(),
            # No ReLU on the output: increments must be able to be negative.
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        # Zero the final layer so the initial update is exactly zero. The grid then
        # starts at a fixed point and training begins as a small perturbation
        # around it; without this the random initial delta blows the state up
        # immediately and learning never starts.
        with torch.no_grad():
            self.update[-1].weight.zero_()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def alive_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Cells with no opaque cell in their 3x3 neighbourhood are dead.

        Multiplying by this mask deletes stray pixels that drift away from the
        body, which is what keeps the grown pattern a single connected organism
        instead of a scattering of debris.
        """
        alpha = x[:, ALPHA_CHANNEL : ALPHA_CHANNEL + 1]
        mature = F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1)
        return (mature > ALIVE_THRESHOLD).to(x.dtype)

    def step(self, x: torch.Tensor, fire_rate: float | None = None) -> torch.Tensor:
        """One asynchronous update of every cell."""
        rate = self.fire_rate if fire_rate is None else fire_rate
        delta = self.update(self.perceive(x))
        if rate is None:
            mask = 1.0
        else:
            # Independent per-cell coin flip. Real cells do not share a clock, and
            # asynchrony stops the whole grid oscillating in lockstep.
            shape = (x.shape[0], 1, x.shape[2], x.shape[3])
            mask = (torch.rand(shape, device=x.device, dtype=x.dtype) <= rate).to(x.dtype)
        x = x + delta * mask
        if self.alive_masking:
            x = x * self.alive_mask(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        steps: int = 1,
        fire_rate: float | None = None,
        record: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        frames: list[torch.Tensor] = []
        for _ in range(steps):
            x = self.step(x, fire_rate)
            if record:
                frames.append(x)
        if record:
            return x, frames
        return x

    @torch.no_grad()
    def rollout(
        self,
        x: torch.Tensor,
        steps: int = 1,
        fire_rate: float | None = None,
        record_every: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Iterate without building a graph, optionally recording frames."""
        frames: list[torch.Tensor] = []
        for i in range(steps):
            x = self.step(x, fire_rate)
            if record_every and (i + 1) % record_every == 0:
                frames.append(x.detach())
        return x, frames


def rgba(x: torch.Tensor) -> torch.Tensor:
    """Visible channels, clamped to a displayable range."""
    return x[:, :RGBA].clamp(0.0, 1.0)


def nca_loss(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pixel-wise L2 (mean squared error) over the visible RGBA channels.

    Written as explicit arithmetic rather than ``F.mse_loss`` because the target
    carries a batch dimension of 1 and is broadcast across the batch; ``mse_loss``
    warns about that shape mismatch even though broadcasting is intended here.
    """
    diff = rgba(x) - target
    return (diff * diff).mean()
