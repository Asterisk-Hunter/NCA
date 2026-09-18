"""Target images and seeds.

Targets are generated procedurally so the whole pipeline runs with no downloaded
assets, and so tests can rely on deterministic inputs. Any RGBA PNG can be used
instead via :func:`load_target`.

Importantly the targets are *premultiplied*: RGB is multiplied by alpha, so
transparent regions carry no colour. Without this the network is asked to
reproduce an arbitrary colour in empty space, which wastes capacity and muddies
the loss.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import ALPHA_CHANNEL, RGBA

SUPERSAMPLE = 4


def _grid(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.linspace(-1.0, 1.0, size)
    return torch.meshgrid(coords, coords, indexing="ij")  # y, x


def _downsample(mask_hi: torch.Tensor) -> torch.Tensor:
    """Average-pool a supersampled boolean mask into an antialiased alpha map."""
    return F.avg_pool2d(mask_hi.to(torch.float32)[None, None], SUPERSAMPLE)[0, 0]


def _polygon(x: torch.Tensor, y: torch.Tensor, verts: list[tuple[float, float]]) -> torch.Tensor:
    """Even-odd ray casting against a polygon."""
    inside = torch.zeros_like(x, dtype=torch.bool)
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        crosses = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
        inside ^= crosses
    return inside


def _angles(n: int) -> list[float]:
    return [math.pi / 2 + i * 2 * math.pi / n for i in range(n)]


def _star_verts(points: int = 5, outer: float = 0.92, inner: float = 0.38) -> list[tuple[float, float]]:
    verts = []
    for i in range(points * 2):
        r = outer if i % 2 == 0 else inner
        # Start at the top and go clockwise so the star is upright.
        theta = math.pi / 2 + i * math.pi / points
        verts.append((r * math.cos(theta), -r * math.sin(theta)))
    return verts


def _solid_mask(shape: str, size: int) -> torch.Tensor:
    """Boolean silhouette for the single-colour shapes, at supersampled resolution."""
    y, x = _grid(size)
    r2 = x * x + y * y

    if shape == "circle":
        return r2 <= 0.81
    if shape == "ring":
        return (r2 <= 0.81) & (r2 >= 0.36)
    if shape == "square":
        return (x.abs() <= 0.8) & (y.abs() <= 0.8)
    if shape == "cross":
        return ((x.abs() <= 0.80) & (y.abs() <= 0.24)) | ((y.abs() <= 0.80) & (x.abs() <= 0.24))
    if shape == "diamond":
        return x.abs() + y.abs() <= 0.92
    if shape == "triangle":
        return _polygon(x, y, [(0.0, -0.85), (0.85, 0.7), (-0.85, 0.7)])
    if shape == "hexagon":
        return _polygon(x, y, [(math.cos(a) * 0.85, math.sin(a) * 0.85) for a in _angles(6)])
    if shape == "star":
        return _polygon(x, y, _star_verts())
    if shape == "heart":
        # Classic implicit heart: (x^2 + y^2 - 1)^3 - x^2 y^3 <= 0. The image
        # y-axis points down, so negate it to keep the heart upright.
        yy = -y
        return (r2 - 1.0) ** 3 - x * x * yy**3 <= 0.0
    raise ValueError(f"unknown shape {shape!r}")


def _face_layers(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Alpha and interior-marks masks for the multi-colour face target.

    The face is the interesting one: it is a single connected organism that has to
    place *internal* structure (eyes and a mouth) at the right coordinates, which
    a plain blob never tests.
    """
    y, x = _grid(size)
    disc = (x * x + y * y) <= 0.81
    eyes = (((x + 0.28) ** 2 + (y + 0.20) ** 2) <= 0.11**2) | (
        ((x - 0.28) ** 2 + (y + 0.20) ** 2) <= 0.11**2
    )
    radius = torch.sqrt(x * x + y * y)
    mouth = (radius >= 0.42) & (radius <= 0.55) & (y > 0.12)
    return disc, eyes | mouth


def _alpha_and_rgb(shape: str, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns antialiased alpha [S, S] and non-premultiplied rgb [3, S, S]."""
    hi = size * SUPERSAMPLE

    if shape == "face":
        disc, marks = _face_layers(hi)
        alpha = _downsample(disc)
        mark_alpha = _downsample(marks)
        base = torch.tensor(COLORS["face"], dtype=torch.float32)[:, None, None]
        mark = torch.tensor(MARK_COLOR, dtype=torch.float32)[:, None, None]
        # Alpha-blend the marks over the base colour.
        rgb = base * (1.0 - mark_alpha[None]) + mark * mark_alpha[None]
        return alpha, rgb

    mask = _solid_mask(shape, hi)
    alpha = _downsample(mask)
    color = torch.tensor(COLORS.get(shape, (1.0, 1.0, 1.0)), dtype=torch.float32)[:, None, None]
    return alpha, color * torch.ones_like(alpha)[None]


SHAPES = (
    "circle",
    "ring",
    "square",
    "cross",
    "diamond",
    "triangle",
    "hexagon",
    "star",
    "heart",
    "face",
)

COLORS: dict[str, tuple[float, float, float]] = {
    "circle": (0.24, 0.64, 0.94),
    "ring": (0.95, 0.55, 0.20),
    "square": (0.42, 0.80, 0.44),
    "cross": (0.90, 0.30, 0.36),
    "diamond": (0.62, 0.44, 0.92),
    "triangle": (0.95, 0.78, 0.24),
    "hexagon": (0.24, 0.80, 0.76),
    "star": (0.98, 0.72, 0.18),
    "heart": (0.92, 0.28, 0.45),
    "face": (0.99, 0.80, 0.22),
}

MARK_COLOR = (0.13, 0.11, 0.10)


def make_target(shape: str, size: int = 40) -> torch.Tensor:
    """Procedurally generate a premultiplied RGBA target [1, 4, size, size] in [0, 1]."""
    if size < 8:
        raise ValueError("size must be at least 8")
    alpha, rgb = _alpha_and_rgb(shape, size)
    return torch.cat([rgb * alpha[None], alpha[None]], dim=0)[None]


def load_target(path: str, size: int = 40) -> torch.Tensor:
    """Load an image file as an RGBA target, resized to size x size."""
    img = Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    img = img.resize((size, size), Image.LANCZOS)
    arr = torch.from_numpy(np.asarray(img, dtype="float32")).permute(2, 0, 1) / 255.0
    # Pillow stores alpha un-premultiplied; premultiply RGB to match make_target.
    rgb = arr[:3] * arr[3:4]
    return torch.cat([rgb, arr[3:4]], dim=0)[None]


def make_seed(size: int, channels: int = 16) -> torch.Tensor:
    """A single live cell at the centre of an empty canvas, shape [1, C, S, S].

    Alpha and every hidden channel are set to 1.0; RGB stays at 0. Growth starts
    from one pixel, so the network has to synthesise the entire pattern rather
    than patch a copy of it back together. Following the paper, the seed is black
    so that it is visible against the white background used for rendering.
    """
    state = torch.zeros(1, channels, size, size)
    state[0, ALPHA_CHANNEL:, size // 2, size // 2] = 1.0
    return state
