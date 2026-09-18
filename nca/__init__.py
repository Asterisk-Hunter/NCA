"""Neural Cellular Automata -- growing and regenerating patterns with a learned local rule."""

from .model import ALPHA_CHANNEL, RGBA, NCA, Perception, nca_loss, rgba
from .targets import SHAPES, load_target, make_seed, make_target

__version__ = "0.1.0"

__all__ = [
    "ALPHA_CHANNEL",
    "RGBA",
    "NCA",
    "Perception",
    "SHAPES",
    "load_target",
    "make_seed",
    "make_target",
    "nca_loss",
    "rgba",
    "__version__",
]
