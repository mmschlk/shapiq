"""Synthetic games with analytic or fully known values."""

from .dummy import DummyGame
from .linear_gaussian import LinearGaussianGame, LinearGaussianSCM, RandomLinearGaussianGame
from .random_table import RandomTableGame
from .soum import SOUM, UnanimityGame

__all__ = [
    "SOUM",
    "DummyGame",
    "LinearGaussianGame",
    "LinearGaussianSCM",
    "RandomLinearGaussianGame",
    "RandomTableGame",
    "UnanimityGame",
]
