"""Synthetic games with analytic or fully known values."""

from .dummy import DummyGame
from .linear_gaussian_scm import LinearGaussianSCM, RandomLinearGaussianSCM
from .random_table import RandomTableGame
from .soum import SOUM, UnanimityGame

__all__ = [
    "SOUM",
    "DummyGame",
    "LinearGaussianSCM",
    "RandomLinearGaussianSCM",
    "RandomTableGame",
    "UnanimityGame",
]
