"""This module contains synthetic games for benchmarking purposes."""

from .dummy import DummyGame
from .linear_gaussian_scm import ConfoundedChainSCM, LinearGaussianSCM
from .random_game import RandomGame
from .soum import SOUM, UnanimityGame

__all__ = [
    "DummyGame",
    "SOUM",
    "UnanimityGame",
    "RandomGame",
    "LinearGaussianSCM",
    "ConfoundedChainSCM",
]
