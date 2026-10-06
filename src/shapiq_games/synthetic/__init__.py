"""Synthetic games with analytic or fully known values."""

from .dummy import DummyGame
from .random_table import RandomTableGame
from .soum import SOUM, UnanimityGame

__all__ = ["SOUM", "DummyGame", "RandomTableGame", "UnanimityGame"]
