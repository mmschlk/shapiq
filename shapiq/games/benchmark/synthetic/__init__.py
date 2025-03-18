"""This module contains synthetic games for benchmarking purposes."""

from .dummy import DummyGame
from .random import RandomGame
from .soum import SOUM, UnanimityGame

__all__ = ["DummyGame", "SOUM", "UnanimityGame", "RandomGame"]

# Path: shapiq/games/benchmark/synthetic/__init__.py
