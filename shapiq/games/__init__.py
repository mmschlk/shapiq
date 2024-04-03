"""This module contains sample game functions for the shapiq package."""

from .base import Game
from .dummy import DummyGame
from .imputer import MarginalImputer

__all__ = [
    "DummyGame",
    "Game",
    "MarginalImputer",
]
