"""Game objects for the shapiq package."""

from . import benchmark, imputer
from .base import Game

__all__ = ["Game"] + imputer.__all__ + benchmark.__all__

# Path: shapiq/games/__init__.py
