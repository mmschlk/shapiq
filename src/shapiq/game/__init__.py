# Copyright (c) 2026 shapiq contributors
"""Definitions of games and related data structures."""

from .game import Game
from .partial import LiftedGame, PartialShape, PGame
from .protocols import Additive, KAdditive

__all__ = ["Game", "PGame", "LiftedGame", "PartialShape", "KAdditive", "Additive"]
