"""Definitions of games and related data structures."""

from .finite import BoolFinitePlayerGame, FinitePlayerGame, PackedFinitePlayerGame
from .game import Game
from .protocols import Additive, KAdditive
from .table import TableGame

__all__ = [
    "Game",
    "FinitePlayerGame",
    "BoolFinitePlayerGame",
    "PackedFinitePlayerGame",
    "TableGame",
    "KAdditive",
    "Additive",
]
