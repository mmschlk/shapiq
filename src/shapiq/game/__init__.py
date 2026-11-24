"""Definitions of games and related data structures."""

from .arithmetic import AffineGame, ArithmeticGame, MappedGame, ProductGame
from .finite import BoolFinitePlayerGame, FinitePlayerGame, PackedFinitePlayerGame
from .game import Game
from .kadditive import AdditiveGame, KAdditiveGame
from .table import TableGame

__all__ = [
    "Game",
    "FinitePlayerGame",
    "BoolFinitePlayerGame",
    "PackedFinitePlayerGame",
    "KAdditiveGame",
    "AdditiveGame",
    "ArithmeticGame",
    "AffineGame",
    "MappedGame",
    "ProductGame",
    "TableGame",
]
