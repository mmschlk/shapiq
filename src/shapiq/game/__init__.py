"""Definitions of games and related data structures."""

from .arithmetic import AffineGame, ArithmeticGame, MappedGame, ProductGame
from .cast import CastGame
from .finite import BoolFinitePlayerGame, FinitePlayerGame, PackedFinitePlayerGame
from .game import Game
from .kadditive import AdditiveGame, KAdditiveGame
from .reduction import ReductionGame, SumReductionGame
from .table import TableGame

__all__ = [
    "Game",
    "FinitePlayerGame",
    "BoolFinitePlayerGame",
    "PackedFinitePlayerGame",
    "KAdditiveGame",
    "AdditiveGame",
    "ArithmeticGame",
    "CastGame",
    "AffineGame",
    "MappedGame",
    "ProductGame",
    "ReductionGame",
    "SumReductionGame",
    "TableGame",
]
