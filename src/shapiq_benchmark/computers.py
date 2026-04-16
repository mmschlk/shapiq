"""Benchmark ground truth computers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapiq import ExactComputer
from shapiq.typing import IndexType

from .base import GroundTruthComputer

if TYPE_CHECKING:
    from shapiq import Game, InteractionValues


class GameExactComputer(GroundTruthComputer[IndexType]):
    """Use the game's own exact_values implementation."""

    def __init__(self, game: Game) -> None:
        self._game = game

    def exact_values(self, index: IndexType, order: int) -> InteractionValues:
        return self._game.exact_values(index=index, order=order)


class InterventionalComputer(GroundTruthComputer[IndexType]):
    """Exact computer for interventional games using brute force."""

    def __init__(self, game: Game) -> None:
        self._computer = ExactComputer(
            game=game, n_players=game.n_players, evaluate_game=False
        )

    def exact_values(self, index: IndexType, order: int) -> InteractionValues:
        return self._computer(index=index, order=order)


class PathdependentComputer(GroundTruthComputer[IndexType]):
    """Exact computer for tree-based games using the game's implementation."""

    def __init__(self, game: Game) -> None:
        self._delegate = GameExactComputer(game)

    def exact_values(self, index: IndexType, order: int) -> InteractionValues:
        return self._delegate.exact_values(index=index, order=order)


class LocalXAIComputer(GroundTruthComputer[IndexType]):
    """Exact computer for local explanation games using brute force."""

    def __init__(self, game: Game) -> None:
        self._computer = ExactComputer(
            game=game, n_players=game.n_players, evaluate_game=False
        )

    def exact_values(self, index: IndexType, order: int) -> InteractionValues:
        return self._computer(index=index, order=order)
