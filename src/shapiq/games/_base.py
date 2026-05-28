from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from shapiq._shape import Shape
    from shapiq.coalitions import CoalitionArray


class Game[ValueT](ABC):
    """Base abstraction for cooperative games."""

    n_players: int
    target_shape: Shape

    def __call__(self, coalitions: CoalitionArray) -> ValueT:
        """Evaluate values for coalitions."""
        self._validate_coalitions(coalitions)
        return self._call(coalitions)

    def _validate_coalitions(self, coalitions: CoalitionArray) -> None:
        """Validate coalition compatibility at the game boundary."""
        if coalitions.n_players != self.n_players:
            msg = "coalitions use a different number of players"
            raise ValueError(msg)

    @abstractmethod
    def _call(self, coalitions: CoalitionArray) -> ValueT:
        """Evaluate values after base validation."""


class LinkFunction[PredictionT, ValueT](Protocol):
    """Callable that maps model-native predictions to game values."""

    def __call__(self, predictions: PredictionT) -> ValueT:
        """Map predictions to values."""


type Model[ModelInputT, PredictionT] = Callable[[ModelInputT], PredictionT]
