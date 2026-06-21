"""Benchmark Implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from shapiq.typing import IndexType

if TYPE_CHECKING:
    from shapiq import Game, InteractionValues

    from .computers import GroundTruthComputer


class Benchmark[T_Index: IndexType](ABC):
    """Protocol for benchmark implementations."""

    @property
    @abstractmethod
    def game(self) -> Game:
        """Return the game associated with the benchmark."""
        ...

    @property
    @abstractmethod
    def computer(self) -> GroundTruthComputer[T_Index]:
        """Return the ground truth computer used by the benchmark."""
        ...

    @abstractmethod
    def exact_values(self, index: T_Index, order: int, **kwargs: object) -> InteractionValues:
        """Compute exact interaction values for the given index and order."""
        ...
