"""Benchmark Implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from shapiq.typing import IndexType

if TYPE_CHECKING:
    from shapiq import Game, InteractionValues

    from .computers import GroundTruthComputer


T_Index_contra = TypeVar("T_Index_contra", bound=IndexType, contravariant=True)


class Benchmark(ABC, Generic[T_Index_contra]):
    """Protocol for benchmark implementations."""

    @property
    @abstractmethod
    def game(self) -> Game:
        """Return the game associated with the benchmark."""
        ...

    @property
    @abstractmethod
    def computer(self) -> GroundTruthComputer[T_Index_contra]:
        """Return the ground truth computer used by the benchmark."""
        ...

    @abstractmethod
    def exact_values(
        self, index: T_Index_contra, order: int, budget: int | None = None
    ) -> InteractionValues:
        """Compute exact interaction values for the given index and order."""
        ...
