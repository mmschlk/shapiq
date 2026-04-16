"""Benchmark Implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from shapiq import ExactComputer
from shapiq.typing import IndexType

if TYPE_CHECKING:
    from shapiq import Game, InteractionValues


T_Index_contra = TypeVar("T_Index_contra", bound=IndexType, contravariant=True)


@runtime_checkable
class GroundTruthComputer(Protocol[T_Index_contra]):
    """A protocol for ground truth computers that compute exact interaction values.

    This protocol defines the interface for any ground truth computer that can compute exact
    interaction values for a given game and index type.
    """

    def exact_values(self, index: T_Index_contra, order: int) -> InteractionValues:
        """Compute the exact interaction values for a given index and order.

        Args:
            index: The index type for which to compute the interaction values.
            order: The order of interactions to compute.

        Returns:
            InteractionValues: The computed interaction values for the specified index and order.
        """
        ...


class BruteForceComputer(GroundTruthComputer[IndexType]):
    """A brute force computer for exact computation of interaction values."""

    def __init__(self, game: Game) -> None:
        """Initialize a BruteForceComputer instance."""
        self.game = game
        self._computer = ExactComputer(
            game=game, n_players=game.n_players, evaluate_game=False
        )

    def exact_values(self, index: IndexType, order: int) -> InteractionValues:
        """Compute the exact values using brute force."""
        return self._computer(index=index, order=order)


@runtime_checkable
class Benchmark(Protocol[T_Index_contra]):
    """Protocol for benchmark implementations."""

    @property
    def game(self) -> Game:
        """Return the game associated with the benchmark."""
        ...

    @property
    def computer(self) -> GroundTruthComputer[T_Index_contra]:
        """Return the ground truth computer used by the benchmark."""
        ...

    def exact_values(self, index: T_Index_contra, order: int) -> InteractionValues:
        """Compute exact interaction values for the given index and order."""
        ...
