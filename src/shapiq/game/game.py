# Copyright (c) 2026 shapiq contributors
"""Base game class for shapiq."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ndlike.jax_like import JaxLikeImplementation

from shapiq.value import NumericValue, Value

if TYPE_CHECKING:
    from shapiq.coalition import Coalition
    from shapiq.players import Players


class Game[C: Coalition, V: Value](JaxLikeImplementation[Any], ABC):
    """An abstract JAX-like batch of cooperative games.

    ``shape`` describes the game batch; a scalar game has shape ``()``. Indexing
    and array operations act on these batch axes, preserving each game's
    players and evaluation semantics. Calling a game evaluates coalitions.

    ``players`` is a ``Players[C]`` batch of player sets. Its logical shape
    excludes each set's internal player structure and must broadcast to
    ``shape`` without enlarging it. For example, games of shape ``(2, 3)`` may
    share a scalar player set or use player sets of shape ``(3,)`` or
    ``(2, 1)``.

    The player sets declare the compatible coalition representations ``C`` and
    validate their admissibility. Subclasses provide the representation-specific
    JAX-like hooks, player sets, and evaluation.
    """

    @property
    @abstractmethod
    def players(self) -> Players[C]:
        """Return player sets whose logical shape broadcasts to ``shape``."""

    def is_valid(self, coalition: object) -> bool:
        """Check coalition admissibility with the players and game batch shape."""
        return self.players.is_valid(coalition, batch_shape=self.shape)

    def __contains__(self, coalition: object) -> bool:
        """Check if a coalition is valid in the game."""
        return self.is_valid(coalition)

    def __call__(self, coalition: C) -> V:
        """Evaluate coalitions through the game's default reduction."""
        return self.eval(coalition)

    @abstractmethod
    def eval(self, coalition: C) -> V:
        """Evaluate coalitions, broadcasting their batch shape with ``shape``.

        The returned values have the broadcasted logical batch shape, excluding
        any internal value dimensions. Implementations accept the coalition
        representations declared by ``players`` and use ``is_valid`` to check
        admissibility at the evaluation boundary.
        """
        msg = "Game evaluation not implemented."
        raise NotImplementedError(msg)


type NumericGame[C: Coalition, V: NumericValue] = Game[C, V]
