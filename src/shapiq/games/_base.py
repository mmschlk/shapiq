from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

from shapiq._shape import broadcast_shapes, shape_of
from shapiq.coalitions import (
    CoalitionArray as _CoalitionArray,
    DenseCoalitionArray,
)

if TYPE_CHECKING:
    from jax import Array

    from shapiq._shape import Shape
    from shapiq.coalitions import CoalitionArray


class Game[ValueT](ABC):
    """Base abstraction for cooperative games.

    Estimators are linear in game values, so ``ValueT`` must form a vector
    space over the reals: addition, scalar multiplication, and the centering
    ``v - v(empty)`` must be meaningful. Class-probability vectors, margin
    vectors, and embeddings qualify; anything nonlinear belongs in the link
    function, before predictions become values.
    """

    n_players: int
    target_shape: Shape
    value_shape: Shape = ()

    def __call__(self, coalitions: CoalitionArray | object) -> ValueT:
        """Evaluate values for coalitions; dense mask arrays wrap themselves."""
        if not isinstance(coalitions, _CoalitionArray):
            import jax.numpy as _jnp  # noqa: PLC0415

            coalitions = DenseCoalitionArray(_jnp.asarray(coalitions, dtype=bool))
        self._validate_coalitions(coalitions)
        values = self._call(coalitions)
        self._validate_values(values, coalitions)
        return values

    def __add__(self, other: Game[Array]) -> Game[Array]:
        """Return the game evaluating to the sum of both games' values."""
        if not isinstance(other, Game):
            return NotImplemented
        from shapiq.games._algebra import SumGame  # noqa: PLC0415 - avoids a base/algebra cycle

        return SumGame(((1.0, cast("Game[Array]", self)), (1.0, other)))

    def __sub__(self, other: Game[Array]) -> Game[Array]:
        """Return the residual game ``self - other``."""
        if not isinstance(other, Game):
            return NotImplemented
        from shapiq.games._algebra import SumGame  # noqa: PLC0415 - avoids a base/algebra cycle

        return SumGame(((1.0, cast("Game[Array]", self)), (-1.0, other)))

    def __mul__(self, scale: float) -> Game[Array]:
        """Return the game with all values scaled by ``scale``."""
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            return NotImplemented
        from shapiq.games._algebra import SumGame  # noqa: PLC0415 - avoids a base/algebra cycle

        return SumGame(((float(scale), cast("Game[Array]", self)),))

    __rmul__ = __mul__

    def _validate_coalitions(self, coalitions: CoalitionArray) -> None:
        """Validate coalition compatibility at the game boundary."""
        if coalitions.n_players != self.n_players:
            msg = "coalitions use a different number of players"
            raise ValueError(msg)

    def _validate_values(self, values: ValueT, coalitions: CoalitionArray) -> None:
        """Validate the declared value contract at the game boundary.

        Dense values carry the broadcast of the target shape and the
        coalition array's leading axes first, then the sample axis, then the
        declared value shape.
        """
        if coalitions.shape == ():
            return
        expected = (
            *broadcast_shapes(self.target_shape, coalitions.shape[:-1]),
            coalitions.shape[-1],
            *self.value_shape,
        )
        actual = shape_of(values)
        if actual != expected:
            msg = (
                f"game values have shape {actual}, expected {expected} "
                "(broadcast targets, then samples, then "
                f"value_shape={self.value_shape}); declare value_shape on the "
                "game if it returns vector values"
            )
            raise ValueError(msg)

    @abstractmethod
    def _call(self, coalitions: CoalitionArray) -> ValueT:
        """Evaluate values after base validation."""


class LinkFunction[PredictionT, ValueT](Protocol):
    """Callable that maps model-native predictions to game values."""

    def __call__(self, predictions: PredictionT) -> ValueT:
        """Map predictions to values."""


type Model[ModelInputT, PredictionT] = Callable[[ModelInputT], PredictionT]
