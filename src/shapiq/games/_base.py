from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

import jax.numpy as jnp
import numpy as np

from shapiq._frozen import Frozen
from shapiq._shape import broadcast_shapes, shape_of
from shapiq.coalitions import (
    CoalitionArray as _CoalitionArray,
    DenseCoalitionArray,
)

if TYPE_CHECKING:
    from jax import Array

    from shapiq._shape import Shape
    from shapiq.coalitions import CoalitionArray

type GameValues = Array | np.ndarray
"""The value currencies games speak: device arrays on the evaluation
plane, host arrays on the exact plane. Verbs that accept any game —
pulling values to host float64 themselves — take ``Game[GameValues]``."""


class Game[ValueT](Frozen, ABC):
    """Base abstraction for cooperative games.

    Estimators are linear in game values, so ``ValueT`` must form a vector
    space over the reals: addition, scalar multiplication, and the centering
    ``v - v(empty)`` must be meaningful. Class-probability vectors, margin
    vectors, and embeddings qualify; anything nonlinear belongs in the link
    function, before predictions become values.

    Games are configuration and freeze after construction: a game's numeric
    identity never changes underneath the estimates derived from it.
    """

    n_players: int
    target_shape: Shape
    value_shape: Shape = ()

    def __call__(self, coalitions: CoalitionArray | object) -> ValueT:
        """Evaluate values for coalitions; dense mask arrays wrap themselves."""
        if not isinstance(coalitions, _CoalitionArray):
            coalitions = DenseCoalitionArray(jnp.asarray(coalitions, dtype=bool))
        self._validate_coalitions(coalitions)
        values = self._call(coalitions)
        self._validate_values(values, coalitions)
        return values

    def __add__(self, other: Game[Array]) -> Game[Array]:
        """Return the game evaluating to the sum of both games' values."""
        if not isinstance(other, Game):
            return NotImplemented
        return SumGame(((1.0, cast("Game[Array]", self)), (1.0, other)))

    def __sub__(self, other: Game[Array]) -> Game[Array]:
        """Return the residual game ``self - other``."""
        if not isinstance(other, Game):
            return NotImplemented
        return SumGame(((1.0, cast("Game[Array]", self)), (-1.0, other)))

    def __mul__(self, scale: float) -> Game[Array]:
        """Return the game with all values scaled by ``scale``."""
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            return NotImplemented
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


# Games form a vector space, and SumGame is its arithmetic: ``v - proxy``
# is the residual game a correction estimator consumes, ``v + w`` composes
# additive structure, ``2.0 * v`` rescales. Every result is again a game —
# the closure property the design rests on, which is also why the algebra
# lives with the base class: Game's operators construct it by name.

class SumGame(Game["Array"]):
    """A weighted sum of games, evaluated part by part."""

    parts: tuple[tuple[float, Game[Array]], ...]

    def __init__(self, parts: tuple[tuple[float, Game[Array]], ...]) -> None:
        """Initialize from ``(scale, game)`` parts; nested sums flatten.

        Args:
            parts: The weighted games to sum, at least one.

        Raises:
            ValueError: If no parts are given or the parts disagree on
                players, target shape, or value shape.
        """
        flattened: list[tuple[float, Game[Array]]] = []
        for scale, game in parts:
            if isinstance(game, SumGame):
                flattened.extend(
                    (scale * inner_scale, inner) for inner_scale, inner in game.parts
                )
            else:
                flattened.append((float(scale), game))
        if not flattened:
            msg = "a sum of games needs at least one part"
            raise ValueError(msg)
        first = flattened[0][1]
        for _, game in flattened[1:]:
            if game.n_players != first.n_players:
                msg = "cannot combine games over different numbers of players"
                raise ValueError(msg)
            if game.target_shape != first.target_shape or game.value_shape != first.value_shape:
                msg = "cannot combine games with different target or value shapes"
                raise ValueError(msg)
        self.parts = tuple(flattened)
        self.n_players = first.n_players
        self.target_shape = first.target_shape
        self.value_shape = first.value_shape

    def _call(self, coalitions: CoalitionArray) -> Array:
        """Evaluate every part and combine."""
        total: Array | None = None
        for scale, game in self.parts:
            values = scale * jnp.asarray(game(coalitions))
            total = values if total is None else total + values
        return cast("Array", total)  # parts is never empty

    def _host_values(self, masks: np.ndarray) -> np.ndarray:
        """Evaluate at host float64 where every part can (exact-op path).

        Parts without a host path go through the game boundary at stack
        precision; the sum is then only as exact as its least exact part.
        """
        total = np.zeros(masks.shape[:-1], dtype=np.float64)
        for scale, game in self.parts:
            host = getattr(game, "_host_values", None)
            if host is not None:
                values = host(masks)
            else:
                values = np.asarray(
                    game(DenseCoalitionArray(jnp.asarray(masks))),
                    dtype=np.float64,
                )
            total = total + scale * values
        return total

    def __repr__(self) -> str:
        """Return a concise representation."""
        return f"{type(self).__name__}(n_parts={len(self.parts)!r}, n_players={self.n_players!r})"
