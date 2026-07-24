"""Game arithmetic: extensional transformers built from other games.

Games form a vector space, and the transformers here are its arithmetic:
``v - proxy`` is the residual game a correction estimator consumes,
``v + w`` composes additive structure, ``2.0 * v`` rescales. Every
result is again a game — the closure property the whole design rests on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import numpy as np

from shapiq.coalitions import DenseCoalitionArray
from shapiq.games._base import Game

if TYPE_CHECKING:
    from jax import Array

    from shapiq.coalitions import CoalitionArray


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
