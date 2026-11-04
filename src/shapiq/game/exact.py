"""Exact computation of interaction indices from a cached table of every coalition value."""

from dataclasses import dataclass
from typing import Any, ClassVar, Self, cast

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Float, Shaped
from ndlike.jax_functions import jax_expand_dims
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.game.discretederivative import WeightedGameValue, subset_patterns
from shapiq.game.finite import FinitePlayerGame
from shapiq.game.game import Game
from shapiq.game.table import TableGame
from shapiq.game.weights import DiscreteDerivativeWeight, Weight
from shapiq.index import Index
from shapiq.typing import typechecked


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class ExactComputer(JaxAxisProtected[jax.Array]):
    """Exact interaction indices of a game batch, evaluating each coalition only once.

    ``from_game`` stores every coalition value as a ``TableGame``; calling the computer with
    an index reuses that table, so several indices cost one round of game evaluations. The
    batch is the game batch. Memory grows with ``2**n_players`` per game.
    """

    table: TableGame
    protected_axes: ClassVar[dict[str, int]] = {"table": 0}

    @classmethod
    def from_game(cls, game: Shaped[FinitePlayerGame, "#*batch"]) -> Self:
        """Evaluate ``game`` on every coalition and cache the values."""
        n_players = game.n_players
        # subset_patterns lists coalitions in TableGame order: row i has the bits of i.
        coalitions = subset_patterns(n_players).reshape(
            2**n_players, *(1,) * game.ndim, n_players
        )  # (2**n_players, *ones, n_players)
        values = jnp.moveaxis(game(coalitions), 0, -1)  # (*batch, 2**n_players)
        return cls(table=TableGame(values, game.players))

    def __call__[R: Game](self, index: Index[Weight, R]) -> R:
        """Compute ``index`` exactly and package it as the index's result game.

        The game and index batches broadcast like arrays: align them explicitly, e.g.
        ``computer[:, None](index)`` for every game with every index.

        Raises:
            NotImplementedError: The index's weight family has no exact route yet.
        """
        interactions = index.interactions(self.table.n_players)
        if isinstance(index.weights, DiscreteDerivativeWeight):
            values = _discrete_derivative_values(self.table, index.weights, interactions)
        else:
            msg = f"No exact route for {type(index.weights).__name__} weights."
            raise NotImplementedError(msg)
        return index.build(values, interactions, self.table.players)


def _discrete_derivative_values(
    game: Shaped[TableGame, "#*batch"],
    weights: Shaped[DiscreteDerivativeWeight, "#*batch"],
    interactions: Bool[jax.Array, "n_interactions n_players"],
) -> Float[jax.Array, "*batch n_interactions"]:
    """Sum the weighted game values over all coalitions (SHAP-IQ representation)."""
    n_players = game.n_players
    # Add a singleton batch axis to the game and weights, so they broadcast into the interaction batch.
    computer = WeightedGameValue(
        game=cast("FinitePlayerGame[Any, Any]", jax_expand_dims(game, -1)),
        weights=cast("DiscreteDerivativeWeight", jax_expand_dims(weights, -1)),
        interaction=interactions,
    )  # (*batch, n_interactions)
    coalitions = subset_patterns(n_players).reshape(
        2**n_players, *(1,) * computer.ndim, n_players
    )  # (2**n_players, *ones, n_players)
    # A TableGame's values are plain arrays, so the weighted values are too.
    weighted = cast("jax.Array", computer(coalition=coalitions))  # (2**n_players, *batch, n_int.)
    return jnp.sum(weighted, axis=0)  # (*batch, n_interactions)
