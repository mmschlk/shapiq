"""Exact computation of interaction indices from a cached dense coalition table."""

from dataclasses import dataclass
from typing import Any, ClassVar, Self

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Float, Shaped
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.game import DenseCoalitionGame, FinitePlayerGame, Game, subset_patterns
from shapiq.index._base import DiscreteDerivativeIndex, weight_parameters
from shapiq.typing import typechecked
from shapiq.value._dtype import resolve_weight_dtype


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class ExactComputer(JaxAxisProtected["ExactComputer"]):
    """Exact interaction indices of a game batch, evaluating each coalition only once.

    ``from_game`` stores every coalition value as a ``DenseCoalitionGame``; calling the computer
    with an index reuses that table, so several indices cost one round of game evaluations. The
    batch is the game batch. Memory grows with ``2**n_players`` per game.
    """

    table: DenseCoalitionGame
    protected_axes: ClassVar[dict[str, int]] = {"table": 0}

    @classmethod
    def from_game(cls, game: Shaped[FinitePlayerGame, "#*batch"]) -> Self:
        """Evaluate ``game`` on every coalition and cache the values."""
        return cls(table=DenseCoalitionGame.from_game(game))

    @typechecked
    def __call__[R: Game](self, index: Shaped[DiscreteDerivativeIndex[R], "#*batch"]) -> R:
        """Compute ``index`` as a weighted sum of the cached values and package its result game.

        The game and index batches broadcast like arrays: align them explicitly, e.g.
        ``computer[:, None](index)`` for every game with every index.
        """
        n_players = self.table.n_players
        interactions = index.interactions(n_players)
        # subset_patterns lists coalitions in DenseCoalitionGame order, matching the last axis.
        weights = discrete_derivative_coefficients(index, subset_patterns(n_players), interactions)
        values = self.table.values
        dtype = resolve_weight_dtype(values, *weight_parameters(index))
        values = jnp.sum(weights.astype(dtype) * values[..., None, :], axis=-1)
        return index.build(values, interactions, self.table.players)


def discrete_derivative_coefficients(
    index: Shaped[DiscreteDerivativeIndex[Any], "#*batch"],
    coalitions: Bool[jax.Array, "#*batch n_coalitions n_players"],
    interactions: Bool[jax.Array, "n_interactions n_players"],
) -> Float[jax.Array, "*batch n_interactions n_coalitions"]:
    """Return the coefficient of v(S) in phi(T): ``(-1)**|T - S| w(|S - T|, |T|, n)``.

    Expanding the discrete derivative in ``phi(T) = sum over S ⊆ N - T of w(|S|, |T|, n)
    Δ_T v(S)`` gives every coalition one coefficient per interaction, the representation used by
    SHAP-IQ (Fumagalli et al., 2023). Summed against every coalition value, they give the index.
    The batches of the index and the coalitions broadcast, so either can stay compact.
    """
    n_players = coalitions.shape[-1]
    order = jnp.sum(interactions, axis=-1)[:, None]  # (n_interactions, 1)
    coalitions = coalitions[..., None, :, :]  # (*batch, 1, n_coalitions, n_players)
    overlap = jnp.sum(interactions[:, None] & coalitions, axis=-1)  # (*batch, n_int., n_coal.)
    outside = jnp.sum(coalitions, axis=-1) - overlap  # |S - T|
    # Two trailing batch axes, so the index parameters broadcast against (T, S).
    weights = index[..., None, None].discrete_derivative_weight(outside, order, n_players)
    signs = jnp.where(((order - overlap) & 1) == 0, 1, -1)
    return signs * weights  # (*batch, n_interactions, n_coalitions)
