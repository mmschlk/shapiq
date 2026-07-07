from __future__ import annotations

from functools import cache
from itertools import combinations
from math import comb

import jax.numpy as jnp
from jax import Array

from shapiq.coalitions import DenseCoalitionArray
from shapiq.explainers._base import Explainer
from shapiq.explainers._faithful import (
    eliminate_constraint,
    interaction_design,
    interaction_masks,
    solve_faithful,
)
from shapiq.explainers._valueaxes import to_leading, to_trailing
from shapiq.explanations import DenseExplanationArray
from shapiq.games import Game
from shapiq.interactions import RegressionIndex, WeightedDerivativeIndex


class ExactExplainer(Explainer[Array, Game[Array]]):
    """Explainer computing interaction indices exactly from all coalitions.

    The exact explainer evaluates the game once on every one of the
    ``2**n_players`` coalitions and computes the requested index without
    sampling error, which is feasible for games with roughly up to fifteen
    players. Any index providing discrete-derivative weights or a regression
    kernel is supported; shapiq ships SV, BV, SII, BII, STII, and FSII. The
    powerset evaluation happens on the first ``explain()`` call and is
    reused afterwards; construction never evaluates the game. Computations
    run in the game's value precision (float32 under JAX defaults; enabling
    JAX x64 yields float64 throughout).

    Example:
        >>> explainer = ExactExplainer(game, SII(order=2))
        >>> explanation = explainer.explain()
        >>> pair_interaction = explanation((0, 1))
    """

    def __init__(
        self,
        game: Game[Array],
        index: WeightedDerivativeIndex | RegressionIndex,
    ) -> None:
        """Initialize without evaluating the game.

        Args:
            game: Game to explain. Must produce scalar values per coalition.
                The game is evaluated on all ``2**n_players`` coalitions.
            index: The interaction index to compute, as an index object such
                as ``SV()`` or ``SII(order=2)``. Any index providing
                discrete-derivative weights or a regression kernel works.

        Raises:
            TypeError: If the index is passed as a string name or provides
                neither supported capability.
            ValueError: If the index order is out of range for the game.
        """
        if isinstance(index, str):
            msg = f"interaction indices are objects: pass shapiq.SII(order=2) instead of {index!r}"
            raise TypeError(msg)
        if not isinstance(index, (WeightedDerivativeIndex, RegressionIndex)):
            name = getattr(index, "name", type(index).__name__)
            msg = (
                f"ExactExplainer does not support {name!r}: the index provides "
                "neither discrete-derivative weights nor a regression kernel"
            )
            raise TypeError(msg)
        super().__init__(game, index)
        self._exact_index: WeightedDerivativeIndex | RegressionIndex = index
        self._powerset_values: Array | None = None

    def explain(self) -> DenseExplanationArray[Array]:
        """Compute the configured index exactly from all game values.

        Returns:
            A dense explanation. Indices anchored at the empty coalition
            (SV, BV, STII, FSII) carry the empty-coalition value as the
            order-0 attribution; SII and BII represent orders one through
            ``order``.
        """
        n_players = self.game.n_players
        n_value_axes = len(self.game.value_shape)
        values = to_leading(self._game_values(), n_value_axes)
        masks = _powerset_masks(n_players)
        index = self._exact_index
        if isinstance(index, WeightedDerivativeIndex):
            attributions = {
                size: _weighted_derivatives(
                    values,
                    masks,
                    size,
                    index.derivative_weights(n_players, size),
                )
                for size in range(1, self.order + 1)
            }
        else:
            attributions = _regression_attributions(values, masks, index, self.order)
        if index.includes_empty_interaction:
            attributions[0] = values[..., :1]
        return DenseExplanationArray(
            attributions_by_order={
                size: to_trailing(block, n_value_axes) for size, block in attributions.items()
            },
            n_players=n_players,
            interaction_index=self.interaction_index,
            order=self.order,
            shape=self.game.target_shape,
            value_shape=self.game.value_shape,
        )

    def _game_values(self) -> Array:
        """Evaluate the game on the full powerset once and reuse the values."""
        if self._powerset_values is None:
            coalitions = DenseCoalitionArray(_powerset_masks(self.game.n_players))
            self._powerset_values = jnp.asarray(self.game(coalitions))
        return self._powerset_values


@cache
def _powerset_masks(n_players: int) -> Array:
    """Return all coalitions as dense masks, ordered by size then lexicographically."""
    return jnp.asarray(
        [
            [player in members for player in range(n_players)]
            for size in range(n_players + 1)
            for members in combinations(range(n_players), size)
        ],
    )


def _weighted_derivatives(
    values: Array,
    masks: Array,
    size: int,
    weights: Array,
) -> Array:
    """Sum signed, weighted game values into per-interaction attributions."""
    n_players = masks.shape[-1]
    member_masks = interaction_masks(n_players, size)
    intersections = masks.astype(jnp.int32) @ member_masks.T.astype(jnp.int32)
    outside_sizes = jnp.sum(masks, axis=-1)[:, None] - intersections
    signs = jnp.where((size - intersections) % 2 == 0, 1.0, -1.0)
    kernel = signs * weights[outside_sizes]
    return jnp.einsum("...c,ci->...i", values, kernel)


def _regression_attributions(
    values: Array,
    masks: Array,
    index: RegressionIndex,
    order: int,
) -> dict[int, Array]:
    """Solve the kernel-weighted least squares fit on the full powerset."""
    n_players = masks.shape[-1]
    sizes = jnp.sum(masks, axis=-1)
    sqrt_weights = jnp.sqrt(index.regression_kernel(n_players)[sizes])
    reduced, pivot = eliminate_constraint(interaction_design(masks, order))
    n_coalitions = masks.shape[-2]
    response = (values - values[..., :1]).reshape(-1, n_coalitions).T
    delta = (values[..., -1] - values[..., 0]).reshape(-1)
    solution = solve_faithful(reduced, pivot, response, delta, sqrt_weights=sqrt_weights)
    attributions: dict[int, Array] = {}
    offset = 0
    for size in range(1, order + 1):
        n_interactions = comb(n_players, size)
        block = solution[offset : offset + n_interactions].T
        attributions[size] = block.reshape(*values.shape[:-1], n_interactions)
        offset += n_interactions
    return attributions
