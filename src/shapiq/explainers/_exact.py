from __future__ import annotations

from functools import cache
from itertools import combinations
from math import comb

import jax.numpy as jnp
from jax import Array

from shapiq.coalitions import DenseCoalitionArray
from shapiq.explainers._base import Explainer
from shapiq.explainers._faithful import (
    bernoulli_design,
    eliminate_constraint,
    interaction_design,
    solve_faithful,
    solve_pinned,
)
from shapiq.explainers._valueaxes import to_leading, to_trailing
from shapiq.explanations import DenseExplanationArray
from shapiq.games import Game
from shapiq.interactions import (
    FBII,
    KADDSHAP,
    AggregationIndex,
    CardinalInteractionIndex,
    CoalitionFunctional,
    GeneralizedValueIndex,
    RegressionIndex,
    aggregate_supersets,
    derive_functional,
)

type ExactIndex = (
    CardinalInteractionIndex
    | GeneralizedValueIndex
    | RegressionIndex
    | AggregationIndex
    | FBII
    | KADDSHAP
)


class ExactExplainer(Explainer[Array, Game[Array]]):
    """Explainer computing interaction indices exactly from all coalitions.

    The exact explainer evaluates the game once on every one of the
    ``2**n_players`` coalitions and computes the requested index without
    sampling error, which is feasible for games with roughly up to fifteen
    players. Any cardinal interaction index (discrete-derivative weights),
    generalized value (bloc-marginal weights), or Shapley-kernel regression
    index is supported through its capability, alongside dedicated solvers
    for k-SII, FBII, and kADD-SHAP; shapiq ships SV, BV, SII, BII, CHII,
    k-SII, STII, FSII, FBII, kADD-SHAP, the generalized values SGV, BGV,
    CHGV, IGV, EGV, and JointSV, and the Moebius and Co-Moebius transforms.
    The powerset evaluation happens on the first ``explain()`` call and is
    reused afterwards; construction never evaluates the game. Computations
    run in the game's value precision (float32 under JAX defaults; enabling
    JAX x64 yields float64 throughout).

    Example:
        >>> explainer = ExactExplainer(game, SII(order=2))
        >>> explanation = explainer.explain()
        >>> pair_interaction = explanation((0, 1))
    """

    def __init__(self, game: Game[Array], index: ExactIndex) -> None:
        """Initialize without evaluating the game.

        Args:
            game: Game to explain. The game is evaluated on all
                ``2**n_players`` coalitions.
            index: The interaction index to compute, as an index object such
                as ``SV()`` or ``SII(order=2)``. Any index providing
                discrete-derivative weights, bloc-marginal weights, or a
                regression kernel works; k-SII, FBII, and kADD-SHAP have
                dedicated solvers.

        Raises:
            TypeError: If the index is passed as a string name or provides
                no capability the exact explainer can compute.
            ValueError: If the index order is out of range for the game.
        """
        if isinstance(index, str):
            msg = f"interaction indices are objects: pass shapiq.SII(order=2) instead of {index!r}"
            raise TypeError(msg)
        if not isinstance(
            index,
            (
                FBII,
                KADDSHAP,
                AggregationIndex,
                CardinalInteractionIndex,
                GeneralizedValueIndex,
                RegressionIndex,
            ),
        ):
            name = getattr(index, "name", type(index).__name__)
            msg = (
                f"ExactExplainer does not support {name!r}: the index provides "
                "neither discrete-derivative weights, bloc-marginal weights, "
                "a regression kernel, nor a dedicated exact solver"
            )
            raise TypeError(msg)
        super().__init__(game, index)
        self._exact_index: ExactIndex = index
        self._powerset_values: Array | None = None

    def explain(self) -> DenseExplanationArray[Array]:
        """Compute the configured index exactly from all game values.

        Returns:
            A dense explanation. Indices that declare an order-0 attribution
            carry the empty-coalition value there — except FBII, whose
            order-0 attribution is its fitted intercept, and the Co-Moebius
            transform, whose order-0 attribution is the grand-coalition
            value.
        """
        n_players = self.game.n_players
        n_value_axes = len(self.game.value_shape)
        values = to_leading(self._game_values(), n_value_axes)
        masks = _powerset_masks(n_players)
        index = self._exact_index
        order = self.order
        if isinstance(index, FBII):
            attributions = _banzhaf_regression_attributions(values, masks, order)
        elif isinstance(index, KADDSHAP):
            attributions = _kadd_regression_attributions(values, masks, index, order)
        elif isinstance(index, AggregationIndex):
            functional = derive_functional(index.base_index, n_players, order)
            attributions = aggregate_supersets(
                _functional_attributions(values, masks, functional),
                index.aggregation_coefficients(),
                n_players,
            )
        elif isinstance(index, (CardinalInteractionIndex, GeneralizedValueIndex)):
            functional = derive_functional(index, n_players, order)
            attributions = _functional_attributions(values, masks, functional)
        else:
            attributions = _regression_attributions(values, masks, index, order)
        if index.includes_empty_interaction and 0 not in attributions:
            attributions[0] = values[..., :1]
        return DenseExplanationArray(
            attributions_by_order={
                size: to_trailing(block, n_value_axes) for size, block in attributions.items()
            },
            n_players=n_players,
            interaction_index=self.interaction_index,
            order=order,
            shape=self.game.target_shape,
            orientation=self.orientation,
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


def _functional_attributions(
    values: Array,
    masks: Array,
    functional: CoalitionFunctional,
) -> dict[int, Array]:
    """Contract the derived coalition functional densely over all game values."""
    return {
        size: jnp.einsum("...c,ci->...i", values, functional.coefficient_matrix(masks, size))
        for size in functional.interaction_sizes
    }


def _regression_attributions(
    values: Array,
    masks: Array,
    index: RegressionIndex,
    order: int,
) -> dict[int, Array]:
    """Solve the constrained kernel-weighted least squares fit on the full powerset."""
    n_players = masks.shape[-1]
    sizes = jnp.sum(masks, axis=-1)
    sqrt_weights = jnp.sqrt(index.regression_kernel(n_players)[sizes])
    reduced, pivot = eliminate_constraint(interaction_design(masks, order))
    n_coalitions = masks.shape[-2]
    response = (values - values[..., :1]).reshape(-1, n_coalitions).T
    delta = (values[..., -1] - values[..., 0]).reshape(-1)
    solution = solve_faithful(reduced, pivot, response, delta, sqrt_weights=sqrt_weights)
    return _solution_blocks(solution, values, n_players, order)


def _banzhaf_regression_attributions(
    values: Array,
    masks: Array,
    order: int,
) -> dict[int, Array]:
    """Solve the unconstrained uniform least squares fit with a free intercept."""
    n_players = masks.shape[-1]
    n_coalitions = masks.shape[-2]
    design = jnp.concatenate(
        [jnp.ones((n_coalitions, 1)), interaction_design(masks, order)],
        axis=-1,
    )
    response = (values - values[..., :1]).reshape(-1, n_coalitions).T
    solution, *_ = jnp.linalg.lstsq(design, response)
    attributions = _solution_blocks(solution[1:], values, n_players, order)
    intercept = solution[0].T.reshape(*values.shape[:-1], 1)
    attributions[0] = intercept + values[..., :1]
    return attributions


def _kadd_regression_attributions(
    values: Array,
    masks: Array,
    index: KADDSHAP,
    order: int,
) -> dict[int, Array]:
    """Solve the Bernoulli-basis Shapley regression pinned at the grand coalition."""
    n_players = masks.shape[-1]
    n_coalitions = masks.shape[-2]
    sizes = jnp.sum(masks, axis=-1)
    sqrt_weights = jnp.sqrt(index.regression_kernel(n_players)[sizes])
    design = bernoulli_design(masks, order)
    response = (values - values[..., :1]).reshape(-1, n_coalitions).T
    delta = (values[..., -1] - values[..., 0]).reshape(-1)
    solution = solve_pinned(design, design[-1], response, delta, sqrt_weights=sqrt_weights)
    return _solution_blocks(solution, values, n_players, order)


def _solution_blocks(
    solution: Array,
    values: Array,
    n_players: int,
    order: int,
) -> dict[int, Array]:
    """Split a stacked least squares solution into per-order attribution blocks."""
    attributions: dict[int, Array] = {}
    offset = 0
    for size in range(1, order + 1):
        n_interactions = comb(n_players, size)
        block = solution[offset : offset + n_interactions].T
        attributions[size] = block.reshape(*values.shape[:-1], n_interactions)
        offset += n_interactions
    return attributions
