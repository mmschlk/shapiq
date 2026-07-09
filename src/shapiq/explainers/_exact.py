from __future__ import annotations

from fractions import Fraction
from functools import cache
from itertools import combinations
from math import comb
from typing import cast

import jax.numpy as jnp
from jax import Array

from shapiq.coalitions import DenseCoalitionArray
from shapiq.explainers._base import (
    Explainer,
    missing_index_members,
    reject_common_index_mistakes,
)
from shapiq.explainers._faithful import (
    eliminate_constraint,
    interaction_design,
    interaction_masks,
    solve_faithful,
)
from shapiq.explainers._valueaxes import to_leading, to_trailing
from shapiq.explanations import DenseExplanationArray
from shapiq.games import Game
from shapiq.interactions import (
    FBII,
    KADDSHAP,
    KSII,
    SII,
    CardinalInteractionIndex,
    GeneralizedValueIndex,
    RegressionIndex,
    WeightedFBII,
)

type ExactIndex = (
    CardinalInteractionIndex
    | GeneralizedValueIndex
    | RegressionIndex
    | KSII
    | FBII
    | WeightedFBII
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
    for k-SII, FBII, WeightedFBII, and kADD-SHAP; shapiq ships SV, BV,
    WeightedBV, SII, BII, WeightedBII, CHII, k-SII, STII, FSII, FBII,
    WeightedFBII, kADD-SHAP, the generalized values SGV, BGV, CHGV, IGV,
    EGV, and JointSV, and the Moebius and Co-Moebius transforms.
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
                regression kernel works; k-SII, FBII, WeightedFBII, and
                kADD-SHAP have dedicated solvers.

        Raises:
            TypeError: If the index is passed as a string name or provides
                no capability the exact explainer can compute.
            ValueError: If the index order is out of range for the game.
        """
        reject_common_index_mistakes(index)
        for shipped in (KSII, FBII, WeightedFBII, KADDSHAP):
            if isinstance(index, shipped) and type(index) is not shipped:
                msg = (
                    f"{type(index).__name__} subclasses {shipped.__name__}, whose exact "
                    f"solver dispatches on the exact index type: pass {shipped.__name__} "
                    "itself or define an independent index with its own capabilities"
                )
                raise TypeError(msg)
        if not isinstance(
            index,
            (
                KSII,
                FBII,
                KADDSHAP,
                CardinalInteractionIndex,
                GeneralizedValueIndex,
                RegressionIndex,
            ),
        ):
            name = getattr(index, "name", type(index).__name__)
            missing = missing_index_members(index)
            hint = f"; it is also missing index members: {', '.join(missing)}" if missing else ""
            msg = (
                f"ExactExplainer does not support {name!r}: the index provides "
                "neither discrete-derivative weights, bloc-marginal weights, "
                "a regression kernel, nor a dedicated exact solver" + hint
            )
            raise TypeError(msg)
        super().__init__(game, index)
        self._exact_index: ExactIndex = index
        self._powerset_values: Array | None = None

    def explain(self) -> DenseExplanationArray[Array]:
        """Compute the configured index exactly from all game values.

        Returns:
            A dense explanation whose baseline is the empty-coalition value
            and whose attributions are computed on the centered game. Only
            indices with a genuine order-0 attribution carry one: FBII and
            WeightedFBII carry their fitted intercept, the Co-Moebius
            transform its grand total ``v(N) - v(empty)``.
        """
        n_players = self.game.n_players
        n_value_axes = len(self.game.value_shape)
        values = to_leading(self._game_values(), n_value_axes)
        baseline = to_trailing(values[..., 0], n_value_axes)
        values = values - values[..., :1]
        masks = _powerset_masks(n_players)
        index = self._exact_index
        order = self.order
        if type(index) is KSII:
            attributions = _aggregated_ksii_attributions(values, masks, order)
        elif type(index) is FBII:
            attributions = _free_intercept_regression_attributions(values, masks, order)
        elif type(index) is WeightedFBII:
            weighted = cast("WeightedFBII", index)  # type(x) is Y does not narrow in ty
            kernel = weighted.regression_kernel(n_players)
            sqrt_weights = jnp.sqrt(kernel / jnp.max(kernel))[jnp.sum(masks, axis=-1)]
            attributions = _free_intercept_regression_attributions(
                values,
                masks,
                order,
                sqrt_weights=sqrt_weights,
            )
        elif type(index) is KADDSHAP:
            attributions = _kadd_regression_attributions(values, masks, index, order)
        elif isinstance(index, CardinalInteractionIndex):
            attributions = {
                size: _weighted_derivatives(
                    values,
                    masks,
                    size,
                    index.derivative_weights(n_players, size),
                )
                for size in range(index.min_interaction_size, order + 1)
            }
        elif isinstance(index, GeneralizedValueIndex):
            attributions = {
                size: _weighted_marginals(
                    values,
                    masks,
                    size,
                    index.marginal_weights(n_players, size),
                )
                for size in range(1, order + 1)
            }
        else:
            # KSII is dispatched above, so only regression-capable indices remain
            regression_index = cast("RegressionIndex", index)
            attributions = _regression_attributions(values, masks, regression_index, order)
        return DenseExplanationArray(
            attributions_by_order={
                size: to_trailing(block, n_value_axes) for size, block in attributions.items()
            },
            n_players=n_players,
            index=self.index,
            order=order,
            shape=self.game.target_shape,
            value_shape=self.game.value_shape,
            baseline=baseline,
        )

    def _game_values(self) -> Array:
        """Evaluate the game on the full powerset once and reuse the values."""
        if self._powerset_values is None:
            coalitions = DenseCoalitionArray(_powerset_masks(self.game.n_players))
            self._powerset_values = jnp.asarray(self.game(coalitions))
        return self._powerset_values


def _require_weight_length(source: str, weights: Array, n_players: int, size: int) -> None:
    """Reject weight vectors whose length would silently misindex under JAX."""
    expected = n_players - size + 1
    if weights.shape[-1] != expected:
        msg = (
            f"{source}(n_players={n_players}, interaction_size={size}) must return "
            f"{expected} weights for outside-coalition sizes 0..{expected - 1}, "
            f"got {weights.shape[-1]}"
        )
        raise ValueError(msg)


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
    _require_weight_length("derivative_weights", weights, n_players, size)
    member_masks = interaction_masks(n_players, size)
    intersections = masks.astype(jnp.int32) @ member_masks.T.astype(jnp.int32)
    outside_sizes = jnp.sum(masks, axis=-1)[:, None] - intersections
    signs = jnp.where((size - intersections) % 2 == 0, 1.0, -1.0)
    kernel = signs * weights[outside_sizes]
    return jnp.einsum("...c,ci->...i", values, kernel)


def _weighted_marginals(
    values: Array,
    masks: Array,
    size: int,
    weights: Array,
) -> Array:
    """Sum weighted marginal contributions of whole interactions joining coalitions."""
    n_players = masks.shape[-1]
    _require_weight_length("marginal_weights", weights, n_players, size)
    member_masks = interaction_masks(n_players, size)
    intersections = masks.astype(jnp.int32) @ member_masks.T.astype(jnp.int32)
    coalition_sizes = jnp.sum(masks, axis=-1)[:, None]
    # the clamp only affects lanes discarded by the where masks below
    largest = weights.shape[0] - 1
    gains = jnp.where(
        intersections == size,
        weights[jnp.clip(coalition_sizes - size, 0, largest)],
        0.0,
    )
    losses = jnp.where(
        intersections == 0,
        weights[jnp.clip(coalition_sizes, 0, largest)],
        0.0,
    )
    return jnp.einsum("...c,ci->...i", values, gains - losses)


def _aggregated_ksii_attributions(
    values: Array,
    masks: Array,
    order: int,
) -> dict[int, Array]:
    """Aggregate exact SII values into efficient k-SII values via Bernoulli numbers."""
    n_players = masks.shape[-1]
    shapley_interactions = SII(order=order)
    sii = {
        size: _weighted_derivatives(
            values,
            masks,
            size,
            shapley_interactions.derivative_weights(n_players, size),
        )
        for size in range(1, order + 1)
    }
    bernoulli = _bernoulli_numbers(order)
    members = {size: interaction_masks(n_players, size) for size in range(1, order + 1)}
    attributions: dict[int, Array] = {}
    for size in range(1, order + 1):
        block = sii[size]
        for larger in range(size + 1, order + 1):
            counts = members[size].astype(jnp.int32) @ members[larger].T.astype(jnp.int32)
            subsets = 1.0 * (counts == size)
            block = block + bernoulli[larger - size] * jnp.einsum(
                "...j,ij->...i",
                sii[larger],
                subsets,
            )
        attributions[size] = block
    return attributions


def _regression_attributions(
    values: Array,
    masks: Array,
    index: RegressionIndex,
    order: int,
) -> dict[int, Array]:
    """Solve the constrained kernel-weighted least squares fit on the full powerset."""
    n_players = masks.shape[-1]
    kernel = index.regression_kernel(n_players)
    if kernel[0] != 0.0 or kernel[-1] != 0.0:
        msg = (
            f"the exact constrained regression solver requires zero kernel weight "
            f"at the empty and grand coalition, got {index.name!r} weights "
            f"{float(kernel[0])!r} and {float(kernel[-1])!r}; unconstrained kernels "
            "with a free intercept (such as FBII's) need a dedicated solver"
        )
        raise TypeError(msg)
    sizes = jnp.sum(masks, axis=-1)
    sqrt_weights = jnp.sqrt(kernel[sizes])
    reduced, pivot = eliminate_constraint(interaction_design(masks, order))
    n_coalitions = masks.shape[-2]
    response = (values - values[..., :1]).reshape(-1, n_coalitions).T
    delta = (values[..., -1] - values[..., 0]).reshape(-1)
    solution = solve_faithful(reduced, pivot, response, delta, sqrt_weights=sqrt_weights)
    return _solution_blocks(solution, values, n_players, order)


def _free_intercept_regression_attributions(
    values: Array,
    masks: Array,
    order: int,
    sqrt_weights: Array | None = None,
) -> dict[int, Array]:
    """Solve an unconstrained kernel least squares fit with a free intercept.

    Without ``sqrt_weights`` every row enters with unit weight (the uniform
    Banzhaf kernel); with them the fit minimizes the kernel-weighted squared
    error, as for the product-measure kernels of the weighted Banzhaf family.
    """
    n_players = masks.shape[-1]
    n_coalitions = masks.shape[-2]
    design = jnp.concatenate(
        [jnp.ones((n_coalitions, 1)), interaction_design(masks, order)],
        axis=-1,
    )
    response = (values - values[..., :1]).reshape(-1, n_coalitions).T
    if sqrt_weights is not None:
        solution, *_ = jnp.linalg.lstsq(
            sqrt_weights[:, None] * design,
            sqrt_weights[:, None] * response,
        )
    else:
        solution, *_ = jnp.linalg.lstsq(design, response)
    attributions = _solution_blocks(solution[1:], values, n_players, order)
    attributions[0] = solution[0].T.reshape(*values.shape[:-1], 1)
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
    design = _bernoulli_design(masks, order)
    constraint = design[-1]
    pivot_column = int(jnp.argmax(jnp.abs(constraint)))
    anchor = constraint[pivot_column]
    pivot = design[:, pivot_column : pivot_column + 1]
    reduced = jnp.delete(design - pivot * (constraint / anchor)[None, :], pivot_column, axis=1)
    response = (values - values[..., :1]).reshape(-1, n_coalitions).T
    delta = (values[..., -1] - values[..., 0]).reshape(-1)
    shifted = response - (pivot / anchor) * delta[None, :]
    partial, *_ = jnp.linalg.lstsq(
        sqrt_weights[:, None] * reduced,
        sqrt_weights[:, None] * shifted,
    )
    others = jnp.delete(constraint, pivot_column)
    back_substituted = (delta[None, :] - others[:, None].T @ partial) / anchor
    solution = jnp.insert(partial, pivot_column, back_substituted[0], axis=0)
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


def _bernoulli_numbers(order: int) -> list[float]:
    """Return the Bernoulli numbers up to ``order`` with the B(1) = -1/2 convention."""
    numbers = [Fraction(1)]
    for m in range(1, order + 1):
        acc = sum((Fraction(comb(m + 1, j)) * numbers[j] for j in range(m)), Fraction(0))
        numbers.append(-acc / (m + 1))
    return [float(number) for number in numbers]


def _bernoulli_design(masks: Array, order: int) -> Array:
    """Return Bernoulli-weighted intersection columns for all interactions up to order."""
    n_players = masks.shape[-1]
    table = _bernoulli_weight_table(order)
    columns = []
    for size in range(1, order + 1):
        member_masks = interaction_masks(n_players, size)
        intersections = masks.astype(jnp.int32) @ member_masks.T.astype(jnp.int32)
        columns.append(table[size][intersections])
    return jnp.concatenate(columns, axis=1)


def _bernoulli_weight_table(order: int) -> Array:
    """Return kADD-SHAP design weights per interaction and intersection size."""
    bernoulli = _bernoulli_numbers(order)
    table = [[0.0] * (order + 1) for _ in range(order + 1)]
    for size in range(1, order + 1):
        for intersection in range(1, size + 1):
            table[size][intersection] = sum(
                comb(intersection, top) * bernoulli[size - top]
                for top in range(1, intersection + 1)
            )
    return jnp.asarray(table)
