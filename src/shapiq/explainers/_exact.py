from __future__ import annotations

from functools import cache
from itertools import combinations

import jax.numpy as jnp
from jax import Array

from shapiq.coalitions import DenseCoalitionArray
from shapiq.explainers._base import Explainer
from shapiq.explainers._valueaxes import to_leading, to_trailing
from shapiq.explanations import DenseExplanationArray
from shapiq.games import Game
from shapiq.interactions import (
    AggregationIndex,
    ArgminIndex,
    CardinalInteractionIndex,
    CoalitionFunctional,
    GeneralizedValueIndex,
    aggregate_supersets,
    derive_functional,
)

type ExactIndex = (
    CardinalInteractionIndex | GeneralizedValueIndex | ArgminIndex | AggregationIndex
)


class ExactExplainer(Explainer[Array, Game[Array]]):
    """Explainer computing interaction indices exactly from all coalitions.

    The exact explainer evaluates the game once on every one of the
    ``2**n_players`` coalitions and contracts the index's derived coalition
    functional over the values, which is feasible for games with roughly up
    to fifteen players. Any index with a derivable functional is supported
    through its capability alone — cardinal interaction indices
    (discrete-derivative weights), generalized values (bloc-marginal
    weights), argmin indices (compiled least squares solution operators),
    and aggregations of any of them; shapiq ships SV, BV, SII, BII, CHII,
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
                discrete-derivative weights, bloc-marginal weights, an
                argmin specification, or an aggregation of one works.

        Raises:
            TypeError: If the index is passed as a string name or provides
                no capability with a derivable coalition functional.
            ValueError: If the index order is out of range for the game.
        """
        if isinstance(index, str):
            msg = f"interaction indices are objects: pass shapiq.SII(order=2) instead of {index!r}"
            raise TypeError(msg)
        if not isinstance(
            index,
            (
                AggregationIndex,
                CardinalInteractionIndex,
                GeneralizedValueIndex,
                ArgminIndex,
            ),
        ):
            name = getattr(index, "name", type(index).__name__)
            msg = (
                f"ExactExplainer does not support {name!r}: the index provides "
                "neither discrete-derivative weights, bloc-marginal weights, an "
                "argmin specification, nor an aggregation of an index providing them"
            )
            raise TypeError(msg)
        super().__init__(game, index)
        self._exact_index: ExactIndex = index
        self._powerset_values: Array | None = None

    def explain(self) -> DenseExplanationArray[Array]:
        """Contract the derived coalition functional over all game values.

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
        if isinstance(index, AggregationIndex):
            functional = derive_functional(index.base_index, n_players, order)
            attributions = aggregate_supersets(
                _functional_attributions(values, masks, functional),
                index.aggregation_coefficients(),
                n_players,
            )
        else:
            functional = derive_functional(index, n_players, order)
            attributions = _functional_attributions(values, masks, functional)
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
