from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from shapiq.errors import InsufficientSamplesError
from shapiq.explainers._evidence import EvidenceApproximator
from shapiq.explainers._valueaxes import to_leading, to_trailing
from shapiq.explanations import DenseExplanationArray
from shapiq.interactions import (
    AggregationIndex,
    ArgminIndex,
    CardinalInteractionIndex,
    GeneralizedValueIndex,
    aggregate_supersets,
    derive_functional,
    validate_interaction_metadata,
)
from shapiq.sampling import CoalitionSizeSampler, EmptyState, SamplingState

if TYPE_CHECKING:
    from shapiq.games import Game
    from shapiq.sampling import ShareSamples

type MonteCarloIndex = (
    CardinalInteractionIndex | GeneralizedValueIndex | ArgminIndex | AggregationIndex
)


class MonteCarlo(EvidenceApproximator):
    """Unbiased sampled estimator derived from an index's coalition functional.

    Any index declaring discrete-derivative weights (cardinal interaction
    index), bloc-marginal weights (generalized value), an argmin
    specification (compiled least squares solution operator, as for FSII,
    FBII, and kADD-SHAP), or a superset aggregation of such an index is
    supported through its capability alone: the index's coalition functional
    is derived mechanically, its coefficient mass profile becomes the size
    distribution of the sampler, and every sampled coalition enters the
    estimate with its coefficient divided by its sampling probability. The
    empty and grand coalition are seed samples and contribute their
    coefficients exactly, so the estimator is unbiased for every supported
    index at any number of completed units — including indices defined
    through ``define_cardinal_index``, ``define_generalized_value``, or
    ``define_regression_index`` that shapiq has never heard of.

    Example:
        >>> approximator = MonteCarlo(game, SII(order=2), random_state=0)
        >>> explanation = approximator.sample(500).explain()
        >>> pair_interaction = explanation((0, 1))
    """

    def __init__(
        self,
        game: Game[Array],
        index: MonteCarloIndex,
        *,
        random_state: Array | int = 0,
        share_samples: ShareSamples = False,
        paired: bool = False,
        track_history: bool = False,
        deduplicate: bool = False,
    ) -> None:
        """Initialize without evaluating the game.

        Args:
            game: Game to explain. Must have at least two players.
            index: The interaction index to estimate. Any index providing
                discrete-derivative weights or bloc-marginal weights works,
                as does any aggregation of one.
            random_state: Integer seed or JAX PRNG key for drawing
                coalitions.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes. ``False`` samples independently per
                target; ``True`` shares across all target axes; an integer or
                tuple of integers shares across the selected axes.
            paired: Whether every sampled coalition is accompanied by its
                complement, which reduces variance for indices whose
                coefficient mass is roughly size-symmetric.
            track_history: Whether to record value-equivalent history for
                rollback and convergence analysis.
            deduplicate: Whether to evaluate each distinct coalition at most
                once; repeats reuse stored values and only novel evaluations
                count toward the budget. Requires shared samples.

        Raises:
            TypeError: If no coalition functional is derivable for the index.
            ValueError: If the game has fewer than two players, if the index
                order is out of range, or if ``deduplicate`` is enabled
                without samples shared across explanation targets.
        """
        if not isinstance(
            index,
            (AggregationIndex, CardinalInteractionIndex, GeneralizedValueIndex, ArgminIndex),
        ):
            name = getattr(index, "name", type(index).__name__)
            msg = (
                f"MonteCarlo does not support {name!r}: the index declares neither "
                "discrete-derivative weights, bloc-marginal weights, an argmin "
                "specification, nor an aggregation of an index declaring them"
            )
            raise TypeError(msg)
        order = game.n_players if index.order is None else index.order
        validate_interaction_metadata(
            interaction_index=index.name,
            order=order,
            orientation=index.orientation,
            n_players=game.n_players,
        )
        functional_index = index.base_index if isinstance(index, AggregationIndex) else index
        self._functional = derive_functional(functional_index, game.n_players, order)
        sampler = CoalitionSizeSampler(
            game.n_players,
            game.target_shape,
            size_weights=self._functional.size_mass(),
            share_samples=share_samples,
            paired=paired,
            random_state=random_state,
        )
        state = EmptyState(track_history=track_history)
        super().__init__(game, sampler, state, index=index)
        self._init_deduplication(deduplicate=deduplicate)

    def explain(self) -> DenseExplanationArray[Array]:
        """Estimate the configured index from the sampled evidence.

        Returns:
            A dense explanation whose attributions are unbiased estimates of
            the index over all completed sampled units. Indices that declare
            an order-0 attribution outside their functional carry the
            empty-coalition value there. Pending samples of an unfinished
            unit are excluded.

        Raises:
            InsufficientSamplesError: If no sampled unit has completed.
        """
        if not isinstance(self.state, SamplingState):
            self._require_no_evidence_yet()
        n_seeds = self.sampler.n_seed_samples
        pending = self.sampler.n_pending_samples
        usable = self.state.n_samples - pending
        if usable - n_seeds < 1:
            msg = (
                "explaining requires at least one completed sampled unit: "
                f"sample at least {self.min_budget} evaluations in total "
                f"(currently {self.state.n_samples} stored, {pending} pending)"
            )
            raise InsufficientSamplesError(msg)
        n_players = self.game.n_players
        n_value_axes = len(self.game.value_shape)
        masks = jnp.asarray(self.state.coalitions.to_dense())[..., :usable, :]
        values = to_leading(jnp.asarray(self.state.values), n_value_axes)[..., :usable]
        attributions = self._estimated_attributions(masks, values, n_seeds)
        if isinstance(self.index, AggregationIndex):
            attributions = aggregate_supersets(
                attributions,
                self.index.aggregation_coefficients(),
                n_players,
            )
        if self.index.includes_empty_interaction and 0 not in attributions:
            attributions[0] = values[..., :1]
        return DenseExplanationArray(
            attributions_by_order={
                size: to_trailing(block, n_value_axes) for size, block in attributions.items()
            },
            n_players=n_players,
            interaction_index=self.interaction_index,
            order=self.order,
            shape=self.game.target_shape,
            orientation=self.orientation,
            value_shape=self.game.value_shape,
        )

    def _estimated_attributions(
        self,
        masks: Array,
        values: Array,
        n_seeds: int,
    ) -> dict[int, Array]:
        """Estimate every represented size from seeds plus weighted samples."""
        functional = self._functional
        sampler = self.sampler
        if not isinstance(sampler, CoalitionSizeSampler):  # pragma: no cover - fixed by __init__
            msg = "MonteCarlo requires its derived CoalitionSizeSampler"
            raise TypeError(msg)
        n_players = self.game.n_players
        seed_masks = masks[..., :n_seeds, :]
        seed_values = values[..., :n_seeds]
        sampled_masks = masks[..., n_seeds:, :]
        sampled_values = values[..., n_seeds:]
        n_sampled = sampled_masks.shape[-2]
        sizes = jnp.sum(sampled_masks, axis=-1).astype(jnp.int32)
        n_size_coalitions = jnp.asarray([comb(n_players, k) for k in range(n_players + 1)])
        inverse_probability = n_size_coalitions[sizes] / sampler.size_probabilities[sizes]
        weighted_values = sampled_values * inverse_probability / n_sampled
        return {
            size: jnp.einsum(
                "...c,...ci->...i",
                seed_values,
                functional.coefficient_matrix(seed_masks, size),
            )
            + jnp.einsum(
                "...c,...ci->...i",
                weighted_values,
                functional.coefficient_matrix(sampled_masks, size),
            )
            for size in functional.interaction_sizes
        }
