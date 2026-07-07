from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from shapiq.errors import InsufficientSamplesError
from shapiq.explainers._evidence import EvidenceApproximator
from shapiq.explainers._faithful import (
    eliminate_constraint,
    interaction_design,
    require_identification,
    solve_faithful,
)
from shapiq.explanations import DenseExplanationArray
from shapiq.interactions import FSII
from shapiq.sampling import EmptyState, SamplingState, ShapleyKernelSampler

if TYPE_CHECKING:
    from shapiq.games import Game
    from shapiq.sampling import ShareSamples


class RegressionFSII(EvidenceApproximator):
    """Faithful Shapley interaction approximator based on kernel regression.

    The faithful Shapley interaction index of order ``k`` is the best
    ``k``-additive approximation of the game under the Shapley kernel, with
    the empty and grand coalition fit exactly. The approximator estimates the
    kernel objective by Monte Carlo: the sampler draws coalitions with
    probability proportional to their kernel weight, so every sampled
    coalition enters the least squares problem with unit weight, and repeated
    coalitions contribute through their multiplicity. ``explain()`` solves
    the accumulated regression exactly, substituting the empty- and
    grand-coalition constraints out of the system, which keeps the solve well
    conditioned in float32.

    ``explain()`` requires the sampled coalitions to identify all
    coefficients and raises ``InsufficientSamplesError`` while the regression
    design is rank deficient; ``deduplicate=True`` reaches identification
    with the fewest evaluations. For a game that is itself ``k``-additive
    the estimate is exact from identification onward. With ``order=1`` the
    estimate converges to the Shapley values (KernelSHAP).

    Example:
        >>> approximator = RegressionFSII(game, order=2, random_state=0)
        >>> explanation = approximator.sample(500).explain()
        >>> pair_interaction = explanation((0, 1))
    """

    def __init__(
        self,
        game: Game[Array],
        *,
        order: int = 2,
        random_state: Array | int = 0,
        share_samples: ShareSamples = False,
        paired: bool = True,
        track_history: bool = False,
        deduplicate: bool = False,
    ) -> None:
        """Initialize without evaluating the game.

        Args:
            game: Game to explain. Must produce scalar values per coalition
                and have at least two players.
            order: Maximum interaction order of the faithful approximation;
                the explanation represents orders one through ``order``.
            random_state: Integer seed or JAX PRNG key for drawing
                coalitions.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes. ``False`` samples independently per
                target; ``True`` shares across all target axes; an integer or
                tuple of integers shares across the selected axes.
            paired: Whether every sampled coalition is accompanied by its
                complement, which reduces estimation variance.
            track_history: Whether to record value-equivalent history for
                rollback and convergence analysis.
            deduplicate: Whether to evaluate each distinct coalition at most
                once; repeats reuse stored values and only novel evaluations
                count toward the budget. Requires shared samples.

        Raises:
            ValueError: If the game has fewer than two players, if ``order``
                is out of range, or if ``deduplicate`` is enabled without
                samples shared across explanation targets.
        """
        index = FSII(order=order)
        sampler = ShapleyKernelSampler(
            game.n_players,
            game.target_shape,
            share_samples=share_samples,
            paired=paired,
            random_state=random_state,
        )
        state = EmptyState(track_history=track_history)
        super().__init__(game, sampler, state, index=index)
        self._init_deduplication(deduplicate=deduplicate)

    @property
    def min_budget(self) -> int:
        """Return the smallest total budget after which ``explain()`` can work.

        Identification of the eliminated regression needs one fewer
        independent evidence row than interaction columns, on top of the
        seed block.
        """
        n_columns = sum(comb(self.game.n_players, size) for size in range(1, self.order + 1))
        return max(super().min_budget, self.sampler.n_seed_samples + n_columns - 1)

    def explain(self) -> DenseExplanationArray[Array]:
        """Solve the faithful regression on the sampled evidence.

        Returns:
            A dense explanation representing orders zero through ``order``.
            The empty interaction carries the empty-coalition value; higher
            orders hold the solution of the constrained least squares problem
            over all completed sampled units. Pending samples of an
            unfinished unit are excluded.

        Raises:
            InsufficientSamplesError: If no sampled unit has completed, or if
                the sampled coalitions do not yet identify all coefficients.
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
        target_shape = self.game.target_shape
        masks = jnp.asarray(self.state.coalitions.to_dense())[..., :usable, :]
        values = jnp.asarray(self.state.values)[..., :usable]
        value_empty = values[..., 0]
        value_grand = values[..., 1]
        n_rows = usable - n_seeds
        response = (values[..., n_seeds:] - value_empty[..., None]).reshape(-1, n_rows).T
        delta = (value_grand - value_empty).reshape(-1)
        sample_masks = masks[..., n_seeds:, :]
        flat_masks = sample_masks.reshape(-1, n_rows, n_players)
        if flat_masks.shape[0] == 1:
            reduced, pivot = eliminate_constraint(interaction_design(flat_masks[0], self.order))
            require_identification(reduced)
            solutions = solve_faithful(reduced, pivot, response, delta).T
        else:
            broadcast_masks = jnp.broadcast_to(
                sample_masks,
                (*target_shape, n_rows, n_players),
            ).reshape(-1, n_rows, n_players)
            columns = []
            for target in range(response.shape[-1]):
                reduced, pivot = eliminate_constraint(
                    interaction_design(broadcast_masks[target], self.order),
                )
                require_identification(reduced)
                columns.append(
                    solve_faithful(
                        reduced,
                        pivot,
                        response[:, target : target + 1],
                        delta[target : target + 1],
                    ),
                )
            solutions = jnp.concatenate(columns, axis=1).T
        attributions: dict[int, Array] = {0: value_empty[..., None]}
        offset = 0
        for size in range(1, self.order + 1):
            n_interactions = comb(n_players, size)
            block = solutions[:, offset : offset + n_interactions]
            attributions[size] = block.reshape(*target_shape, n_interactions)
            offset += n_interactions
        return DenseExplanationArray(
            attributions_by_order=attributions,
            n_players=n_players,
            interaction_index=self.interaction_index,
            order=self.order,
            shape=target_shape,
        )
