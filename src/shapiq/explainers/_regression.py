from __future__ import annotations

from itertools import combinations
from math import comb
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import Array

from shapiq._shape import validate_int
from shapiq.errors import InsufficientSamplesError
from shapiq.explainers._evidence import EvidenceApproximator
from shapiq.explanations import DenseExplanationArray
from shapiq.sampling import EmptyState, SamplingState, ShapleyKernelSampler

if TYPE_CHECKING:
    from shapiq.games import Game
    from shapiq.sampling import ShareSamples

_CONSTRAINT_WEIGHT = 1e7


class RegressionFSII(EvidenceApproximator):
    """Faithful Shapley interaction approximator based on kernel regression.

    The faithful Shapley interaction index of order ``k`` is the best
    ``k``-additive approximation of the game under the Shapley kernel, with
    the empty and grand coalition fit exactly. The approximator estimates the
    kernel objective by Monte Carlo: the sampler draws coalitions with
    probability proportional to their kernel weight, so every sampled
    coalition enters an ordinary least squares problem with unit weight, and
    repeated coalitions contribute through their multiplicity. ``explain()``
    solves the accumulated regression exactly.

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
        sampler = ShapleyKernelSampler(
            game.n_players,
            game.target_shape,
            share_samples=share_samples,
            paired=paired,
            random_state=random_state,
        )
        validate_int("order", order, minimum=1)
        state = EmptyState(track_history=track_history)
        super().__init__(game, sampler, state, interaction_index="FSII", order=order)
        self._init_deduplication(deduplicate=deduplicate)

    @property
    def min_budget(self) -> int:
        """Return the smallest total budget after which ``explain()`` can work.

        Identification needs at least as many evidence rows as regression
        columns, so the seed-plus-one-unit minimum is raised accordingly.
        """
        n_columns = 1 + sum(
            comb(self.game.n_players, size) for size in range(1, self.order + 1)
        )
        return max(super().min_budget, n_columns)

    def explain(self) -> DenseExplanationArray[Array]:
        """Solve the faithful regression on the sampled evidence.

        Returns:
            A dense explanation representing orders zero through ``order``.
            The empty interaction carries the empty-coalition value; higher
            orders hold the solution of the weighted least squares problem
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
        masks = np.asarray(jnp.asarray(self.state.coalitions.to_dense()))[..., :usable, :]
        values = np.asarray(jnp.asarray(self.state.values), dtype=np.float64)[..., :usable]
        value_empty = values[..., 0]
        response = (values - value_empty[..., None]).reshape(-1, usable)
        row_weights = np.ones(usable)
        # the constraint rows must outweigh the growing evidence, or the
        # solution drifts to the unconstrained fit as samples accumulate
        row_weights[:n_seeds] = _CONSTRAINT_WEIGHT * usable
        sqrt_weights = np.sqrt(row_weights)
        n_columns = 1 + sum(comb(n_players, size) for size in range(1, self.order + 1))
        flat_masks = masks.reshape(-1, usable, n_players)
        if flat_masks.shape[0] == 1:
            design = _design_matrix(flat_masks[0], self.order)
            _require_identification(design, n_columns)
            solution, *_ = np.linalg.lstsq(
                sqrt_weights[:, None] * design,
                sqrt_weights[:, None] * response.T,
                rcond=None,
            )
            solutions = solution.T
        else:
            broadcast_masks = np.broadcast_to(
                masks,
                (*target_shape, usable, n_players),
            ).reshape(-1, usable, n_players)
            solutions = np.zeros((response.shape[0], n_columns))
            for target in range(response.shape[0]):
                design = _design_matrix(broadcast_masks[target], self.order)
                _require_identification(design, n_columns)
                solutions[target], *_ = np.linalg.lstsq(
                    sqrt_weights[:, None] * design,
                    sqrt_weights * response[target],
                    rcond=None,
                )
        attributions: dict[int, Array] = {0: jnp.asarray(value_empty[..., None])}
        offset = 1  # skip the empty-interaction column
        for size in range(1, self.order + 1):
            n_interactions = comb(n_players, size)
            block = solutions[:, offset : offset + n_interactions]
            attributions[size] = jnp.asarray(block.reshape(*target_shape, n_interactions))
            offset += n_interactions
        return DenseExplanationArray(
            attributions_by_order=attributions,
            n_players=n_players,
            interaction_index="FSII",
            order=self.order,
            shape=target_shape,
        )


def _require_identification(design: np.ndarray, n_columns: int) -> None:
    """Raise when the sampled coalitions do not yet identify all coefficients."""
    rank = int(np.linalg.matrix_rank(design))
    if rank < n_columns:
        msg = (
            "the faithful regression is not yet identified: the sampled coalitions "
            f"give rank {rank} of the {n_columns} required; sample more evaluations "
            "(deduplicate=True reaches distinct coalitions with the fewest evaluations)"
        )
        raise InsufficientSamplesError(msg)


def _design_matrix(masks: np.ndarray, order: int) -> np.ndarray:
    """Return subset-membership columns for all interactions up to order."""
    n_players = masks.shape[-1]
    columns = [np.ones((masks.shape[0], 1))]
    for size in range(1, order + 1):
        member_masks = np.zeros((comb(n_players, size), n_players), dtype=bool)
        for row, members in enumerate(combinations(range(n_players), size)):
            member_masks[row, list(members)] = True
        intersections = masks.astype(np.int64) @ member_masks.T.astype(np.int64)
        columns.append((intersections == size).astype(np.float64))
    return np.concatenate(columns, axis=1)
