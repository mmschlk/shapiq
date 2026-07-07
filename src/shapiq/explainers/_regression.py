from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from shapiq.errors import InsufficientSamplesError
from shapiq.explainers._evidence import EvidenceApproximator
from shapiq.explainers._faithful import (
    bernoulli_design,
    eliminate_constraint,
    interaction_design,
    require_identification,
    solve_faithful,
    solve_pinned,
)
from shapiq.explainers._valueaxes import to_leading, to_trailing
from shapiq.explanations import DenseExplanationArray
from shapiq.interactions import KADDSHAP, RegressionIndex
from shapiq.sampling import CoalitionSizeSampler, EmptyState, SamplingState

if TYPE_CHECKING:
    from shapiq.games import Game
    from shapiq.sampling import ShareSamples


class Regression(EvidenceApproximator):
    """Kernel-regression approximator derived from the index's declared kernel.

    Any index declaring a regression kernel is supported through the
    capability alone: the sampler draws coalitions with probability
    proportional to the declared kernel weight (size distribution
    ``kernel(size) * comb(n, size)``, members uniform within a size), so
    every sampled coalition enters the least squares problem with unit
    weight and repeated coalitions contribute through their multiplicity.
    The fit interpolates the empty and grand coalition exactly as
    constraints; shapiq ships ``SV()`` (KernelSHAP), ``FSII(order=k)``, and
    ``kADD-SHAP(order=k)`` — which dispatches to its Bernoulli basis — and
    indices defined through ``define_regression_index`` work the same way.
    ``explain()`` solves the accumulated regression exactly, substituting the
    constraints out of the system, which keeps the solve well conditioned in
    float32.

    ``explain()`` requires the sampled coalitions to identify all
    coefficients and raises ``InsufficientSamplesError`` while the
    regression design is rank deficient; ``deduplicate=True`` reaches
    identification with the fewest evaluations. For a game that is itself
    ``k``-additive the estimate is exact from identification onward.

    Example:
        >>> approximator = Regression(game, FSII(order=2), random_state=0)
        >>> explanation = approximator.sample(500).explain()
        >>> pair_interaction = explanation((0, 1))
    """

    def __init__(
        self,
        game: Game[Array],
        index: RegressionIndex,
        *,
        random_state: Array | int = 0,
        share_samples: ShareSamples = False,
        paired: bool = True,
        track_history: bool = False,
        deduplicate: bool = False,
    ) -> None:
        """Initialize without evaluating the game.

        Args:
            game: Game to explain. Must have at least two players.
            index: The interaction index to estimate. Any index declaring a
                regression kernel works: ``SV()`` for Shapley values via
                KernelSHAP, ``FSII(order=k)`` for faithful Shapley
                interactions, ``KADDSHAP(order=k)`` for the k-additive
                Shapley fit, or a defined regression index.
            random_state: Integer seed or JAX PRNG key for drawing
                coalitions.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes. ``False`` samples independently per
                target; ``True`` shares across all target axes; an integer or
                tuple of integers shares across the selected axes.
            paired: Whether every sampled coalition is accompanied by its
                complement, which reduces estimation variance. Requires a
                size-symmetric kernel, since pairing must not change the
                probability a coalition is sampled with.
            track_history: Whether to record value-equivalent history for
                rollback and convergence analysis.
            deduplicate: Whether to evaluate each distinct coalition at most
                once; repeats reuse stored values and only novel evaluations
                count toward the budget. Requires shared samples.

        Raises:
            TypeError: If the index declares no regression kernel.
            ValueError: If the game has fewer than two players, if the order
                is out of range, if the declared kernel violates the
                capability contract (nonnegative, finite, zero-weight
                endpoints), if ``paired`` is enabled with a size-asymmetric
                kernel, or if ``deduplicate`` is enabled without samples
                shared across explanation targets.
        """
        if not isinstance(index, RegressionIndex):
            name = getattr(index, "name", type(index).__name__)
            msg = (
                f"Regression does not support {name!r}: the index declares no "
                "regression kernel with exact endpoint constraints"
            )
            raise TypeError(msg)
        kernel = _checked_kernel(index, game.n_players)
        if paired and not bool(jnp.allclose(kernel, kernel[::-1])):
            msg = (
                f"paired sampling requires a size-symmetric kernel, but {index.name!r} "
                "declares an asymmetric one; pass paired=False"
            )
            raise ValueError(msg)
        counts = jnp.asarray([comb(game.n_players, size) for size in range(game.n_players + 1)])
        sampler = CoalitionSizeSampler(
            game.n_players,
            game.target_shape,
            size_weights=kernel * counts,
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
        """Solve the constrained regression on the sampled evidence.

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
        value_shape = self.game.value_shape
        n_value_axes = len(value_shape)
        masks = jnp.asarray(self.state.coalitions.to_dense())[..., :usable, :]
        values = to_leading(jnp.asarray(self.state.values), n_value_axes)[..., :usable]
        value_empty = values[..., 0]
        value_grand = values[..., 1]
        n_rows = usable - n_seeds
        response = (values[..., n_seeds:] - value_empty[..., None]).reshape(-1, n_rows).T
        delta = (value_grand - value_empty).reshape(-1)
        sample_masks = masks[..., n_seeds:, :]
        flat_masks = sample_masks.reshape(-1, n_rows, n_players)
        if flat_masks.shape[0] == 1:
            solutions = self._solve(flat_masks[0], response, delta)
        else:
            broadcast_masks = jnp.broadcast_to(
                sample_masks,
                (*target_shape, n_rows, n_players),
            ).reshape(-1, n_rows, n_players)
            n_targets = broadcast_masks.shape[0]
            per_target = [
                self._solve(
                    broadcast_masks[target],
                    response[:, target::n_targets],
                    delta[target::n_targets],
                )
                for target in range(n_targets)
            ]
            stacked = jnp.stack(per_target, axis=-1)
            solutions = stacked.reshape(stacked.shape[0], -1)
        coefficients = solutions.T
        attributions: dict[int, Array] = {
            0: to_trailing(value_empty[..., None], n_value_axes),
        }
        offset = 0
        for size in range(1, self.order + 1):
            n_interactions = comb(n_players, size)
            block = coefficients[:, offset : offset + n_interactions]
            attributions[size] = to_trailing(
                block.reshape(*value_shape, *target_shape, n_interactions),
                n_value_axes,
            )
            offset += n_interactions
        return DenseExplanationArray(
            attributions_by_order=attributions,
            n_players=n_players,
            interaction_index=self.interaction_index,
            order=self.order,
            shape=target_shape,
            orientation=self.orientation,
            value_shape=value_shape,
        )

    def _solve(self, mask_rows: Array, response: Array, delta: Array) -> Array:
        """Solve the constrained fit of one target's evidence in the index basis."""
        if isinstance(self.index, KADDSHAP):
            design = bernoulli_design(mask_rows, self.order)
            grand = jnp.ones((1, self.game.n_players), dtype=bool)
            constraint = bernoulli_design(grand, self.order)[0]
            return solve_pinned(design, constraint, response, delta, require_identified=True)
        reduced, pivot = eliminate_constraint(interaction_design(mask_rows, self.order))
        require_identification(reduced)
        return solve_faithful(reduced, pivot, response, delta)


def _checked_kernel(index: RegressionIndex, n_players: int) -> Array:
    """Validate a declared regression kernel before sampling from it."""
    kernel = jnp.asarray(index.regression_kernel(n_players), dtype=jnp.float32)
    if kernel.shape != (n_players + 1,):
        msg = (
            f"{index.name!r} declares a regression kernel of shape {kernel.shape}, "
            f"expected one weight per coalition size 0..{n_players}, "
            f"shape ({n_players + 1},)"
        )
        raise ValueError(msg)
    if not bool(jnp.all(jnp.isfinite(kernel)) & jnp.all(kernel >= 0)):
        msg = f"{index.name!r} declares a regression kernel with negative or non-finite weights"
        raise ValueError(msg)
    if not bool(jnp.all(kernel[jnp.asarray([0, n_players])] == 0)):
        msg = (
            f"{index.name!r} declares nonzero kernel weight on the empty or grand "
            "coalition; the regression capability interpolates them exactly as "
            "constraints, so their kernel weight must be zero"
        )
        raise ValueError(msg)
    return kernel
