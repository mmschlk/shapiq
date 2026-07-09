from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from shapiq._shape import ensure_bool
from shapiq.errors import InsufficientSamplesError
from shapiq.explainers._base import reject_common_index_mistakes
from shapiq.explainers._evidence import EvidenceApproximator
from shapiq.explainers._faithful import (
    eliminate_constraint,
    interaction_design,
    require_identification,
    solve_faithful,
)
from shapiq.explainers._valueaxes import to_leading, to_trailing
from shapiq.explanations import DenseExplanationArray
from shapiq.interactions import FBII, FSII, SV, WeightedFBII
from shapiq.sampling import (
    BanzhafKernelSampler,
    EmptyState,
    PairedSampler,
    ProductKernelSampler,
    SamplingState,
    ShapleyKernelSampler,
    UnitScheduleSampler,
)

if TYPE_CHECKING:
    from shapiq.games import Game
    from shapiq.sampling import ShareSamples


class Regression(EvidenceApproximator):
    """Kernel-regression approximator dispatching on the interaction index.

    Each supported index is defined by a least squares fit under its kernel,
    and the sampler draws coalitions with probability proportional to that
    kernel, so every sampled coalition enters the fit with unit weight and
    repeated coalitions contribute through their multiplicity; support is
    therefore a closed set of indices whose kernel has a matching sampler.
    ``FSII(order=k)`` is the best ``k``-additive approximation of the game
    under the Shapley kernel with the empty and grand coalition fit exactly,
    and ``SV()`` is its order-1 special case (KernelSHAP); their constraints
    are substituted out of the system exactly, which keeps the solve well
    conditioned in float32. ``FBII(order=k)`` is the best ``k``-additive
    approximation under the uniform kernel, fit without constraints and with
    a free intercept as its order-0 attribution; its order-1 special case
    converges to the Banzhaf value. ``WeightedFBII(p, order=k)`` is the best
    ``k``-additive approximation under the product measure in which every
    player joins with probability ``p``, sampled by membership flips and fit
    like FBII; pairing is reserved for complement-symmetric kernels, so it
    switches off automatically at ``p != 0.5``.

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
        index: SV | FSII | FBII | WeightedFBII,
        *,
        random_state: Array | int = 0,
        share_samples: ShareSamples = False,
        paired: bool | None = None,
        track_history: bool = False,
        deduplicate: bool = False,
    ) -> None:
        """Initialize without evaluating the game.

        Args:
            game: Game to explain. Must have at least two players.
            index: The interaction index to estimate: ``SV()`` for Shapley
                values via KernelSHAP, ``FSII(order=k)`` for faithful
                Shapley interactions, ``FBII(order=k)`` for faithful
                Banzhaf interactions, or ``WeightedFBII(p, order=k)`` for
                faithful weighted Banzhaf interactions.
            random_state: Integer seed or JAX PRNG key for drawing
                coalitions.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes. ``False`` samples independently per
                target; ``True`` shares across all target axes; an integer or
                tuple of integers shares across the selected axes.
            paired: Whether every sampled coalition is accompanied by its
                complement, which reduces estimation variance. The default
                ``None`` pairs exactly when the index's kernel is
                complement-symmetric: always except for ``WeightedFBII``
                with ``p != 0.5``, whose complements would enter the fit
                with the wrong implicit weighting.
            track_history: Whether to record value-equivalent history for
                rollback and convergence analysis.
            deduplicate: Whether to evaluate each distinct coalition at most
                once; repeats reuse stored values and only novel evaluations
                count toward the budget. Requires shared samples.

        Raises:
            TypeError: If the index has no kernel-matched regression
                estimator.
            ValueError: If the game has fewer than two players, if the order
                is out of range, if ``paired=True`` is requested for a
                kernel that is not complement-symmetric, or if
                ``deduplicate`` is enabled without samples shared across
                explanation targets.
        """
        reject_common_index_mistakes(index)
        if type(index) not in (SV, FSII, FBII, WeightedFBII):
            name = getattr(index, "name", type(index).__name__)
            if isinstance(index, (SV, FSII, FBII, WeightedFBII)):
                msg = (
                    f"Regression dispatches on the exact index type: {type(index).__name__} "
                    "subclasses a supported index, but the sampler and solver are "
                    "kernel-matched to the shipped type; pass SV(), FSII(order=k), "
                    "FBII(order=k), or WeightedFBII(p, order=k) itself"
                )
                raise TypeError(msg)
            msg = (
                f"Regression does not support {name!r}: each supported index "
                "samples coalitions from its own kernel, and matching samplers "
                "exist for SV(), FSII(order=k), FBII(order=k), and "
                "WeightedFBII(p, order=k)"
            )
            raise TypeError(msg)
        base_sampler: UnitScheduleSampler
        symmetric_kernel = True
        if type(index) is WeightedFBII:
            symmetric_kernel = index.p == 0.5
            base_sampler = ProductKernelSampler(
                game.n_players,
                index.p,
                game.target_shape,
                share_samples=share_samples,
                random_state=random_state,
            )
        else:
            sampler_type = BanzhafKernelSampler if type(index) is FBII else ShapleyKernelSampler
            base_sampler = sampler_type(
                game.n_players,
                game.target_shape,
                share_samples=share_samples,
                random_state=random_state,
            )
        if paired is None:
            paired = symmetric_kernel
        elif ensure_bool("paired", paired) and not symmetric_kernel:
            msg = (
                "paired sampling adds the complement of every sampled coalition, but "
                "the product measure at p != 0.5 is not complement-symmetric, so the "
                "complements would enter the unweighted fit with the wrong implicit "
                "weighting; pass paired=False (or use p=0.5)"
            )
            raise ValueError(msg)
        sampler = PairedSampler(base_sampler) if paired else base_sampler
        state = EmptyState(track_history=track_history)
        super().__init__(game, sampler, state, index=index)
        self._init_deduplication(deduplicate=deduplicate)

    @property
    def min_budget(self) -> int:
        """Return the smallest total budget after which ``explain()`` can work.

        Identification needs at least as many independent evidence rows as
        free coefficients, on top of the seed block: one fewer than the
        interaction columns for the constrained Shapley fits, one more for
        the unconstrained Banzhaf fit with its free intercept.
        """
        n_columns = sum(comb(self.game.n_players, size) for size in range(1, self.order + 1))
        if type(self.index) in (FBII, WeightedFBII):
            return max(super().min_budget, self.sampler.n_seed_samples + n_columns + 1)
        return max(super().min_budget, self.sampler.n_seed_samples + n_columns - 1)

    def _solve(self, masks: Array, response: Array, delta: Array) -> Array:
        """Solve one design's least squares fit per the index's kernel family."""
        design = interaction_design(masks, self.order)
        if type(self.index) in (FBII, WeightedFBII):
            design = jnp.concatenate([jnp.ones((design.shape[0], 1)), design], axis=-1)
            require_identification(design, deduplicating=self.deduplicate)
            solution, *_ = jnp.linalg.lstsq(design, response)
            return solution
        reduced, pivot = eliminate_constraint(design)
        require_identification(reduced, deduplicating=self.deduplicate)
        return solve_faithful(reduced, pivot, response, delta)

    def explain(self) -> DenseExplanationArray[Array]:
        """Solve the kernel regression on the sampled evidence.

        Returns:
            A dense explanation whose baseline is the empty-coalition value.
            Attributions hold the solution of the kernel least squares
            problem on the centered game over all completed sampled units;
            FBII and WeightedFBII additionally carry their fitted intercept
            at order zero.
            Pending samples of an unfinished unit are excluded.

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
        if type(self.index) in (FBII, WeightedFBII):
            intercept = coefficients[:, :1].reshape(*value_shape, *target_shape, 1)
            coefficients = coefficients[:, 1:]
            attributions: dict[int, Array] = {0: to_trailing(intercept, n_value_axes)}
        else:
            attributions = {}
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
            index=self.index,
            order=self.order,
            shape=target_shape,
            value_shape=value_shape,
            baseline=to_trailing(value_empty, n_value_axes),
        )
