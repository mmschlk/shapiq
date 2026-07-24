from __future__ import annotations

from functools import singledispatch
from math import comb
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import jax.numpy as jnp
from jax import Array

from shapiq._shape import ensure_bool
from shapiq.errors import InsufficientSamplesError
from shapiq.explainers._base import reject_common_index_mistakes
from shapiq.explainers._faithful import (
    bernoulli_design,
    eliminate_constraint,
    interaction_design,
    lstsq_identified,
    solve_faithful,
)
from shapiq.explainers._valueaxes import to_trailing
from shapiq.explainers.approximators._base import Approximator
from shapiq.explanations import DenseExplanationArray
from shapiq.interactions import FBII, FSII, KADDSHAP, SV, WeightedFBII
from shapiq.sampling import (
    BanzhafKernelSampler,
    CoalitionSampler,
    PairedSampler,
    ProductKernelSampler,
    SamplingState,
    ShapleyKernelSampler,
)

if TYPE_CHECKING:

    from shapiq.games import Game
    from shapiq.sampling import ShareSamples


class Regression(Approximator):
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
    switches off automatically at ``p != 0.5``. ``KADDSHAP(order=k)`` fits
    the ``k``-additive game in the Bernoulli interaction basis under the
    Shapley kernel, interpolating the grand coalition exactly, so its
    order-1 attributions remain Shapley values at every order.

    The sampler, pairing rule, intercept convention, and least squares
    solve travel together as a ``RegressionFamily``, single-dispatched on
    the index type via ``regression_family``. Registering a family for a
    new index type extends the method atomically (a library-internal
    mechanism), and subclasses of supported indices inherit their parent's
    complete family through the method resolution order — an experimenter's
    index riding a shipped kernel answers for its own semantics.

    ``explain()`` requires the sampled coalitions to identify all
    coefficients and raises ``InsufficientSamplesError`` while the
    regression design is rank deficient; ``deduplicate=True`` reaches
    identification with the fewest evaluations. For a game that is itself
    ``k``-additive the estimate is exact from identification onward.

    Example:
        >>> approximator = Regression(game, FSII(order=2), random_state=0)
        >>> estimate = approximator.estimate(500)
        >>> pair_interaction = explanation((0, 1))
    """

    _family: RegressionFamily

    def __init__(
        self,
        game: Game[Array],
        index: SV | FSII | FBII | WeightedFBII | KADDSHAP,
        *,
        random_state: Array | int = 0,
        share_samples: ShareSamples = False,
        paired: bool | None = None,
        deduplicate: bool = False,
    ) -> None:
        """Initialize without evaluating the game.

        Args:
            game: Game to explain. Must have at least two players.
            index: The interaction index to estimate: ``SV()`` for Shapley
                values via KernelSHAP, ``FSII(order=k)`` for faithful
                Shapley interactions, ``FBII(order=k)`` for faithful
                Banzhaf interactions, ``WeightedFBII(p, order=k)`` for
                faithful weighted Banzhaf interactions, or
                ``KADDSHAP(order=k)`` for k-additive Shapley values.
            random_state: Integer seed or JAX PRNG key for drawing
                coalitions.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes. ``False`` samples independently per
                target; ``True`` shares across all target axes; an integer or
                tuple of integers shares across the selected axes.
            paired: Whether every sampled coalition is accompanied by its
                complement, which reduces estimation variance. The default
                ``None`` resolves to the family default: paired exactly when
                the index's kernel is complement-symmetric — always except
                for ``WeightedFBII`` with ``p != 0.5``, whose complements
                would enter the fit with the wrong implicit weighting.
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
        family = regression_family(index)
        base_sampler = family.build_sampler(
            index,
            game,
            share_samples=share_samples,
            random_state=random_state,
        )
        if paired is None:
            paired = family.symmetric_kernel
        elif ensure_bool("paired", paired) and not family.symmetric_kernel:
            msg = (
                "paired sampling adds the complement of every sampled coalition, but "
                f"the sampling kernel of {type(index).__name__} is not "
                "complement-symmetric, so the complements would enter the unweighted "
                "fit with the wrong implicit weighting; pass paired=False"
            )
            raise ValueError(msg)
        sampler = PairedSampler(base_sampler) if paired else base_sampler
        super().__init__(
            game,
            sampler,
            index,
            render=_coalition_rows,
            unit_length=1,
            deduplicate=deduplicate,
        )
        self._family = family

    @property
    def min_budget(self) -> int:
        """Return the floor below which ``explain()`` cannot succeed.

        Identification needs at least as many independent evidence rows as
        free coefficients, on top of the seed block: one fewer than the
        interaction columns for the constrained Shapley fits, one more for
        the unconstrained Banzhaf fit with its free intercept. Reaching the
        floor does not guarantee identification — that depends on the drawn
        coalitions (repeated coalitions add no rank, and the kADD-SHAP
        Bernoulli basis typically identifies later than the floor) — and
        ``explain()`` raises with the rank shortfall until they identify.
        """
        n_columns = sum(comb(self.game.n_players, size) for size in range(1, self.order + 1))
        offset = 1 if self._family.intercept else -1
        return max(super().min_budget, self.n_seed_samples + n_columns + offset)

    def _solve(self, masks: Array, response: Array, delta: Array) -> Array:
        """Solve one design's least squares fit per the index's kernel family."""
        return self._family.solve(
            masks,
            response,
            delta,
            order=self.order,
            deduplicating=self.deduplicate,
        )

    def _view(self) -> DenseExplanationArray[Array]:
        """Solve the kernel regression on the sampled evidence.

        Returns:
            A dense explanation whose baseline is the empty-coalition value.
            Attributions hold the solution of the kernel least squares
            problem on the centered game over all sampled units; FBII and
            WeightedFBII additionally carry their fitted intercept at order
            zero.

        Raises:
            InsufficientSamplesError: If no sampled unit has completed, or if
                the sampled coalitions do not yet identify all coefficients.
        """
        if not isinstance(self.state, SamplingState):
            self._require_no_evidence_yet()
        n_seeds = self.n_seed_samples
        # whole-unit spending guarantees every stored row is usable evidence
        usable = self.state.n_samples
        if usable - n_seeds < 1:
            msg = (
                "explaining requires at least one completed sampled unit: "
                f"estimate with at least {self.min_budget} evaluations in total "
                f"(currently {usable} stored, {self.bank} banked)"
            )
            raise InsufficientSamplesError(msg)
        n_players = self.game.n_players
        target_shape = self.game.target_shape
        value_shape = self.game.value_shape
        n_value_axes = len(value_shape)
        masks = jnp.asarray(self.state.coalitions.to_dense())[..., :usable, :]
        values = jnp.asarray(self.state.values)[..., :usable]  # canonical: sample axis last
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
            # per-target solves stay sequential: LAPACK multithreads inside
            # each least squares solve, which beats one batched SVD on CPU
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
        if self._family.intercept:
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


class BuildSampler(Protocol):
    """Family callback building the kernel-matched coalition sampler."""

    def __call__(
        self,
        index: Any,  # noqa: ANN401 - per-index builders narrow to their family's index
        game: Game[Array],
        *,
        share_samples: ShareSamples = False,
        random_state: Array | int = 0,
    ) -> CoalitionSampler:
        """Return the sampler drawing coalitions proportional to the kernel."""
        ...


class KernelSolve(Protocol):
    """Family callback solving one design's kernel least squares fit."""

    def __call__(
        self,
        masks: Array,
        response: Array,
        delta: Array,
        *,
        order: int,
        deduplicating: bool,
    ) -> Array:
        """Return the fitted coefficients, identification enforced."""
        ...


class RegressionFamily(NamedTuple):
    """Kernel-matched machinery of one regression index family.

    A family is registered atomically on ``regression_family``: the sampler
    matching the index's kernel, whether that kernel is complement-symmetric
    (pairing is variance-reducing only then), whether the fit carries a free
    intercept as its order-0 attribution (which also costs one extra
    identification row), and the least squares solve itself.
    """

    build_sampler: BuildSampler
    symmetric_kernel: bool
    intercept: bool
    solve: KernelSolve


@singledispatch
def regression_family(index: object) -> RegressionFamily:
    """Return the kernel-regression family matching an interaction index.

    Sampler, pairing rule, intercept convention, and solve dispatch together
    on the index type; subclasses resolve to their parent's family through
    the MRO, and registering a family for a new index type extends
    ``Regression``. Unregistered indices raise the teaching error.
    """
    raise _unsupported_regression_index(index)


def _registered_regression_indices() -> tuple[type, ...]:
    """Return the index types with a registered regression family."""
    return tuple(kind for kind in regression_family.registry if kind is not object)


def _supported_regression_names() -> str:
    return ", ".join(sorted(kind.__name__ for kind in _registered_regression_indices()))


def _unsupported_regression_index(index: object) -> TypeError:
    name = getattr(index, "name", type(index).__name__)
    msg = (
        f"Regression does not support {name!r}: each supported index samples "
        f"coalitions from its own kernel, and families are registered for "
        f"{_supported_regression_names()} (e.g. FSII(order=2))"
    )
    return TypeError(msg)


@regression_family.register
def _shapley_family(index: SV | FSII) -> RegressionFamily:
    del index
    return RegressionFamily(
        build_sampler=_shapley_kernel_sampler,
        symmetric_kernel=True,
        intercept=False,
        solve=_constrained_solve,
    )


@regression_family.register
def _banzhaf_family(index: FBII) -> RegressionFamily:
    del index
    return RegressionFamily(
        build_sampler=_banzhaf_kernel_sampler,
        symmetric_kernel=True,
        intercept=True,
        solve=_free_intercept_solve,
    )


@regression_family.register
def _weighted_banzhaf_family(index: WeightedFBII) -> RegressionFamily:
    return RegressionFamily(
        build_sampler=_product_kernel_sampler,
        symmetric_kernel=index.p == 0.5,
        intercept=True,
        solve=_free_intercept_solve,
    )


@regression_family.register
def _kadd_family(index: KADDSHAP) -> RegressionFamily:
    del index
    return RegressionFamily(
        build_sampler=_shapley_kernel_sampler,
        symmetric_kernel=True,
        intercept=False,
        solve=_kadd_solve,
    )


def _shapley_kernel_sampler(
    index: object,
    game: Game[Array],
    *,
    share_samples: ShareSamples = False,
    random_state: Array | int = 0,
) -> ShapleyKernelSampler:
    del index
    return ShapleyKernelSampler(
        game.n_players,
        game.target_shape,
        share_samples=share_samples,
        random_state=random_state,
    )


def _banzhaf_kernel_sampler(
    index: object,
    game: Game[Array],
    *,
    share_samples: ShareSamples = False,
    random_state: Array | int = 0,
) -> BanzhafKernelSampler:
    del index
    return BanzhafKernelSampler(
        game.n_players,
        game.target_shape,
        share_samples=share_samples,
        random_state=random_state,
    )


def _product_kernel_sampler(
    index: WeightedFBII,
    game: Game[Array],
    *,
    share_samples: ShareSamples = False,
    random_state: Array | int = 0,
) -> ProductKernelSampler:
    return ProductKernelSampler(
        game.n_players,
        index.p,
        game.target_shape,
        share_samples=share_samples,
        random_state=random_state,
    )


def _constrained_solve(
    masks: Array,
    response: Array,
    delta: Array,
    *,
    order: int,
    deduplicating: bool,
) -> Array:
    """Fit the constrained Shapley-kernel regression (empty and grand exact)."""
    reduced, pivot = eliminate_constraint(interaction_design(masks, order))
    return solve_faithful(
        reduced,
        pivot,
        response,
        delta,
        identify=True,
        deduplicating=deduplicating,
    )


def _free_intercept_solve(
    masks: Array,
    response: Array,
    delta: Array,
    *,
    order: int,
    deduplicating: bool,
) -> Array:
    """Fit the unconstrained kernel regression with a free intercept row."""
    del delta
    design = interaction_design(masks, order)
    design = jnp.concatenate([jnp.ones((*design.shape[:-1], 1)), design], axis=-1)
    return lstsq_identified(design, response, deduplicating=deduplicating)


def _kadd_solve(
    masks: Array,
    response: Array,
    delta: Array,
    *,
    order: int,
    deduplicating: bool,
) -> Array:
    """Fit the Bernoulli-basis Shapley regression pinned at the grand coalition.

    The sampled twin of the exact kADD-SHAP solver: the Bernoulli design row
    of the grand coalition forms the constraint, one pivot column is
    substituted out against ``delta``, and the reduced system is fit
    unweighted because the Shapley kernel sampler already draws rows with
    the kernel's probabilities. The empty coalition's design row is zero, so
    the empty-shifted response interpolates it automatically.
    """
    n_players = masks.shape[-1]
    design = bernoulli_design(masks, order)
    constraint = bernoulli_design(jnp.ones((1, n_players)), order)[0]
    pivot_column = int(jnp.argmax(jnp.abs(constraint)))
    anchor = constraint[pivot_column]
    pivot = design[..., pivot_column : pivot_column + 1]
    reduced = jnp.delete(
        design - pivot * (constraint / anchor)[None, :],
        pivot_column,
        axis=-1,
    )
    shifted = response - (pivot / anchor) * delta[..., None, :]
    partial = lstsq_identified(reduced, shifted, deduplicating=deduplicating)
    others = jnp.delete(constraint, pivot_column)
    back_substituted = (delta[..., None, :] - others[None, :] @ partial) / anchor
    return jnp.insert(partial, pivot_column, back_substituted[..., 0, :], axis=-2)


def _coalition_rows(draws: Array) -> Array:
    """Enter drawn coalitions directly as single-row units."""
    return draws[..., None, :]
