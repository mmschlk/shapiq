from __future__ import annotations

from functools import singledispatch
from math import comb
from typing import TYPE_CHECKING, NamedTuple, cast

import jax.numpy as jnp
from jax import Array

from shapiq._shape import broadcast_shapes, ensure_bool
from shapiq.errors import InsufficientSamplesError
from shapiq.explainers._base import reject_common_index_mistakes
from shapiq.explainers._evidence import EvidenceApproximator
from shapiq.explainers._valueaxes import to_leading, to_trailing
from shapiq.explanations import DenseExplanationArray
from shapiq.interactions import SII, STII, SV
from shapiq.interactions._ranks import interaction_ranks
from shapiq.sampling import (
    EmptyState,
    PairedSampler,
    PermutationSIISampler,
    PermutationSTIISampler,
    PermutationWalkSampler,
    SamplingState,
)
from shapiq.sampling._permutation import (
    interaction_members,
    nonempty_patterns,
    off_chain_patterns,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.games import Game
    from shapiq.sampling import ShareSamples


class _WalkEvidence(NamedTuple):
    """Completed-walk view of a permutation sampling state."""

    n_walks: int
    seed_values: Array
    value_empty: Array
    value_grand: Array
    walk_masks: Array
    walk_values: Array


class PermutationSampling(EvidenceApproximator):
    """Permutation-walk approximator dispatching on the interaction index.

    The index object selects the concrete walk layout and estimator: ``SV()``
    evaluates the proper prefixes of random permutations and yields one
    marginal contribution per player and walk, so order-1 attributions sum to
    ``v(N) - v(empty)`` exactly at any budget. ``SII(order=k)`` additionally
    yields one discrete derivative per consecutive permutation window of
    every size up to ``k``, so higher-order sample counts are random and
    explaining requires every represented interaction to have at least one
    sample. ``STII(order=k)`` computes interactions below the top order
    exactly from seed evaluations and samples one top-order derivative per
    interaction and walk, so coverage is deterministic; its seed block and
    walks both grow quickly with the order.

    The walk layout and the estimator travel together as a
    ``PermutationFamily``, single-dispatched on the exact index type via
    ``permutation_family``. Registering a family for a new index type
    extends the method atomically; subclasses of supported indices stay
    rejected with a teaching error unless they register their own family,
    so MRO resolution never hands an index a shipped estimator silently.

    Example:
        >>> approximator = PermutationSampling(game, SII(order=2), random_state=0)
        >>> explanation = approximator.sample(500).explain()
        >>> pair_interaction = explanation((0, 1))
    """

    _family: PermutationFamily

    def __init__(
        self,
        game: Game[Array],
        index: SV | SII | STII,
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
            index: The interaction index to estimate: ``SV()``,
                ``SII(order=k)``, or ``STII(order=k)``.
            random_state: Integer seed or JAX PRNG key for drawing
                permutations.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes. ``False`` samples independently per
                target; ``True`` shares across all target axes; an integer or
                tuple of integers shares across the selected axes.
            paired: Whether every sampled unit also contains the walk of the
                reversed permutation (antithetic sampling), which reduces
                estimation variance and doubles the sampling quantum. The
                default ``None`` resolves to the family default: permutation
                walks are unpaired unless requested.
            track_history: Whether to record value-equivalent history for
                rollback and convergence analysis.
            deduplicate: Whether to evaluate each distinct coalition at most
                once; repeats reuse stored values and only novel evaluations
                count toward the budget. Requires shared samples.

        Raises:
            TypeError: If the index has no permutation-walk estimator.
            ValueError: If the game has fewer than two players, if the order
                is out of range, or if ``deduplicate`` is enabled without
                samples shared across explanation targets.
        """
        reject_common_index_mistakes(index)
        registered = _registered_permutation_indices()
        if type(index) not in registered and isinstance(index, registered):
            msg = (
                f"PermutationSampling dispatches on the exact index type: "
                f"{type(index).__name__} subclasses a supported index; "
                f"pass one of {_supported_permutation_names()} itself (e.g. SII(order=2))"
            )
            raise TypeError(msg)
        family = permutation_family(index)
        base_sampler = family.build_sampler(
            index,
            game,
            share_samples=share_samples,
            random_state=random_state,
        )
        if paired is None:
            paired = False  # every permutation family pairs by reversal, none by default
        else:
            ensure_bool("paired", paired)
        sampler = PairedSampler(base_sampler) if paired else base_sampler
        state = EmptyState(track_history=track_history)
        super().__init__(game, sampler, state, index=index)
        self._init_deduplication(deduplicate=deduplicate)
        self._family = family

    def explain(self) -> DenseExplanationArray[Array]:
        """Estimate the configured index from completed permutation walks.

        Returns:
            A dense explanation whose baseline is the empty-coalition value;
            attributions cover orders one and above. Pending samples of an
            unfinished walk are excluded.

        Raises:
            InsufficientSamplesError: If no permutation walk has completed,
                or, for SII, if some represented interaction has no sample
                yet.
        """
        return self._family.explain(self.index, self)

    def _completed_walks(self) -> _WalkEvidence:
        """Return seed values and completed walks, masking pending samples."""
        # paired units hold two walks (the permutation's and its reversal's),
        # so completed walks are cut by walk length, not by the quantum
        walker = cast("PermutationWalkSampler", self.sampler)
        quantum = walker.walk_length
        n_seeds = self.sampler.n_seed_samples
        if not isinstance(self.state, SamplingState):
            self._require_no_evidence_yet()
        n_walk_samples = self.state.n_samples - n_seeds - self.sampler.n_pending_samples
        n_walks = n_walk_samples // quantum
        if self.state.n_samples < n_seeds or n_walks < 1:
            msg = (
                "explaining requires at least one completed permutation walk: "
                f"sample at least {self.min_budget} evaluations in total "
                f"(currently {self.state.n_samples} stored, "
                f"{self.sampler.n_pending_samples} pending)"
            )
            raise InsufficientSamplesError(msg)
        coalitions = jnp.asarray(self.state.coalitions.to_dense())
        values = to_leading(jnp.asarray(self.state.values), len(self.game.value_shape))
        stop = n_seeds + n_walks * quantum
        walk_masks = jnp.reshape(
            coalitions[..., n_seeds:stop, :],
            (*coalitions.shape[:-2], n_walks, quantum, self.game.n_players),
        )
        walk_values = jnp.reshape(
            values[..., n_seeds:stop],
            (*values.shape[:-1], n_walks, quantum),
        )
        return _WalkEvidence(
            n_walks=n_walks,
            seed_values=values[..., :n_seeds],
            value_empty=values[..., 0],
            value_grand=values[..., 1],
            walk_masks=walk_masks,
            walk_values=walk_values,
        )


class PermutationFamily(NamedTuple):
    """Walk-sampler builder and estimator of one permutation index family.

    A family is registered atomically on ``permutation_family``, so an
    index either brings both pieces or neither; drift between the walk
    layout and its estimator is unrepresentable.
    """

    build_sampler: Callable[..., PermutationWalkSampler]
    explain: Callable[..., DenseExplanationArray[Array]]


@singledispatch
def permutation_family(index: object) -> PermutationFamily:
    """Return the permutation-walk family matching an interaction index.

    The walk layout and its estimator dispatch together on the exact index
    type; registering a family for a new index type extends
    ``PermutationSampling``. Unregistered indices raise the teaching error.
    """
    raise _unsupported_permutation_index(index)


def _registered_permutation_indices() -> tuple[type, ...]:
    """Return the index types with a registered permutation family."""
    return tuple(kind for kind in permutation_family.registry if kind is not object)


def _supported_permutation_names() -> str:
    return ", ".join(sorted(kind.__name__ for kind in _registered_permutation_indices()))


def _unsupported_permutation_index(index: object) -> TypeError:
    name = getattr(index, "name", type(index).__name__)
    msg = (
        f"PermutationSampling does not support {name!r}: supported "
        f"indices are {_supported_permutation_names()} (e.g. SII(order=2))"
    )
    return TypeError(msg)


@permutation_family.register
def _shapley_value_family(index: SV) -> PermutationFamily:
    del index
    return PermutationFamily(_walk_sampler, _explain_shapley_values)


@permutation_family.register
def _interaction_family(index: SII) -> PermutationFamily:
    del index
    return PermutationFamily(_walk_sampler, _explain_interactions)


@permutation_family.register
def _taylor_family(index: STII) -> PermutationFamily:
    del index
    return PermutationFamily(_taylor_sampler, _explain_taylor_interactions)


def _walk_sampler(
    index: SV | SII,
    game: Game[Array],
    *,
    share_samples: ShareSamples = False,
    random_state: Array | int = 0,
) -> PermutationSIISampler:
    return PermutationSIISampler(
        game.n_players,
        game.target_shape,
        share_samples=share_samples,
        order=index.order,
        random_state=random_state,
    )


def _taylor_sampler(
    index: STII,
    game: Game[Array],
    *,
    share_samples: ShareSamples = False,
    random_state: Array | int = 0,
) -> PermutationSTIISampler:
    return PermutationSTIISampler(
        game.n_players,
        game.target_shape,
        share_samples=share_samples,
        order=index.order,
        random_state=random_state,
    )


def _explain_shapley_values(
    index: SV,
    approximator: PermutationSampling,
) -> DenseExplanationArray[Array]:
    """Average per-player marginal contributions over completed chains."""
    evidence = approximator._completed_walks()  # noqa: SLF001
    sums = _chain_marginal_sums(
        evidence.walk_masks,
        evidence.walk_values,
        evidence.value_empty,
        evidence.value_grand,
    )
    n_value_axes = len(approximator.game.value_shape)
    return DenseExplanationArray(
        attributions_by_order={
            1: to_trailing(sums / evidence.n_walks, n_value_axes),
        },
        n_players=approximator.game.n_players,
        index=index,
        order=1,
        shape=approximator.game.target_shape,
        value_shape=approximator.game.value_shape,
        baseline=to_trailing(evidence.value_empty, n_value_axes),
    )


def _explain_interactions(
    index: SII,
    approximator: PermutationSampling,
) -> DenseExplanationArray[Array]:
    """Average chain marginals and windowed discrete derivatives per interaction."""
    n_players = approximator.game.n_players
    order = index.order
    evidence = approximator._completed_walks()  # noqa: SLF001
    chain_masks = evidence.walk_masks[..., : n_players - 1, :]
    chain_values = evidence.walk_values[..., : n_players - 1]
    order_one_sums = _chain_marginal_sums(
        chain_masks,
        chain_values,
        evidence.value_empty,
        evidence.value_grand,
    )
    attributions: dict[int, Array] = {1: order_one_sums / evidence.n_walks}
    if order >= 2:
        permutations = _recover_permutations(chain_masks)
        anchor_shape = (*chain_values.shape[:-1], 1)
        prefix_values = jnp.concatenate(
            [
                jnp.broadcast_to(evidence.value_empty[..., None, None], anchor_shape),
                chain_values,
                jnp.broadcast_to(evidence.value_grand[..., None, None], anchor_shape),
            ],
            axis=-1,
        )
        offset = n_players - 1
        for size in range(2, order + 1):
            patterns = off_chain_patterns(size)
            n_windows = n_players - size + 1
            off_values = jnp.reshape(
                evidence.walk_values[..., offset : offset + len(patterns) * n_windows],
                (*evidence.walk_values.shape[:-1], len(patterns), n_windows),
            )
            offset += len(patterns) * n_windows
            derivatives = jnp.zeros_like(prefix_values[..., :n_windows])
            for length in range(size + 1):
                sign = (-1) ** (size - length)
                derivatives = derivatives + sign * prefix_values[..., length : length + n_windows]
            for pattern_index, pattern in enumerate(patterns):
                sign = (-1) ** (size - len(pattern))
                derivatives = derivatives + sign * off_values[..., pattern_index, :]
            window_members = jnp.stack(
                [permutations[..., start : start + n_windows] for start in range(size)],
                axis=-1,
            )
            ranks = interaction_ranks(window_members, n_players)
            n_interactions = comb(n_players, size)
            sums = _scatter_by_rank(
                jnp.broadcast_to(
                    derivatives,
                    broadcast_shapes(ranks.shape, derivatives.shape),
                ),
                ranks,
                n_interactions,
            )
            counts = _scatter_by_rank(
                jnp.ones(ranks.shape, dtype=jnp.int32),
                ranks,
                n_interactions,
            )
            if bool(jnp.any(counts == 0)):
                missing = int(jnp.sum(counts == 0))
                walks_needed = -(-missing // n_windows)
                msg = (
                    f"an order-{order} SII explanation needs at least one sample "
                    f"for every interaction of each size up to {order}: "
                    f"{missing} of {int(counts.size)} size-{size} interaction estimates "
                    f"have no sample yet; sample at least {walks_needed} more completed "
                    f"walks (each walk yields {n_windows} size-{size} window samples, "
                    "and window coverage is random, so more may be needed)"
                )
                raise InsufficientSamplesError(msg)
            attributions[size] = sums / counts
    n_value_axes = len(approximator.game.value_shape)
    return DenseExplanationArray(
        attributions_by_order={
            size: to_trailing(block, n_value_axes) for size, block in attributions.items()
        },
        n_players=n_players,
        index=index,
        order=order,
        shape=approximator.game.target_shape,
        value_shape=approximator.game.value_shape,
        baseline=to_trailing(evidence.value_empty, n_value_axes),
    )


def _explain_taylor_interactions(
    index: STII,
    approximator: PermutationSampling,
) -> DenseExplanationArray[Array]:
    """Combine exact lower-order interactions with sampled top-order ones."""
    n_players = approximator.game.n_players
    top_order = index.order
    evidence = approximator._completed_walks()  # noqa: SLF001
    attributions: dict[int, Array] = {}
    for size in range(1, top_order):
        attributions[size] = _taylor_exact_empty_derivatives(
            approximator,
            evidence.seed_values,
            size,
        )
    if top_order == 1:
        sums = _chain_marginal_sums(
            evidence.walk_masks,
            evidence.walk_values,
            evidence.value_empty,
            evidence.value_grand,
        )
        attributions[1] = sums / evidence.n_walks
    else:
        attributions[top_order] = _taylor_sampled_top_order(index, approximator, evidence)
    n_value_axes = len(approximator.game.value_shape)
    return DenseExplanationArray(
        attributions_by_order={
            size: to_trailing(block, n_value_axes) for size, block in attributions.items()
        },
        n_players=n_players,
        index=index,
        order=top_order,
        shape=approximator.game.target_shape,
        value_shape=approximator.game.value_shape,
        baseline=to_trailing(evidence.value_empty, n_value_axes),
    )


def _taylor_exact_empty_derivatives(
    approximator: PermutationSampling,
    seed_values: Array,
    size: int,
) -> Array:
    """Return discrete derivatives at the empty coalition for one order."""
    n_players = approximator.game.n_players
    members = interaction_members(n_players, size)
    derivatives = ((-1) ** size) * seed_values[..., :1]
    for pattern in nonempty_patterns(size):
        subset_members = members[:, jnp.asarray(pattern)]
        block_offset = 2 + sum(comb(n_players, s) for s in range(1, len(pattern)))
        indices = block_offset + interaction_ranks(subset_members, n_players)
        sign = (-1) ** (size - len(pattern))
        derivatives = derivatives + sign * seed_values[..., indices]
    return derivatives


def _taylor_sampled_top_order(
    index: STII,
    approximator: PermutationSampling,
    evidence: _WalkEvidence,
) -> Array:
    """Average sampled top-order discrete derivatives over walks."""
    n_players = approximator.game.n_players
    top_order = index.order
    chain_length = n_players - top_order
    chain_masks = evidence.walk_masks[..., :chain_length, :]
    chain_values = evidence.walk_values[..., :chain_length]
    patterns = nonempty_patterns(top_order)
    n_interactions = comb(n_players, top_order)
    off_values = jnp.reshape(
        evidence.walk_values[..., chain_length:],
        (*evidence.walk_values.shape[:-1], len(patterns), n_interactions),
    )
    # a player's permutation position is recoverable from how many stored
    # prefixes contain it; players beyond the chain clamp to chain_length,
    # which is exactly the prefix their interactions see
    positions = chain_length - jnp.sum(chain_masks, axis=-2)
    members = interaction_members(n_players, top_order)
    first_positions = jnp.min(positions[..., members], axis=-1)
    prefix_values = jnp.concatenate(
        [
            jnp.broadcast_to(
                evidence.value_empty[..., None, None],
                (*chain_values.shape[:-1], 1),
            ),
            chain_values,
        ],
        axis=-1,
    )
    base_indices = jnp.broadcast_to(
        first_positions,
        (*prefix_values.shape[:-1], n_interactions),
    )
    derivatives = ((-1) ** top_order) * jnp.take_along_axis(
        prefix_values,
        base_indices,
        axis=-1,
    )
    for pattern_index, pattern in enumerate(patterns):
        sign = (-1) ** (top_order - len(pattern))
        derivatives = derivatives + sign * off_values[..., pattern_index, :]
    return jnp.mean(derivatives, axis=-2)


def _chain_marginal_sums(
    chain_masks: Array,
    chain_values: Array,
    value_empty: Array,
    value_grand: Array,
) -> Array:
    """Sum per-player marginal contributions over all prefix chains."""
    previous_masks = jnp.concatenate(
        [jnp.zeros_like(chain_masks[..., :1, :]), chain_masks[..., :-1, :]],
        axis=-2,
    )
    added_players = chain_masks & ~previous_masks
    previous_values = jnp.concatenate(
        [
            jnp.broadcast_to(value_empty[..., None, None], (*chain_values.shape[:-1], 1)),
            chain_values[..., :-1],
        ],
        axis=-1,
    )
    marginals = chain_values - previous_values
    sums = jnp.sum(added_players * marginals[..., None], axis=(-3, -2))
    last_marginals = value_grand[..., None] - chain_values[..., -1]
    return sums + jnp.sum(~chain_masks[..., -1, :] * last_marginals[..., None], axis=-2)


def _scatter_by_rank(values: Array, ranks: Array, n_interactions: int) -> Array:
    """Sum per-window values into per-interaction bins by lexicographic rank.

    The scatter-add avoids materializing a one-hot
    ``(walks, windows, interactions)`` tensor, whose size grows with the
    full interaction count times the sample count.
    """
    lead = values.shape[:-2]
    n_samples = values.shape[-2] * values.shape[-1]
    flat_values = values.reshape(-1, n_samples)
    flat_ranks = jnp.broadcast_to(ranks, values.shape).reshape(-1, n_samples)
    rows = jnp.arange(flat_values.shape[0])[:, None]
    binned = (
        jnp.zeros((flat_values.shape[0], n_interactions), dtype=values.dtype)
        .at[rows, flat_ranks]
        .add(flat_values)
    )
    return binned.reshape(*lead, n_interactions)


def _recover_permutations(chain_masks: Array) -> Array:
    """Recover player orderings from stored prefix chains."""
    previous_masks = jnp.concatenate(
        [jnp.zeros_like(chain_masks[..., :1, :]), chain_masks[..., :-1, :]],
        axis=-2,
    )
    added_players = chain_masks & ~previous_masks
    last_players = ~chain_masks[..., -1:, :]
    return jnp.argmax(jnp.concatenate([added_players, last_players], axis=-2), axis=-1)


