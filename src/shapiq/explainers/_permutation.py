from __future__ import annotations

import warnings
from math import comb
from typing import TYPE_CHECKING, NamedTuple, cast

import jax.numpy as jnp
import numpy as np
from jax import Array

from shapiq._shape import ensure_bool, logical_size, validate_int
from shapiq.coalitions import DenseCoalitionArray
from shapiq.errors import InsufficientSamplesError, SamplingStallWarning, UnsupportedGameError
from shapiq.explainers._approximator import Approximator
from shapiq.explanations import DenseExplanationArray
from shapiq.games import Game
from shapiq.sampling import (
    ApproximationState,
    EmptyState,
    PermutationSIISampler,
    PermutationSTIISampler,
    SamplingState,
)
from shapiq.sampling._permutation import (
    PermutationWalkSampler,
    interaction_members,
    nonempty_patterns,
    off_chain_patterns,
)

if TYPE_CHECKING:
    from shapiq.coalitions import CoalitionArray
    from shapiq.sampling import ShareSamples


class _WalkEvidence(NamedTuple):
    """Completed-walk view of a permutation sampling state."""

    n_walks: int
    seed_values: Array
    value_empty: Array
    value_grand: Array
    walk_masks: Array
    walk_values: Array


class _PermutationApproximator(
    Approximator[Array, Game[Array], ApproximationState, PermutationWalkSampler],
):
    """Shared machinery for permutation-walk approximators.

    Approximators start from an empty state; the sampler emits its
    deterministic seed block (empty and grand coalition, plus index-specific
    evaluations) at the start of the first sampled budget, so construction
    never evaluates the game. Pending samples of an unfinished unit stay in
    the state but are masked by ``explain()``.

    With deduplication enabled, every distinct coalition is evaluated on the
    game at most once: repeated coalitions reuse stored values, only novel
    evaluations consume budget, and the extra walks are free. The estimate is
    identical to the non-deduplicated one over the same walks. If the sampler
    stops producing novel coalitions, the remaining budget stays unspent and
    a ``SamplingStallWarning`` is issued.
    """

    deduplicate: bool

    @property
    def min_budget(self) -> int:
        """Return the smallest total budget after which ``explain()`` works.

        The first explanation needs the seed block plus one completed walk;
        this property saves users the arithmetic over the sampler's
        ``n_seed_samples`` and ``sampling_quantum``.
        """
        return self.sampler.n_seed_samples + self.sampler.sampling_quantum

    def __repr__(self) -> str:
        """Return a concise representation for the rebind workflow."""
        n_samples = getattr(self.state, "n_samples", 0)
        return (
            f"{type(self).__name__}(interaction_index={self.interaction_index!r}, "
            f"order={self.order!r}, n_samples={n_samples!r}, "
            f"n_pending_samples={self.sampler.n_pending_samples!r}, "
            f"deduplicate={self.deduplicate!r})"
        )

    def _init_deduplication(self, *, deduplicate: bool) -> None:
        """Validate and store the deduplication policy."""
        self.deduplicate = ensure_bool("deduplicate", deduplicate)
        if self.deduplicate and logical_size(self.sampler.shared_target_shape) != 1:
            msg = (
                "deduplicate=True requires the same coalitions to be sampled for every "
                "explanation target; pass share_samples=True (or share the selected axes)"
            )
            raise ValueError(msg)

    def sample(
        self,
        budget: int,
    ) -> Approximator[Array, Game[Array], ApproximationState, PermutationWalkSampler]:
        """Sample and evaluate additional coalitions, deduplicating if enabled.

        The budget is spent exactly, starting with the one-time seed block on
        the first call. Walks cut short by the budget stay pending and resume
        on the next call, so budgets may be split freely across calls without
        changing the sampled evidence.

        Args:
            budget: Number of new coalition evaluations to spend. With
                deduplication enabled, only novel coalitions count and
                repeated coalitions reuse stored values free of charge.

        Returns:
            A new approximator whose state includes the sampled evidence;
            this approximator is unchanged.

        Warns:
            SamplingStallWarning: If deduplication is enabled and the sampler
                stops producing novel coalitions before the budget is spent.
        """
        if not self.deduplicate:
            return super().sample(budget)
        validate_int("budget", budget)
        if budget == 0:
            return self
        return self._sample_deduplicated(budget)

    def _sample_deduplicated(
        self,
        budget: int,
    ) -> Approximator[Array, Game[Array], ApproximationState, PermutationWalkSampler]:
        """Spend budget on novel evaluations only, reusing stored values."""
        known = self._known_coalitions()
        sampler = self.sampler
        stall_limit = 10 * sampler.sampling_quantum
        chunks: list[Array] = []
        novel_positions: list[int] = []
        state_duplicates: list[tuple[int, int]] = []
        batch_duplicates: list[tuple[int, int]] = []
        batch_ranks: dict[bytes, int] = {}
        position = 0
        raw_since_novel = 0
        remaining = budget
        while remaining > 0:
            coalitions, sampler = sampler.sample(self.state, remaining)
            dense = jnp.asarray(coalitions.to_dense())
            chunks.append(dense)
            found = 0
            for key in _coalition_keys(dense):
                if key in batch_ranks:
                    batch_duplicates.append((position, batch_ranks[key]))
                elif key in known:
                    state_duplicates.append((position, known[key]))
                else:
                    batch_ranks[key] = len(novel_positions)
                    novel_positions.append(position)
                    found += 1
                position += 1
            remaining -= found
            raw_since_novel = 0 if found else raw_since_novel + int(dense.shape[-2])
            if remaining > 0 and raw_since_novel >= stall_limit:
                msg = (
                    f"sampling stopped after {budget - remaining} of {budget} requested "
                    "evaluations: no novel coalitions were found (a game with "
                    f"{self.game.n_players} players has at most 2**{self.game.n_players} "
                    "distinct coalitions); evidence gathered so far remains valid"
                )
                warnings.warn(msg, SamplingStallWarning, stacklevel=3)
                break
        masks = jnp.concatenate(chunks, axis=-2)
        values = self._stitch_values(masks, novel_positions, state_duplicates, batch_duplicates)
        next_state = self._append_state(DenseCoalitionArray(masks), values)
        next_history = self._next_sampler_history(sampler)
        return self._replace(state=next_state, sampler=sampler, sampler_history=next_history)

    def _known_coalitions(self) -> dict[bytes, int]:
        """Map every stored coalition to its first sample index."""
        if not isinstance(self.state, SamplingState):
            return {}
        dense = jnp.asarray(self.state.coalitions.to_dense())
        known: dict[bytes, int] = {}
        for index, key in enumerate(_coalition_keys(dense)):
            known.setdefault(key, index)
        return known

    def _stitch_values(
        self,
        masks: Array,
        novel_positions: list[int],
        state_duplicates: list[tuple[int, int]],
        batch_duplicates: list[tuple[int, int]],
    ) -> Array:
        """Evaluate novel coalitions and fill duplicates from stored values."""
        target_shape = self.game.target_shape
        novel_values: Array | None = None
        if novel_positions:
            novel_index = jnp.asarray(novel_positions)
            novel_values = jnp.asarray(
                self.game(DenseCoalitionArray(masks[..., novel_index, :])),
            )
            if novel_values.shape != (*target_shape, len(novel_positions)):
                msg = (
                    "permutation sampling requires scalar game values per coalition: "
                    f"expected shape {(*target_shape, len(novel_positions))}, "
                    f"got {novel_values.shape}"
                )
                raise UnsupportedGameError(msg)
        state_values = None
        if isinstance(self.state, SamplingState):
            state_values = jnp.asarray(cast("SamplingState[Array]", self.state).values)
        reference = novel_values if novel_values is not None else state_values
        if reference is None:  # unreachable: a first call always yields novel seeds
            msg = "deduplicated sampling produced neither novel nor stored values"
            raise RuntimeError(msg)
        values = jnp.zeros((*target_shape, int(masks.shape[-2])), dtype=reference.dtype)
        if novel_values is not None:
            values = values.at[..., jnp.asarray(novel_positions)].set(novel_values)
        if state_duplicates and state_values is not None:
            positions = jnp.asarray([position for position, _ in state_duplicates])
            sources = jnp.asarray([source for _, source in state_duplicates])
            values = values.at[..., positions].set(state_values[..., sources])
        if batch_duplicates and novel_values is not None:
            positions = jnp.asarray([position for position, _ in batch_duplicates])
            ranks = jnp.asarray([rank for _, rank in batch_duplicates])
            values = values.at[..., positions].set(novel_values[..., ranks])
        return values

    def _append_state(self, coalitions: CoalitionArray, values: Array) -> SamplingState[Array]:
        """Append sampled coalitions, creating the evidence state on first use."""
        checked_values = jnp.asarray(values)
        if checked_values.shape != (*self.game.target_shape, coalitions.shape[-1]):
            msg = (
                "permutation sampling requires scalar game values per coalition: "
                f"expected shape {(*self.game.target_shape, coalitions.shape[-1])}, "
                f"got {checked_values.shape}"
            )
            raise UnsupportedGameError(msg)
        if isinstance(self.state, SamplingState):
            return cast("SamplingState[Array]", self.state).append(coalitions, checked_values)
        return SamplingState(
            coalitions=coalitions,
            values=checked_values,
            target_shape=self.game.target_shape,
            track_history=self.state.track_history,
        )

    def _next_sampler_history(
        self,
        next_sampler: PermutationWalkSampler,
    ) -> tuple[PermutationWalkSampler, ...] | None:
        """Start sampler history at the first evidence state."""
        if self.state.track_history and not isinstance(self.state, SamplingState):
            return (next_sampler,)
        return super()._next_sampler_history(next_sampler)

    def _completed_walks(self) -> _WalkEvidence:
        """Return seed values and completed walks, masking pending samples."""
        quantum = self.sampler.sampling_quantum
        n_seeds = self.sampler.n_seed_samples
        if not isinstance(self.state, SamplingState):
            msg = (
                f"no samples yet: sample at least {self.min_budget} evaluations first; "
                "note that sample() returns a new approximator: "
                "`approximator = approximator.sample(budget)`"
            )
            raise InsufficientSamplesError(msg)
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
        values = jnp.asarray(self.state.values)
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


class PermutationSamplingSV(_PermutationApproximator):
    """Shapley-value approximator based on permutation walks.

    Each walk evaluates the proper prefixes of one random permutation and
    yields one marginal contribution per player. Because the empty and grand
    coalition anchor every completed walk, the order-1 attributions sum to
    ``v(N) - v(empty)`` exactly at any budget.

    Example:
        >>> approximator = PermutationSamplingSV(game, random_state=0)
        >>> approximator = approximator.sample(100)
        >>> explanation = approximator.explain()
        >>> shapley_value_of_player_0 = explanation((0,))
    """

    def __init__(
        self,
        game: Game[Array],
        *,
        random_state: Array | int = 0,
        share_samples: ShareSamples = False,
        track_history: bool = False,
        deduplicate: bool = False,
    ) -> None:
        """Initialize without evaluating the game.

        Args:
            game: Game to explain. Must produce scalar values per coalition
                and have at least two players.
            random_state: Integer seed or JAX PRNG key for drawing
                permutations.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes. ``False`` samples independently per
                target; ``True`` shares across all target axes; an integer or
                tuple of integers shares across the selected axes.
            track_history: Whether to record value-equivalent history for
                rollback and convergence analysis.
            deduplicate: Whether to evaluate each distinct coalition at most
                once; repeats reuse stored values and only novel evaluations
                count toward the budget. Requires shared samples.

        Raises:
            ValueError: If the game has fewer than two players, or if
                ``deduplicate`` is enabled without samples shared across
                explanation targets.
        """
        sampler = PermutationSIISampler(
            game.n_players,
            game.target_shape,
            share_samples=share_samples,
            order=1,
            random_state=random_state,
        )
        state = EmptyState(track_history=track_history)
        super().__init__(game, sampler, state, interaction_index="SV", order=1)
        self._init_deduplication(deduplicate=deduplicate)

    def explain(self) -> DenseExplanationArray[Array]:
        """Estimate Shapley values from completed permutation walks.

        Returns:
            A dense order-1 explanation. The empty interaction carries the
            empty-coalition value, and the order-1 attributions sum to
            ``v(N) - v(empty)`` exactly. Pending samples of an unfinished
            walk are excluded.

        Raises:
            InsufficientSamplesError: If no permutation walk has completed.
        """
        evidence = self._completed_walks()
        sums = _chain_marginal_sums(
            evidence.walk_masks,
            evidence.walk_values,
            evidence.value_empty,
            evidence.value_grand,
        )
        return DenseExplanationArray(
            attributions_by_order={
                0: evidence.value_empty[..., None],
                1: sums / evidence.n_walks,
            },
            n_players=self.game.n_players,
            interaction_index="SV",
            order=1,
            shape=self.game.target_shape,
        )


class PermutationSamplingSII(_PermutationApproximator):
    """Any-order Shapley interaction approximator based on permutation walks.

    Each walk yields one order-1 marginal per player from its prefix chain
    and one discrete derivative per consecutive window of every larger size,
    so higher-order sample counts are random. Explaining requires every
    represented interaction to have at least one sample, which may need many
    walks when ``comb(n_players, order)`` is large.

    Example:
        >>> approximator = PermutationSamplingSII(game, order=2, random_state=0)
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
        track_history: bool = False,
        deduplicate: bool = False,
    ) -> None:
        """Initialize without evaluating the game.

        Args:
            game: Game to explain. Must produce scalar values per coalition
                and have at least two players.
            order: Maximum interaction order to estimate; the explanation
                represents orders one through ``order``. Must satisfy
                ``1 <= order <= n_players``.
            random_state: Integer seed or JAX PRNG key for drawing
                permutations.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes. ``False`` samples independently per
                target; ``True`` shares across all target axes; an integer or
                tuple of integers shares across the selected axes.
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
        sampler = PermutationSIISampler(
            game.n_players,
            game.target_shape,
            share_samples=share_samples,
            order=order,
            random_state=random_state,
        )
        state = EmptyState(track_history=track_history)
        super().__init__(game, sampler, state, interaction_index="SII", order=order)
        self._init_deduplication(deduplicate=deduplicate)

    def explain(self) -> DenseExplanationArray[Array]:
        """Estimate interactions of all orders from completed walks.

        Returns:
            A dense explanation representing orders one through ``order``.
            Order-1 attributions come from the prefix chains; higher orders
            average discrete derivatives of consecutive permutation windows.
            Pending samples of an unfinished walk are excluded.

        Raises:
            InsufficientSamplesError: If no walk has completed, or if some
                represented interaction has no sample yet.
        """
        n_players = self.game.n_players
        evidence = self._completed_walks()
        chain_masks = evidence.walk_masks[..., : n_players - 1, :]
        chain_values = evidence.walk_values[..., : n_players - 1]
        order_one_sums = _chain_marginal_sums(
            chain_masks,
            chain_values,
            evidence.value_empty,
            evidence.value_grand,
        )
        attributions: dict[int, Array] = {1: order_one_sums / evidence.n_walks}
        if self.order >= 2:
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
            for size in range(2, self.order + 1):
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
                ranks = _interaction_ranks(window_members, n_players)
                onehots = ranks[..., None] == jnp.arange(comb(n_players, size))
                sums = jnp.sum(onehots * derivatives[..., None], axis=(-3, -2))
                counts = jnp.sum(onehots, axis=(-3, -2))
                if bool(jnp.any(counts == 0)):
                    missing = int(jnp.sum(counts == 0))
                    msg = (
                        f"an order-{self.order} SII explanation needs at least one sample "
                        f"for every interaction of each size up to {self.order}: "
                        f"{missing} of {int(counts.size)} size-{size} interaction estimates "
                        "have no sample yet; sample a larger budget "
                        f"(each walk yields {n_windows} size-{size} window samples)"
                    )
                    raise InsufficientSamplesError(msg)
                attributions[size] = sums / counts
        return DenseExplanationArray(
            attributions_by_order=attributions,
            n_players=n_players,
            interaction_index="SII",
            order=self.order,
            shape=self.game.target_shape,
        )


class PermutationSamplingSTII(_PermutationApproximator):
    """Any-order Shapley-Taylor approximator based on permutation walks.

    Interactions below the top order are computed exactly from seed
    evaluations of all lower-order coalitions, following the Shapley-Taylor
    definition as discrete derivatives at the empty coalition. Top-order
    interactions are sampled: every walk yields one discrete derivative for
    every top-order interaction, so sample counts are deterministic and a
    single completed walk covers all interactions. Note that the seed block
    and the walks both grow quickly with ``order``.

    Example:
        >>> approximator = PermutationSamplingSTII(game, order=2, random_state=0)
        >>> explanation = approximator.sample(500).explain()
        >>> pair_interaction = explanation((0, 1))
        >>> baseline = explanation(())  # the empty-coalition value
    """

    def __init__(
        self,
        game: Game[Array],
        *,
        order: int = 2,
        random_state: Array | int = 0,
        share_samples: ShareSamples = False,
        track_history: bool = False,
        deduplicate: bool = False,
    ) -> None:
        """Initialize without evaluating the game.

        Args:
            game: Game to explain. Must produce scalar values per coalition
                and have at least two players.
            order: Top interaction order. Orders below it are computed
                exactly from seed evaluations; order ``order`` is sampled.
                Must satisfy ``1 <= order <= n_players``.
            random_state: Integer seed or JAX PRNG key for drawing
                permutations.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes. ``False`` samples independently per
                target; ``True`` shares across all target axes; an integer or
                tuple of integers shares across the selected axes.
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
        sampler = PermutationSTIISampler(
            game.n_players,
            game.target_shape,
            share_samples=share_samples,
            order=order,
            random_state=random_state,
        )
        state = EmptyState(track_history=track_history)
        super().__init__(game, sampler, state, interaction_index="STII", order=order)
        self._init_deduplication(deduplicate=deduplicate)

    def explain(self) -> DenseExplanationArray[Array]:
        """Combine exact lower-order interactions with sampled top-order ones.

        Returns:
            A dense explanation representing orders zero through ``order``.
            The empty interaction carries the empty-coalition value, orders
            below the top order are exact discrete derivatives at the empty
            coalition, and top-order attributions average one sampled
            derivative per completed walk. Pending samples of an unfinished
            walk are excluded.

        Raises:
            InsufficientSamplesError: If no permutation walk has completed.
        """
        n_players = self.game.n_players
        top_order = self.order
        evidence = self._completed_walks()
        attributions: dict[int, Array] = {0: evidence.value_empty[..., None]}
        for size in range(1, top_order):
            attributions[size] = self._exact_empty_derivatives(evidence.seed_values, size)
        if top_order == 1:
            sums = _chain_marginal_sums(
                evidence.walk_masks,
                evidence.walk_values,
                evidence.value_empty,
                evidence.value_grand,
            )
            attributions[1] = sums / evidence.n_walks
        else:
            attributions[top_order] = self._sampled_top_order(evidence)
        return DenseExplanationArray(
            attributions_by_order=attributions,
            n_players=n_players,
            interaction_index="STII",
            order=top_order,
            shape=self.game.target_shape,
        )

    def _exact_empty_derivatives(self, seed_values: Array, size: int) -> Array:
        """Return discrete derivatives at the empty coalition for one order."""
        n_players = self.game.n_players
        members = interaction_members(n_players, size)
        derivatives = ((-1) ** size) * seed_values[..., :1]
        for pattern in nonempty_patterns(size):
            subset_members = members[:, jnp.asarray(pattern)]
            block_offset = 2 + sum(comb(n_players, s) for s in range(1, len(pattern)))
            indices = block_offset + _interaction_ranks(subset_members, n_players)
            sign = (-1) ** (size - len(pattern))
            derivatives = derivatives + sign * seed_values[..., indices]
        return derivatives

    def _sampled_top_order(self, evidence: _WalkEvidence) -> Array:
        """Average sampled top-order discrete derivatives over walks."""
        n_players = self.game.n_players
        top_order = self.order
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


def _coalition_keys(dense: Array) -> list[bytes]:
    """Return a hashable identity per sample-axis coalition (shared targets only)."""
    rows = np.asarray(dense).reshape(-1, dense.shape[-2], dense.shape[-1])[0]
    return [row.tobytes() for row in np.packbits(rows, axis=-1)]


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


def _recover_permutations(chain_masks: Array) -> Array:
    """Recover player orderings from stored prefix chains."""
    previous_masks = jnp.concatenate(
        [jnp.zeros_like(chain_masks[..., :1, :]), chain_masks[..., :-1, :]],
        axis=-2,
    )
    added_players = chain_masks & ~previous_masks
    last_players = ~chain_masks[..., -1:, :]
    return jnp.argmax(jnp.concatenate([added_players, last_players], axis=-2), axis=-1)


def _interaction_ranks(members: Array, n_players: int) -> Array:
    """Return lexicographic ranks of fixed-size interactions among all combinations."""
    size = members.shape[-1]
    ordered = jnp.sort(members, axis=-1)
    binomials = jnp.asarray(
        [[comb(row, column) for column in range(size + 1)] for row in range(n_players)],
    )
    terms = binomials[n_players - 1 - ordered, size - jnp.arange(size)]
    return comb(n_players, size) - 1 - jnp.sum(terms, axis=-1)
