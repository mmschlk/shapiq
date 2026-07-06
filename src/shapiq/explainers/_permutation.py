from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING, NamedTuple, Self

import jax.numpy as jnp
from jax import Array

from shapiq.coalitions import DenseCoalitionArray
from shapiq.errors import InsufficientSamplesError, UnsupportedGameError
from shapiq.explainers._approximator import Approximator
from shapiq.explanations import DenseExplanationArray
from shapiq.games import Game
from shapiq.sampling import PermutationSIISampler, PermutationSTIISampler, SamplingState
from shapiq.sampling._permutation import (
    PermutationWalkSampler,
    interaction_members,
    nonempty_patterns,
    off_chain_patterns,
)

if TYPE_CHECKING:
    from shapiq.coalitions import CoalitionArray
    from shapiq.sampling import SampleSharing


class _WalkEvidence(NamedTuple):
    """Completed-walk view of a permutation sampling state."""

    n_walks: int
    seed_values: Array
    value_empty: Array
    value_grand: Array
    walk_masks: Array
    walk_values: Array


class _PermutationApproximator(
    Approximator[Array, Game[Array], SamplingState[Array], PermutationWalkSampler],
):
    """Shared machinery for permutation-walk approximators.

    The initial state holds seed evidence as its first samples: the empty and
    grand coalition, plus index-specific deterministic evaluations. Pending
    samples of an unfinished walk stay in the state but are masked by
    ``explain()``.
    """

    @property
    def _n_seed_samples(self) -> int:
        """Return the number of seed samples at the start of the state."""
        return 2

    def _append_state(self, coalitions: CoalitionArray, values: Array) -> SamplingState[Array]:
        """Append sampled walk coalitions to the sampling state."""
        return self.state.append(coalitions, values)

    def _completed_walks(self) -> _WalkEvidence:
        """Return seed values and completed walks, masking pending samples."""
        quantum = self.sampler.sampling_quantum
        n_seeds = self._n_seed_samples
        n_walk_samples = self.state.n_samples - n_seeds - self.sampler.n_pending
        n_walks = n_walk_samples // quantum
        if self.state.n_samples < n_seeds or n_walks < 1:
            msg = "explaining requires at least one completed permutation walk"
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

    Order-1 walks are plain prefix chains. Because the empty and grand
    coalition anchor every completed walk, the order-1 attributions sum to
    ``v(N) - v(empty)`` exactly.
    """

    def __init__(
        self,
        game: Game[Array],
        sampler: PermutationSIISampler,
        state: SamplingState[Array],
    ) -> None:
        """Initialize from an explicit game, sampler, and state."""
        super().__init__(game, sampler, state, interaction_index="SV", order=1)

    @classmethod
    def create(
        cls,
        game: Game[Array],
        *,
        key: Array | int = 0,
        sample_sharing: SampleSharing = None,
        track_history: bool = False,
    ) -> Self:
        """Create an approximator, evaluating the empty and grand coalition once."""
        sampler = PermutationSIISampler(
            game.n_players,
            game.target_shape,
            sample_sharing,
            order=1,
            key=key,
        )
        seeds = _base_seed_masks(game.n_players)
        return cls(game, sampler, _seeded_state(game, sampler, seeds, track_history=track_history))

    def explain(self) -> DenseExplanationArray[Array]:
        """Estimate Shapley values from completed permutation walks."""
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
    represented interaction to have at least one sample.
    """

    def __init__(
        self,
        game: Game[Array],
        sampler: PermutationSIISampler,
        state: SamplingState[Array],
    ) -> None:
        """Initialize from an explicit game, sampler, and state."""
        super().__init__(game, sampler, state, interaction_index="SII", order=sampler.order)

    @classmethod
    def create(
        cls,
        game: Game[Array],
        *,
        order: int = 2,
        key: Array | int = 0,
        sample_sharing: SampleSharing = None,
        track_history: bool = False,
    ) -> Self:
        """Create an approximator, evaluating the empty and grand coalition once."""
        sampler = PermutationSIISampler(
            game.n_players,
            game.target_shape,
            sample_sharing,
            order=order,
            key=key,
        )
        seeds = _base_seed_masks(game.n_players)
        return cls(game, sampler, _seeded_state(game, sampler, seeds, track_history=track_history))

    def explain(self) -> DenseExplanationArray[Array]:
        """Estimate interactions of all orders from completed walks."""
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
                    msg = f"explaining requires a sample for every order-{size} interaction"
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
    every top-order interaction, so sample counts are deterministic.
    """

    def __init__(
        self,
        game: Game[Array],
        sampler: PermutationSTIISampler,
        state: SamplingState[Array],
    ) -> None:
        """Initialize from an explicit game, sampler, and state."""
        super().__init__(game, sampler, state, interaction_index="STII", order=sampler.order)

    @property
    def _n_seed_samples(self) -> int:
        """Return the number of seed samples at the start of the state."""
        n_players = self.game.n_players
        return 2 + sum(comb(n_players, size) for size in range(1, self.order))

    @classmethod
    def create(
        cls,
        game: Game[Array],
        *,
        order: int = 2,
        key: Array | int = 0,
        sample_sharing: SampleSharing = None,
        track_history: bool = False,
    ) -> Self:
        """Create an approximator, evaluating all lower-order coalitions once."""
        sampler = PermutationSTIISampler(
            game.n_players,
            game.target_shape,
            sample_sharing,
            order=order,
            key=key,
        )
        seed_blocks = [_base_seed_masks(game.n_players)]
        for size in range(1, order):
            members = interaction_members(game.n_players, size)
            block = jnp.zeros((members.shape[0], game.n_players), dtype=bool)
            block = block.at[jnp.arange(members.shape[0])[:, None], members].set(True)
            seed_blocks.append(block)
        seeds = jnp.concatenate(seed_blocks, axis=0)
        return cls(game, sampler, _seeded_state(game, sampler, seeds, track_history=track_history))

    def explain(self) -> DenseExplanationArray[Array]:
        """Combine exact lower-order interactions with sampled top-order ones."""
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


def _base_seed_masks(n_players: int) -> Array:
    """Return the empty and grand coalition seed masks."""
    return jnp.stack(
        [jnp.zeros(n_players, dtype=bool), jnp.ones(n_players, dtype=bool)],
    )


def _seeded_state(
    game: Game[Array],
    sampler: PermutationWalkSampler,
    seed_masks: Array,
    *,
    track_history: bool = False,
) -> SamplingState[Array]:
    """Build the initial state by evaluating seed coalitions once."""
    n_seeds = seed_masks.shape[0]
    seed_coalitions = DenseCoalitionArray(
        jnp.broadcast_to(seed_masks, (*sampler.shared_target_shape, n_seeds, game.n_players)),
    )
    values = jnp.asarray(game(seed_coalitions))
    if values.shape != (*game.target_shape, n_seeds):
        msg = "permutation sampling requires scalar game values per coalition"
        raise UnsupportedGameError(msg)
    return SamplingState(
        coalitions=seed_coalitions,
        values=values,
        target_shape=game.target_shape,
        _track_history=track_history,
    )


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
