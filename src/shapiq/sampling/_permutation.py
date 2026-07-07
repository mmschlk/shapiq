from __future__ import annotations

from abc import abstractmethod
from itertools import combinations
from math import comb
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from shapiq._shape import validate_int
from shapiq.sampling._schedule import UnitScheduleSampler

if TYPE_CHECKING:
    from shapiq._shape import ShapeLike
    from shapiq.sampling._base import ShareSamples


class PermutationWalkSampler(UnitScheduleSampler):
    """Base sampler for coalition walks derived from random permutations.

    One walk is the coalition block derived from one permutation, so the
    sampling quantum is the walk length. Every walk starts with a block of
    proper permutation prefixes (the chain); order-specific off-chain
    coalitions follow. Each walk derives its permutation from
    ``fold_in(random_state, walk_index)``, so sampling does not depend on how
    a budget is split across calls.
    """

    order: int

    def __init__(
        self,
        n_players: int,
        target_shape: ShapeLike = (),
        *,
        share_samples: ShareSamples = False,
        order: int = 1,
        random_state: Array | int = 0,
    ) -> None:
        """Initialize a permutation walk sampler.

        Args:
            n_players: Number of players in the explained game. Must be at
                least two.
            target_shape: Shape of the explanation targets, matching the
                game's target shape.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes. ``False`` samples independently per
                target; ``True`` shares across all target axes; an integer or
                tuple of integers shares across the selected axes.
            order: Maximum interaction order of the walks. Must satisfy
                ``1 <= order <= n_players``.
            random_state: Integer seed or JAX PRNG key used to derive one
                permutation per walk.

        Raises:
            ValueError: If ``n_players`` is smaller than two, or if ``order``
                is out of range.
            TypeError: If ``order`` is not an integer, or if ``random_state``
                is neither an integer nor a JAX PRNG key.
        """
        super().__init__(
            n_players,
            target_shape,
            share_samples=share_samples,
            random_state=random_state,
        )
        validate_int("order", order, minimum=1)
        if order > self.n_players:
            msg = "order must not exceed the number of players"
            raise ValueError(msg)
        self.order = order

    def _sampled_unit_masks(self, unit_index: int) -> Array:
        """Return the walk masks of one sampled unit."""
        return self._walk_masks(unit_index)

    @abstractmethod
    def _walk_masks(self, walk_index: int) -> Array:
        """Return the dense coalition masks of one full walk."""

    def _player_positions(self, walk_index: int) -> Array:
        """Return each player's position in the permutation of a walk."""
        players = jnp.broadcast_to(
            jnp.arange(self.n_players),
            (*self.shared_target_shape, self.n_players),
        )
        walk_key = jax.random.fold_in(self._key, walk_index)
        permutation = jax.random.permutation(walk_key, players, axis=-1, independent=True)
        return jnp.argsort(permutation, axis=-1)


class PermutationSIISampler(PermutationWalkSampler):
    """Sampler emitting consecutive-window interaction walks.

    The seed block holds the empty and grand coalition. A walk contains the
    proper prefix chain followed, for every window size ``s`` from two to
    ``order``, by one block per off-chain pattern of that size (patterns
    ordered by size then lexicographically, windows ascending within each
    block). Window ``k`` of size ``s`` covers the players at permutation
    positions ``k`` to ``k + s - 1``; together with the chain its coalitions
    provide the discrete derivative of the window players. With ``order=1``
    a walk is the plain Shapley-value chain.
    """

    @property
    def sampling_quantum(self) -> int:
        """Return the walk length in coalitions.

        A walk costs ``(n - 1)`` chain coalitions plus, for every window size
        ``s`` from two to ``order``, ``(n - s + 1) * (2**s - s - 1)``
        off-chain coalitions. With ``order=1`` this is the plain chain length
        ``n - 1``.
        """
        return (self.n_players - 1) + sum(
            (self.n_players - size + 1) * len(off_chain_patterns(size))
            for size in range(2, self.order + 1)
        )

    def _walk_masks(self, walk_index: int) -> Array:
        """Return chain masks followed by per-size, per-pattern window masks."""
        positions = self._player_positions(walk_index)
        blocks = [positions[..., None, :] < jnp.arange(1, self.n_players)[:, None]]
        for size in range(2, self.order + 1):
            window_starts = jnp.arange(self.n_players - size + 1)
            prefixes = positions[..., None, :] < window_starts[:, None]
            members = [
                positions[..., None, :] == (window_starts + offset)[:, None]
                for offset in range(size)
            ]
            for pattern in off_chain_patterns(size):
                mask = prefixes
                for offset in pattern:
                    mask = mask | members[offset]
                blocks.append(mask)
        return jnp.concatenate(blocks, axis=-2)


class PermutationSTIISampler(PermutationWalkSampler):
    """Sampler emitting Shapley-Taylor top-order walks.

    The seed block holds the empty and grand coalition followed by every
    coalition of size one to ``order - 1`` in lexicographic order, providing
    the exact lower-order discrete derivatives at the empty coalition. A walk
    contains the prefix chain up to length ``n_players - order`` followed by
    one block per non-empty subset pattern of the interaction (patterns
    ordered by size then lexicographically, interactions in lexicographic
    order within each block). For every top-order interaction the walk
    provides its discrete derivative at the set of players strictly preceding
    the whole interaction. With ``order=1`` a walk is the plain Shapley-value
    chain.
    """

    @property
    def n_seed_samples(self) -> int:
        """Return the length of the deterministic seed block."""
        return 2 + sum(comb(self.n_players, size) for size in range(1, self.order))

    @property
    def sampling_quantum(self) -> int:
        """Return the walk length in coalitions.

        A walk costs ``(n - order)`` chain coalitions plus
        ``comb(n, order) * (2**order - 1)`` off-chain coalitions. With
        ``order=1`` this is the plain chain length ``n - 1``. Quanta grow
        quickly with ``order``; prefer small orders or generous budgets.
        """
        if self.order == 1:
            return self.n_players - 1
        n_interactions = comb(self.n_players, self.order)
        return (self.n_players - self.order) + n_interactions * len(
            nonempty_patterns(self.order),
        )

    def _seed_masks(self) -> Array:
        """Return seed masks for the empty, grand, and lower-order coalitions."""
        blocks = [super()._seed_masks()]
        for size in range(1, self.order):
            members = interaction_members(self.n_players, size)
            block = jnp.zeros((members.shape[0], self.n_players), dtype=bool)
            block = block.at[jnp.arange(members.shape[0])[:, None], members].set(True)
            blocks.append(block)
        return jnp.concatenate(blocks, axis=0)

    def _walk_masks(self, walk_index: int) -> Array:
        """Return chain masks followed by per-pattern interaction masks."""
        positions = self._player_positions(walk_index)
        if self.order == 1:
            return positions[..., None, :] < jnp.arange(1, self.n_players)[:, None]
        chain = positions[..., None, :] < jnp.arange(1, self.n_players - self.order + 1)[:, None]
        members = interaction_members(self.n_players, self.order)
        first_positions = jnp.min(positions[..., members], axis=-1)
        predecessors = positions[..., None, :] < first_positions[..., :, None]
        blocks = [chain]
        for pattern in nonempty_patterns(self.order):
            selected = members[:, jnp.asarray(pattern)]
            pattern_mask = (
                jnp.zeros(
                    (members.shape[0], self.n_players),
                    dtype=bool,
                )
                .at[jnp.arange(members.shape[0])[:, None], selected]
                .set(True)
            )
            blocks.append(predecessors | pattern_mask)
        return jnp.concatenate(blocks, axis=-2)


def off_chain_patterns(size: int) -> tuple[tuple[int, ...], ...]:
    """Return window subset patterns whose coalitions are not chain prefixes.

    Discrete derivatives of a size ``size`` window need all ``2**size``
    subsets of the window; the ``size + 1`` prefix runs are already on the
    walk's chain, and the remaining patterns are evaluated off-chain. The
    pattern order defines the walk layout shared between the SII sampler and
    the SII approximator.

    Args:
        size: Window size, i.e. the interaction order of the window.

    Returns:
        All subsets of ``range(size)`` that are not prefix runs, ordered by
        size then lexicographically.
    """
    runs = {tuple(range(length)) for length in range(size + 1)}
    return tuple(
        pattern
        for length in range(size + 1)
        for pattern in combinations(range(size), length)
        if pattern not in runs
    )


def nonempty_patterns(size: int) -> tuple[tuple[int, ...], ...]:
    """Return all non-empty subset patterns ordered by size then lexicographically.

    The pattern order defines the walk layout shared between the STII
    sampler and the STII approximator.

    Args:
        size: Interaction order of the patterns.

    Returns:
        All non-empty subsets of ``range(size)``, ordered by size then
        lexicographically.
    """
    return tuple(
        pattern for length in range(1, size + 1) for pattern in combinations(range(size), length)
    )


def interaction_members(n_players: int, order: int) -> Array:
    """Return the member table of all order-sized interactions.

    Args:
        n_players: Number of players.
        order: Interaction order of the listed interactions.

    Returns:
        An integer array of shape ``(comb(n_players, order), order)`` whose
        rows are the interactions in lexicographic order, matching the order
        used by explanations.
    """
    return jnp.asarray(list(combinations(range(n_players), order)))
