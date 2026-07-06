from __future__ import annotations

from abc import abstractmethod
from itertools import combinations
from typing import TYPE_CHECKING, Self

import jax
import jax.numpy as jnp

from shapiq._shape import validate_int
from shapiq.coalitions import DenseCoalitionArray
from shapiq.sampling._base import Sampler

if TYPE_CHECKING:
    from jax import Array

    from shapiq._shape import ShapeLike
    from shapiq.coalitions import CoalitionArray
    from shapiq.sampling._base import SampleSharing
    from shapiq.sampling._state import ApproximationState


class PermutationWalkSampler(Sampler["ApproximationState"]):
    """Base sampler for coalition walks derived from random permutations.

    One walk is the coalition block derived from one permutation, so the
    sampling quantum is the walk length. Every walk starts with a block of
    proper permutation prefixes (the chain); order-specific off-chain
    coalitions follow. Budgets are spent exactly: a walk cut short by the
    budget stays pending and is resumed by the evolved sampler. Each walk
    derives its permutation from ``fold_in(key, walk_index)``, so sampling
    does not depend on how a budget is split across calls.
    """

    order: int
    _walks_started: int
    _pending_pos: int

    def __init__(
        self,
        n_players: int,
        target_shape: ShapeLike = (),
        sample_sharing: SampleSharing = None,
        *,
        order: int = 1,
        key: Array | int = 0,
        _walks_started: int = 0,
        _pending_pos: int = 0,
    ) -> None:
        """Initialize a permutation walk sampler."""
        super().__init__(n_players, target_shape, sample_sharing)
        if self.n_players < 2:
            msg = "permutation walks require at least two players"
            raise ValueError(msg)
        validate_int("order", order, minimum=1)
        if order > self.n_players:
            msg = "order must not exceed the number of players"
            raise ValueError(msg)
        self.order = order
        self._key = jax.random.key(key) if isinstance(key, int) else key
        self._walks_started = validate_int("_walks_started", _walks_started)
        self._pending_pos = validate_int("_pending_pos", _pending_pos)

    @property
    def n_pending(self) -> int:
        """Return the number of emitted coalitions of the unfinished walk."""
        return self._pending_pos

    def _sample(
        self,
        state: ApproximationState,  # noqa: ARG002 - permutation sampling is not adaptive
        budget: int,
    ) -> tuple[CoalitionArray, Self]:
        """Emit exactly budget walk coalitions, resuming any pending walk."""
        quantum = self.sampling_quantum
        chunks: list[Array] = []
        walks = self._walks_started
        position = self._pending_pos
        remaining = budget
        if position > 0:
            take = min(quantum - position, remaining)
            chunks.append(self._walk_masks(walks - 1)[..., position : position + take, :])
            position = (position + take) % quantum
            remaining -= take
        while remaining > 0:
            take = min(quantum, remaining)
            chunks.append(self._walk_masks(walks)[..., :take, :])
            walks += 1
            position = take % quantum
            remaining -= take
        coalitions = DenseCoalitionArray(jnp.concatenate(chunks, axis=-2))
        next_sampler = type(self)(
            self.n_players,
            self.target_shape,
            self.sample_sharing,
            order=self.order,
            key=self._key,
            _walks_started=walks,
            _pending_pos=position,
        )
        return coalitions, next_sampler

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

    A walk contains the proper prefix chain followed, for every window size
    ``s`` from two to ``order``, by one block per off-chain pattern of that
    size (patterns ordered by size then lexicographically, windows ascending
    within each block). Window ``k`` of size ``s`` covers the players at
    permutation positions ``k`` to ``k + s - 1``; together with the chain its
    coalitions provide the discrete derivative of the window players. With
    ``order=1`` a walk is the plain Shapley-value chain.
    """

    @property
    def sampling_quantum(self) -> int:
        """Return the walk length in coalitions."""
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

    A walk contains the prefix chain up to length ``n_players - order``
    followed by one block per non-empty subset pattern of the interaction
    (patterns ordered by size then lexicographically, interactions in
    lexicographic order within each block). For every top-order interaction
    the walk provides its discrete derivative at the set of players strictly
    preceding the whole interaction. With ``order=1`` a walk is the plain
    Shapley-value chain.
    """

    @property
    def sampling_quantum(self) -> int:
        """Return the walk length in coalitions."""
        if self.order == 1:
            return self.n_players - 1
        n_interactions = len(list(combinations(range(self.n_players), self.order)))
        return (self.n_players - self.order) + n_interactions * len(
            nonempty_patterns(self.order),
        )

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
            pattern_mask = jnp.zeros(
                (members.shape[0], self.n_players),
                dtype=bool,
            ).at[jnp.arange(members.shape[0])[:, None], selected].set(True)
            blocks.append(predecessors | pattern_mask)
        return jnp.concatenate(blocks, axis=-2)


def off_chain_patterns(size: int) -> tuple[tuple[int, ...], ...]:
    """Return window subset patterns whose coalitions are not chain prefixes."""
    runs = {tuple(range(length)) for length in range(size + 1)}
    return tuple(
        pattern
        for length in range(size + 1)
        for pattern in combinations(range(size), length)
        if pattern not in runs
    )


def nonempty_patterns(size: int) -> tuple[tuple[int, ...], ...]:
    """Return all non-empty subset patterns ordered by size then lexicographically."""
    return tuple(
        pattern
        for length in range(1, size + 1)
        for pattern in combinations(range(size), length)
    )


def interaction_members(n_players: int, order: int) -> Array:
    """Return the member table of all order-sized interactions in lexicographic order."""
    return jnp.asarray(list(combinations(range(n_players), order)))
