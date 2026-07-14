"""Combinatorial ranking of fixed-size interactions.

The lexicographic rank of a size-``k`` interaction among all ``comb(n, k)``
combinations is a closed-form sum of binomial coefficients, so positions in
dense per-order attribution blocks resolve in ``O(k)`` per interaction
instead of materializing the full combination list.
"""

from __future__ import annotations

from functools import cache
from math import comb
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from jax import Array

    from shapiq.interactions._types import Interaction


def interaction_rank(interaction: Interaction, n_players: int) -> int:
    """Return the lexicographic rank of one sorted interaction.

    Args:
        interaction: Sorted tuple of distinct player indices.
        n_players: Number of players.

    Returns:
        The position of the interaction among all ``comb(n_players, size)``
        combinations in lexicographic order, matching ``iter_interactions``.
    """
    size = len(interaction)
    terms = sum(
        comb(n_players - 1 - player, size - position)
        for position, player in enumerate(interaction)
    )
    return comb(n_players, size) - 1 - terms


def interaction_ranks(members: Array, n_players: int) -> Array:
    """Return lexicographic ranks of fixed-size interactions among all combinations.

    The device twin of ``host_interaction_ranks`` for member arrays that
    live inside estimator pipelines.

    Args:
        members: Integer array whose final axis holds the players of one
            interaction each; player order within an interaction is free.
        n_players: Number of players.

    Returns:
        An integer array of ``members.shape[:-1]`` whose entries are the
        interactions' positions in lexicographic order, matching
        ``iter_interactions``.
    """
    size = members.shape[-1]
    ordered = jnp.sort(members, axis=-1)
    binomials = jnp.asarray(_binomial_table(n_players, size))
    terms = binomials[n_players - 1 - ordered, size - jnp.arange(size)]
    return comb(n_players, size) - 1 - jnp.sum(terms, axis=-1)


def host_interaction_ranks(members: np.ndarray, n_players: int) -> np.ndarray:
    """Return lexicographic ranks as plain NumPy, compilation-free.

    The host twin of ``interaction_ranks`` for lookup paths, where the
    members arrive from the caller and positions index host-side storage:
    plain NumPy resolves them at full speed on the first call, with no
    kernel compilation.

    Args:
        members: Integer array whose final axis holds the players of one
            interaction each; player order within an interaction is free.
        n_players: Number of players.

    Returns:
        An integer array of ``members.shape[:-1]`` whose entries are the
        interactions' positions in lexicographic order, matching
        ``iter_interactions``.
    """
    size = members.shape[-1]
    ordered = np.sort(members, axis=-1)
    binomials = _binomial_table(n_players, size)
    terms = binomials[n_players - 1 - ordered, size - np.arange(size)]
    return comb(n_players, size) - 1 - np.sum(terms, axis=-1)


@cache
def _binomial_table(n_players: int, size: int) -> np.ndarray:
    """Return ``comb(row, column)`` for rows ``0..n-1`` and columns ``0..size``."""
    return np.asarray(
        [[comb(row, column) for column in range(size + 1)] for row in range(n_players)],
    )
