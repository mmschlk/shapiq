"""Shared least squares machinery for regression-defined interaction indices."""

from __future__ import annotations

from fractions import Fraction
from functools import cache
from itertools import combinations
from math import comb

import jax.numpy as jnp
import numpy as np
from jax import Array

from shapiq.errors import InsufficientSamplesError


@cache
def interaction_masks(n_players: int, size: int) -> Array:
    """Return dense member masks of all size-``size`` interactions, lexicographic."""
    if size == 0:
        return jnp.zeros((1, n_players), dtype=bool)
    members = np.fromiter(
        combinations(range(n_players), size),
        dtype=np.dtype((np.int64, (size,))),
        count=comb(n_players, size),
    )
    masks = np.zeros((members.shape[0], n_players), dtype=bool)
    masks[np.arange(members.shape[0])[:, None], members] = True
    return jnp.asarray(masks)


def interaction_design(masks: Array, order: int) -> Array:
    """Return subset-membership columns for all interactions up to ``order``.

    The column of an interaction is one exactly for coalitions containing all
    of its players. The empty-interaction column is not represented: with the
    response shifted by ``v(empty)``, its coefficient is exactly zero.
    """
    n_players = masks.shape[-1]
    coalitions = masks.astype(jnp.int32)
    columns = []
    for size in range(1, order + 1):
        members = interaction_masks(n_players, size).astype(jnp.int32)
        intersections = coalitions @ members.T
        columns.append(1.0 * (intersections == size))
    return jnp.concatenate(columns, axis=-1)


def eliminate_constraint(design: Array) -> tuple[Array, Array]:
    """Substitute the grand-coalition constraint out of the design.

    The faithful fit predicts ``v(T) - v(empty)`` under the constraint that
    all attributions sum to ``v(N) - v(empty)``. Substituting the last column
    removes the constraint exactly and keeps the reduced system as well
    conditioned as the design itself, so float32 solves are safe.

    Returns:
        The reduced design ``(rows, columns - 1)`` and the pivot column
        ``(rows, 1)``; leading batch axes pass through.
    """
    pivot = design[..., -1:]
    return design[..., :-1] - pivot, pivot


def solve_faithful(
    reduced: Array,
    pivot: Array,
    response: Array,
    delta: Array,
    *,
    sqrt_weights: Array | None = None,
    identify: bool = False,
    deduplicating: bool = False,
) -> Array:
    """Solve the eliminated least squares problem for all interaction columns.

    Args:
        reduced: Reduced design of shape ``(rows, columns - 1)``.
        pivot: Pivot column of shape ``(rows, 1)``.
        response: Empty-shifted values per target, shape ``(rows, targets)``.
        delta: Grand-minus-empty value per target, shape ``(targets,)``.
        sqrt_weights: Optional square roots of row weights, shape ``(rows,)``;
            rows with zero weight drop out of the fit.
        identify: Whether to require the design to identify all columns,
            raising ``InsufficientSamplesError`` otherwise.
        deduplicating: Whether the caller deduplicates coalitions, selecting
            the identification hint.

    Returns:
        Attributions for every interaction column, shape ``(columns, targets)``.
    """
    shifted = response - pivot * delta[..., None, :]
    if sqrt_weights is not None:
        reduced = reduced * sqrt_weights[:, None]
        shifted = shifted * sqrt_weights[:, None]
    if identify:
        partial = lstsq_identified(reduced, shifted, deduplicating=deduplicating)
    else:
        partial, *_ = jnp.linalg.lstsq(reduced, shifted)
    last = delta[..., None, :] - jnp.sum(partial, axis=-2, keepdims=True)
    return jnp.concatenate([partial, last], axis=-2)


def lstsq_identified(design: Array, response: Array, *, deduplicating: bool = False) -> Array:
    """Solve a least squares fit, requiring the design to identify all columns.

    The identifying rank comes from the solve itself, so no second
    decomposition is paid on top of the fit.

    Args:
        design: Design matrix of shape ``(rows, columns)``.
        response: Responses of shape ``(rows, targets)``.
        deduplicating: Whether the caller deduplicates coalitions, selecting
            the identification hint.

    Returns:
        The least squares solution of shape ``(columns, targets)``.

    Raises:
        InsufficientSamplesError: If the design does not identify every
            column.
    """
    solution, _, rank, _ = jnp.linalg.lstsq(design, response)
    _require_rank(int(rank), int(design.shape[-1]), deduplicating=deduplicating)
    return solution


def bernoulli_numbers(order: int) -> list[float]:
    """Return the Bernoulli numbers up to ``order`` with the B(1) = -1/2 convention."""
    numbers = [Fraction(1)]
    for m in range(1, order + 1):
        acc = sum((Fraction(comb(m + 1, j)) * numbers[j] for j in range(m)), Fraction(0))
        numbers.append(-acc / (m + 1))
    return [float(number) for number in numbers]


def bernoulli_design(masks: Array, order: int) -> Array:
    """Return Bernoulli-weighted intersection columns for all interactions up to ``order``.

    The kADD-SHAP basis: columns follow the same size-then-lexicographic
    interaction layout as ``interaction_design``, weighted by the Bernoulli
    table, and the empty coalition's row is zero, so a response shifted by
    ``v(empty)`` interpolates the empty coalition automatically.
    """
    n_players = masks.shape[-1]
    table = _bernoulli_weight_table(order)
    columns = []
    for size in range(1, order + 1):
        member_masks = interaction_masks(n_players, size)
        intersections = masks.astype(jnp.int32) @ member_masks.T.astype(jnp.int32)
        columns.append(table[size][intersections])
    return jnp.concatenate(columns, axis=-1)


def _bernoulli_weight_table(order: int) -> Array:
    """Return kADD-SHAP design weights per interaction and intersection size."""
    bernoulli = bernoulli_numbers(order)
    table = [[0.0] * (order + 1) for _ in range(order + 1)]
    for size in range(1, order + 1):
        for intersection in range(1, size + 1):
            table[size][intersection] = sum(
                comb(intersection, top) * bernoulli[size - top]
                for top in range(1, intersection + 1)
            )
    return jnp.asarray(table)


def require_identification(reduced: Array, *, deduplicating: bool = False) -> None:
    """Raise when the sampled coalitions do not yet identify all attributions."""
    _require_rank(
        int(jnp.linalg.matrix_rank(reduced)),
        int(reduced.shape[-1]),
        deduplicating=deduplicating,
    )


def _require_rank(rank: int, needed: int, *, deduplicating: bool) -> None:
    """Raise the identification error when a solved rank falls short."""
    if rank >= needed:
        return
    hint = (
        "sample more evaluations and retry"
        if deduplicating
        else "sample more evaluations (deduplicate=True reaches distinct "
        "coalitions with the fewest evaluations)"
    )
    msg = (
        "the faithful regression is not yet identified: the sampled coalitions "
        f"give rank {rank} of the {needed} required, so at least "
        f"{needed - rank} more distinct informative coalitions are needed; {hint}"
    )
    raise InsufficientSamplesError(msg)
