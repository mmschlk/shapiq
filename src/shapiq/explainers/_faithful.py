"""Shared least squares machinery for regression-defined interaction indices."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from shapiq.errors import InsufficientSamplesError
from shapiq.interactions._iteration import interaction_masks


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
        ``(rows, 1)``.
    """
    pivot = design[:, -1:]
    return design[:, :-1] - pivot, pivot


def solve_faithful(
    reduced: Array,
    pivot: Array,
    response: Array,
    delta: Array,
    *,
    sqrt_weights: Array | None = None,
) -> Array:
    """Solve the eliminated least squares problem for all interaction columns.

    Args:
        reduced: Reduced design of shape ``(rows, columns - 1)``.
        pivot: Pivot column of shape ``(rows, 1)``.
        response: Empty-shifted values per target, shape ``(rows, targets)``.
        delta: Grand-minus-empty value per target, shape ``(targets,)``.
        sqrt_weights: Optional square roots of row weights, shape ``(rows,)``;
            rows with zero weight drop out of the fit.

    Returns:
        Attributions for every interaction column, shape ``(columns, targets)``.
    """
    shifted = response - pivot * delta[None, :]
    if sqrt_weights is not None:
        reduced = reduced * sqrt_weights[:, None]
        shifted = shifted * sqrt_weights[:, None]
    partial, *_ = jnp.linalg.lstsq(reduced, shifted)
    last = delta[None, :] - jnp.sum(partial, axis=0, keepdims=True)
    return jnp.concatenate([partial, last], axis=0)


def require_identification(reduced: Array) -> None:
    """Raise when the sampled coalitions do not yet identify all attributions."""
    needed = int(reduced.shape[-1])
    rank = int(jnp.linalg.matrix_rank(reduced))
    if rank < needed:
        msg = (
            "the faithful regression is not yet identified: the sampled coalitions "
            f"give rank {rank} of the {needed} required; sample more evaluations "
            "(deduplicate=True reaches distinct coalitions with the fewest evaluations)"
        )
        raise InsufficientSamplesError(msg)
