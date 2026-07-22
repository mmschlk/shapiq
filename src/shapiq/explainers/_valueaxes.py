"""Axis moves between the boundary value layout and the canonical internal layout.

Game values cross exactly two seams. On entry — ``Approximator._call_game``
for sampled evidence, ``ExactExplainer._game_values`` for the powerset — boundary
values (broadcast targets, then samples, then value axes, per ADR 0006) become
the canonical internal layout: value axes leading, sample axis last. Leading
value axes broadcast against mask-derived arrays by left-padding, which is
always correct, whereas trailing value axes would misalign silently. On exit —
explanation construction — attribution blocks and baselines return to the
public trailing layout. Between the seams, states and estimators never move
value axes.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def to_leading(values: Array, n_value_axes: int) -> Array:
    """Move trailing value axes to the front for broadcast-safe accumulation."""
    if n_value_axes == 0:
        return values
    sources = tuple(range(values.ndim - n_value_axes, values.ndim))
    return jnp.moveaxis(values, sources, tuple(range(n_value_axes)))


def to_trailing(values: Array, n_value_axes: int) -> Array:
    """Move leading value axes behind all other axes for storage or display."""
    if n_value_axes == 0:
        return values
    destinations = tuple(range(values.ndim - n_value_axes, values.ndim))
    return jnp.moveaxis(values, tuple(range(n_value_axes)), destinations)
