"""Axis moves between the boundary value layout and the internal compute form.

At the game boundary and in sampling states, dense values carry logical axes
first, then the sample axis, then the value's internal axes (ADR 0006).
Estimators accumulate with the value axes moved to the front instead: leading
axes broadcast against mask-derived arrays by left-padding, which is always
correct, whereas trailing value axes would misalign silently.
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
