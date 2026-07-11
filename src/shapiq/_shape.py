from __future__ import annotations

from collections.abc import Iterable
from math import prod
from typing import Any, cast

import numpy as np

type Shape = tuple[int, ...]
type ShapeLike = int | Iterable[int]


def broadcast_shapes(*shapes: Shape) -> Shape:
    """Return the broadcast of logical shapes as a plain tuple.

    Shape arithmetic is host metadata, so it lives here rather than in a
    compute backend.
    """
    return tuple(int(dim) for dim in np.broadcast_shapes(*shapes))


def indexed_shape(shape: Shape, key: object) -> Shape:
    """Return the logical shape produced by indexing ``shape`` with ``key``.

    The key semantics are resolved by indexing a stride-zero host dummy, so
    no allocation or kernel is spent on shape metadata; out-of-range indices
    raise ``IndexError``.
    """
    dummy = cast("Any", np.broadcast_to(np.empty((), dtype=bool), shape))
    return tuple(int(dim) for dim in dummy[key].shape)


def expand_ellipsis(key: tuple[object, ...], ndim: int) -> tuple[object, ...]:
    """Replace an ``Ellipsis`` key entry with explicit full slices.

    Container ``__getitem__`` implementations append their own trailing
    axes to the key, which an unexpanded user ``Ellipsis`` would clash
    with. Entries other than ``None`` count as consuming one axis, so a
    multi-axis boolean mask combined with an ellipsis is not supported.
    """
    positions = [index for index, entry in enumerate(key) if entry is Ellipsis]
    if len(positions) != 1:
        return key
    consumed = sum(1 for entry in key if entry is not Ellipsis and entry is not None)
    fill = (slice(None),) * max(ndim - consumed, 0)
    return (*key[: positions[0]], *fill, *key[positions[0] + 1 :])


def shape_of(value: object) -> Shape:
    """Return a value's logical shape as a plain tuple, host-side.

    Prefers the value's own ``shape`` attribute; values without one
    (nested sequences, scalars) are probed through NumPy.
    """
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    return tuple(int(dim) for dim in np.shape(cast("Any", value)))


def normalize_shape(shape: ShapeLike = ()) -> Shape:
    """Return a canonical shape tuple."""
    if isinstance(shape, bool):
        msg = "shape dimensions must be integers, got bool"
        raise TypeError(msg)
    dims = (shape,) if isinstance(shape, int) else tuple(shape)
    for dim in dims:
        if isinstance(dim, bool) or not isinstance(dim, int):
            msg = f"shape dimensions must be integers, got {type(dim).__name__}"
            raise TypeError(msg)
        if dim < 0:
            msg = "shape dimensions must be non-negative"
            raise ValueError(msg)
    return dims


def validate_n_players(n_players: int) -> int:
    """Validate and return a player count."""
    if isinstance(n_players, bool) or not isinstance(n_players, int):
        msg = f"n_players must be an integer, got {type(n_players).__name__}"
        raise TypeError(msg)
    if n_players < 0:
        msg = "n_players must be non-negative"
        raise ValueError(msg)
    return n_players


def validate_int(name: str, value: int, *, minimum: int = 0) -> int:
    """Validate an integer argument excluding bool."""
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer, got {type(value).__name__}"
        raise TypeError(msg)
    if value < minimum:
        msg = f"{name} must be at least {minimum}"
        raise ValueError(msg)
    return value


def logical_size(shape: Shape) -> int:
    """Return the number of logical elements for a shape."""
    return prod(shape)


def ensure_bool(name: str, value: object) -> bool:
    """Validate a parameter that must be exactly bool."""
    if type(value) is not bool:
        msg = f"{name} must be a bool"
        raise TypeError(msg)
    return value
