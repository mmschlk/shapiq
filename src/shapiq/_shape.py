from __future__ import annotations

from collections.abc import Iterable
from math import prod

type Shape = tuple[int, ...]
type ShapeLike = int | Iterable[int]


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
