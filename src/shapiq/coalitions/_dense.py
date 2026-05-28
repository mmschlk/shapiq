from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Self, cast

import jax.numpy as jnp
from array_api_compat import array_namespace

from shapiq._shape import Shape, ShapeLike, normalize_shape, validate_n_players
from shapiq.coalitions._base import CoalitionArray


class _DenseStorage(Protocol):
    """Minimal storage protocol for dense coalitions."""

    shape: tuple[int, ...]
    dtype: object

    def __getitem__(self, key: object) -> object:
        """Return indexed storage."""


@dataclass(frozen=True, repr=False)
class DenseCoalitionArray(CoalitionArray):
    """Coalition array backed by a dense boolean array."""

    data: object

    def __post_init__(self) -> None:
        """Validate dense boolean storage."""
        if not hasattr(self.data, "shape") or not hasattr(self.data, "dtype"):
            msg = "data must be an array-api-compatible boolean array"
            raise TypeError(msg)
        if len(cast("_DenseStorage", self.data).shape) < 1:
            msg = "dense coalition storage must include a player axis"
            raise ValueError(msg)
        namespace = array_namespace(self.data)
        if not namespace.isdtype(cast("_DenseStorage", self.data).dtype, "bool"):
            msg = "dense coalition storage must have boolean dtype"
            raise TypeError(msg)

    @classmethod
    def empty(cls, shape: ShapeLike, n_players: int) -> Self:
        """Allocate uninitialized JAX-backed dense coalition storage."""
        logical_shape = normalize_shape(shape)
        return cls(jnp.empty((*logical_shape, validate_n_players(n_players)), dtype=bool))

    @classmethod
    def zeros(cls, shape: ShapeLike, n_players: int) -> Self:
        """Allocate all-absent JAX-backed dense coalition storage."""
        logical_shape = normalize_shape(shape)
        return cls(jnp.zeros((*logical_shape, validate_n_players(n_players)), dtype=bool))

    @classmethod
    def ones(cls, shape: ShapeLike, n_players: int) -> Self:
        """Allocate all-present JAX-backed dense coalition storage."""
        logical_shape = normalize_shape(shape)
        return cls(jnp.ones((*logical_shape, validate_n_players(n_players)), dtype=bool))

    @property
    def n_players(self) -> int:
        """Return the fixed number of players."""
        return int(cast("_DenseStorage", self.data).shape[-1])

    @property
    def shape(self) -> Shape:
        """Return the logical coalition-array shape."""
        return tuple(int(dim) for dim in cast("_DenseStorage", self.data).shape[:-1])

    def __getitem__(self, key: object) -> Self:
        """Index logical coalition axes and preserve dense representation."""
        if self.shape == ():
            msg = "cannot index a scalar coalition"
            raise IndexError(msg)
        if not isinstance(key, tuple):
            key = (key,)
        return type(self)(cast("_DenseStorage", self.data)[(*key, slice(None))])

    def __iter__(self) -> object:
        """Iterate over the first logical axis."""
        if self.shape == ():
            msg = "scalar coalition arrays are not iterable"
            raise TypeError(msg)
        for index in range(self.shape[0]):
            yield self[index]

    def __len__(self) -> int:
        """Return the first logical axis length."""
        if self.shape == ():
            msg = "scalar coalition arrays have no len()"
            raise TypeError(msg)
        return self.shape[0]

    def __repr__(self) -> str:
        """Return a concise representation."""
        return f"{type(self).__name__}(shape={self.shape!r}, n_players={self.n_players!r})"

    def to_dense(self) -> object:
        """Return the underlying dense boolean array."""
        return self.data
