"""Read-only, identity-keyed catalogs for the shared latent player space."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from operator import index as integer_index
from typing import TYPE_CHECKING, Self, SupportsIndex, cast, overload, override

import jax
import jax.numpy as jnp
import numpy as np
from ndlike.array_like import NumpyLike
from ndlike.jax_like import JaxLike

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

_BackingArray = np.ndarray | jax.Array | JaxLike | NumpyLike


def _is_numeric(data: np.ndarray | jax.Array) -> bool:
    """Recognize numeric dtypes, including JAX's extended floating-point types."""
    return jnp.issubdtype(data.dtype, jnp.number) or data.dtype == jnp.bool_


def _object_storage(entries: Sequence[object]) -> np.ndarray:
    """Retain one reference per player without expanding nested objects."""
    data = np.empty(len(entries), dtype=object)
    for position, player in enumerate(entries):
        data[position] = player
    return data


def _catalog_storage(players: Iterable[object]) -> _BackingArray:
    """Reuse array-like backings; store Python sequences as object references."""
    if isinstance(players, (np.ndarray, jax.Array)):
        data = players
        if data.ndim == 0:
            msg = "A player array must have a leading player dimension."
            raise ValueError(msg)
        if _is_numeric(data) or (data.ndim == 1 and data.dtype == object):
            return data
        return _object_storage(list(players))
    if isinstance(players, (JaxLike, NumpyLike)):
        if players.ndim == 0:
            msg = "A player array must have a leading player dimension."
            raise ValueError(msg)
        return players

    return _object_storage(players if isinstance(players, Sequence) else list(players))


def _same_player(left: object, right: object) -> bool:
    """Compare whole players, not individual coordinates of array players."""
    if left is right:
        return True
    if isinstance(left, (np.ndarray, jax.Array)):
        return bool(np.array_equal(cast("ArrayLike", left), cast("ArrayLike", right)))
    if isinstance(left, (JaxLike, NumpyLike)):
        return bool(np.all(left == right))
    return bool(left == right)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class LatentPlayers[P](Sequence[P]):
    """A read-only sequence of player metadata, hashed and compared by identity.

    Numeric arrays use their leading dimension as the player dimension and
    retain their NumPy or JAX backend. Python lists and other non-array iterables
    always become one-dimensional NumPy object arrays, preserving each entry
    by reference. No stacking, numeric conversion, or dtype promotion is inferred
    from sequence contents; pass an array explicitly for packed storage.
    Existing backings are reused without defensive copies or write-flag changes.
    ndlike ``JaxLike`` and ``NumpyLike`` backings retain their logical indexing
    and representation rather than being coerced to native arrays. Callers must
    not mutate shared storage, wrappers, or metadata objects after construction.

    Integer indexing directly returns the backing's player. Slices and
    one-dimensional boolean masks return another catalog, retaining its backend
    and the backing's usual view/allocation behavior. Returned players and views
    are not protected against mutation.
    Membership, ``count``, and ``index`` compare array players as whole arrays.
    Players and query values are expected to have homogeneous representations
    with symmetric equality. Mixed native arrays and array-like wrappers are
    not rejected, but their comparisons are outside the supported contract.
    Catalog equality itself is identity-based, not elementwise or list-style.

    This is Python-side metadata, not a JAX array or a pytree of dynamic values.
    Mutating shared data does not invalidate JIT caches and is unsupported.
    """

    _data: _BackingArray

    def __init__(self, players: Iterable[P]) -> None:
        """Reuse an array-like backing, or store a sequence as object references."""
        object.__setattr__(self, "_data", _catalog_storage(players))

    @classmethod
    def _from_storage(cls, data: _BackingArray) -> Self:
        """Wrap an indexing result without copying or repacking it."""
        catalog = object.__new__(cls)
        object.__setattr__(catalog, "_data", data)
        return catalog

    def __hash__(self) -> int:
        """Hash this catalog by identity, independently of its entries."""
        return id(self)

    @override
    def __len__(self) -> int:
        return len(self._data)

    @overload
    def __getitem__(self, index: SupportsIndex) -> P: ...

    @overload
    def __getitem__(
        self, index: slice | np.ndarray | jax.Array | list[bool]
    ) -> "LatentPlayers[P]": ...

    @override
    def __getitem__(
        self, index: SupportsIndex | slice | np.ndarray | jax.Array | list[bool]
    ) -> "P | LatentPlayers[P]":
        if isinstance(index, slice):
            return self._from_storage(self._data[index])
        if isinstance(index, (np.ndarray, jax.Array, list)):
            mask = index if isinstance(index, jax.Array) else np.asarray(index)
            if mask.ndim != 1 or mask.dtype != np.bool_:
                msg = "A player filter must be a one-dimensional boolean mask."
                raise IndexError(msg)
            if len(mask) != len(self):
                msg = "A player filter must match the catalog length."
                raise IndexError(msg)
            return self._from_storage(self._data[mask])
        position = integer_index(index)
        # JAX clips out-of-bounds indexing; sequences must instead raise,
        # including to terminate the inherited Sequence iterator.
        if not -len(self) <= position < len(self):
            msg = "Player index out of range."
            raise IndexError(msg)
        return cast("P", self._data[position])

    @override
    def __contains__(self, value: object) -> bool:
        return any(_same_player(player, value) for player in self)

    @override
    def count(self, value: object) -> int:
        """Count occurrences of a whole player, including array-valued players."""
        return sum(_same_player(player, value) for player in self)

    @override
    def index(
        self, value: object, start: SupportsIndex = 0, stop: SupportsIndex | None = None
    ) -> int:
        """Find the first matching whole player within list-style index bounds."""
        begin, end, _ = slice(start, stop).indices(len(self))
        for position in range(begin, end):
            if _same_player(self[position], value):
                return position
        msg = "Player is not in the catalog."
        raise ValueError(msg)
