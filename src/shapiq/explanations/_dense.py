from __future__ import annotations

from dataclasses import dataclass
from itertools import repeat
from typing import TYPE_CHECKING, Protocol, cast

import jax.numpy as jnp
from jax import Array

from shapiq._shape import Shape, normalize_shape, validate_n_players
from shapiq.explanations._base import ExplanationArray
from shapiq.interactions import (
    Interaction,
    InteractionIndexName,
    InteractionOrientation,
    iter_interactions,
    normalize_interaction,
    validate_interaction_metadata,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True, repr=False)
class DenseExplanationArray[ValueT](ExplanationArray[ValueT]):
    """Explanation array with dense attribution storage by interaction order."""

    attributions_by_order: Mapping[int, ValueT]
    n_players: int
    interaction_index: InteractionIndexName
    order: int
    shape: Shape = ()
    orientation: InteractionOrientation = "undirected"

    def __post_init__(self) -> None:
        """Normalize and validate metadata."""
        n_players = validate_n_players(self.n_players)
        shape = normalize_shape(self.shape)
        object.__setattr__(self, "n_players", n_players)
        object.__setattr__(self, "shape", shape)
        validate_interaction_metadata(
            interaction_index=self.interaction_index,
            order=self.order,
            orientation=self.orientation,
            n_players=n_players,
        )

    def __getitem__(self, key: object) -> DenseExplanationArray[ValueT]:
        """Index explanation target axes and preserve dense storage."""
        if self.shape == ():
            msg = "cannot index a scalar explanation"
            raise IndexError(msg)
        key_tuple = key if isinstance(key, tuple) else (key,)
        new_shape = tuple(int(dim) for dim in jnp.empty(self.shape)[key].shape)
        new_values = {
            order: _slice_attributions(values, key_tuple)
            for order, values in self.attributions_by_order.items()
        }
        return type(self)(
            new_values,
            n_players=self.n_players,
            interaction_index=self.interaction_index,
            order=self.order,
            shape=new_shape,
            orientation=self.orientation,
        )

    def __iter__(self) -> object:
        """Iterate over the first explanation target axis."""
        if self.shape == ():
            msg = "scalar explanation arrays are not iterable"
            raise TypeError(msg)
        for index in range(self.shape[0]):
            yield self[index]

    def __len__(self) -> int:
        """Return the first explanation target axis length."""
        if self.shape == ():
            msg = "scalar explanation arrays have no len()"
            raise TypeError(msg)
        return self.shape[0]

    def __repr__(self) -> str:
        """Return a concise representation."""
        return (
            f"{type(self).__name__}(shape={self.shape!r}, n_players={self.n_players!r}, "
            f"interaction_index={self.interaction_index!r}, order={self.order!r}, "
            f"orientation={self.orientation!r})"
        )

    def attribution(self, interaction: Sequence[int] | object) -> ValueT:
        """Return attributions for an interaction or fixed-size interaction array."""
        if isinstance(interaction, tuple):
            normalized = self._normalize_represented(cast("Sequence[int]", interaction))
            values = self.attributions_by_order[len(normalized)]
            key = (*repeat(slice(None), len(self.shape)), self._position(normalized), Ellipsis)
            return cast("ValueT", cast("_Indexable", values)[key])
        interactions = _as_interaction_array(interaction)
        positions = self._positions(interactions)
        values = self.attributions_by_order[int(interactions.shape[-1])]
        return cast("ValueT", jnp.take(jnp.asarray(values), positions, axis=len(self.shape)))

    def has(self, interaction: Sequence[int] | object) -> object:
        """Return where attributions are available for an interaction."""
        if isinstance(interaction, tuple):
            try:
                self._normalize_represented(cast("Sequence[int]", interaction))
            except (TypeError, ValueError, KeyError):
                return jnp.zeros(self.shape, dtype=bool)
            return jnp.ones(self.shape, dtype=bool)
        interactions = _as_interaction_array(interaction)
        mask = jnp.asarray([self._is_represented(tuple(row)) for row in _rows(interactions)])
        mask = jnp.reshape(mask, interactions.shape[:-1])
        return jnp.broadcast_to(mask, jnp.broadcast_shapes(self.shape, interactions.shape[:-1]))

    def _normalize_represented(self, interaction: Sequence[int]) -> Interaction:
        normalized = normalize_interaction(
            interaction,
            orientation=self.orientation,
            n_players=self.n_players,
        )
        if len(normalized) > self.order or len(normalized) not in self.attributions_by_order:
            msg = "interaction is not represented"
            raise KeyError(msg)
        return normalized

    def _is_represented(self, interaction: Sequence[int]) -> bool:
        try:
            self._normalize_represented(interaction)
        except (TypeError, ValueError, KeyError):
            return False
        return True

    def _position(self, interaction: Interaction) -> int:
        return list(iter_interactions(
            self.n_players,
            len(interaction),
            min_order=len(interaction),
            orientation=self.orientation,
        )).index(interaction)

    def _positions(self, interactions: Array) -> Array:
        return jnp.reshape(
            jnp.asarray([self._position(self._normalize_represented(tuple(row))) for row in _rows(interactions)]),
            interactions.shape[:-1],
        )


def _slice_attributions(value: object, key: tuple[object, ...]) -> object:
    return cast("_Indexable", value)[(*key, slice(None), Ellipsis)]


def _as_interaction_array(value: object) -> Array:
    array = jnp.asarray(value)
    if array.ndim < 1:
        msg = "interaction arrays must have a final interaction axis"
        raise TypeError(msg)
    if array.dtype == jnp.bool_ or not jnp.issubdtype(array.dtype, jnp.integer):
        msg = "interaction arrays must have integer dtype"
        raise TypeError(msg)
    return array


def _rows(array: Array) -> list[list[int]]:
    return jnp.reshape(array, (-1, array.shape[-1])).tolist()


class _Indexable(Protocol):
    """Minimal protocol for indexable attribution storage."""

    def __getitem__(self, key: object) -> object:
        """Return indexed data."""
