from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import jax.numpy as jnp

from shapiq._shape import (
    Shape,
    broadcast_shapes,
    expand_ellipsis,
    indexed_shape,
    normalize_shape,
    shape_of,
    validate_n_players,
)
from shapiq.explanations._base import (
    ExplanationArray,
    as_interaction_array,
    check_represented_window,
    interaction_rows,
    validate_explained_index,
)
from shapiq.interactions import (
    Interaction,
    InteractionIndex,
    InteractionOrientation,
    normalize_interaction,
    validate_interaction_metadata,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


@dataclass(frozen=True, repr=False)
class SparseExplanationArray[ValueT](ExplanationArray[ValueT]):
    """Explanation array with sparse attribution storage."""

    attributions: Mapping[Interaction, ValueT]
    n_players: int
    index: InteractionIndex
    order: int
    shape: Shape = ()
    orientation: InteractionOrientation = "undirected"
    default_attribution: Callable[[Sequence[int] | object], ValueT] | None = None
    value_shape: Shape = ()
    baseline: ValueT | None = None

    def __post_init__(self) -> None:
        """Normalize and validate metadata, then validate attribution shapes."""
        n_players = validate_n_players(self.n_players)
        shape = normalize_shape(self.shape)
        value_shape = normalize_shape(self.value_shape)
        object.__setattr__(self, "n_players", n_players)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "value_shape", value_shape)
        validate_explained_index(self.index, order=self.order)
        validate_interaction_metadata(
            index_name=self.index.name,
            order=self.order,
            orientation=self.orientation,
            n_players=n_players,
        )
        normalized = {
            self._normalize_valid(interaction): value
            for interaction, value in self.attributions.items()
        }
        object.__setattr__(self, "attributions", normalized)
        expected = (*shape, *value_shape)
        for interaction, value in normalized.items():
            actual = shape_of(value)
            if actual != expected:
                msg = (
                    f"attributions for {interaction} have shape {actual}, expected "
                    f"{expected} (targets, then value_shape)"
                )
                raise ValueError(msg)
        if self.baseline is not None:
            actual = shape_of(self.baseline)
            if actual != expected:
                msg = f"the baseline has shape {actual}, expected {expected}"
                raise ValueError(msg)

    def __getitem__(self, key: object) -> SparseExplanationArray[ValueT]:
        """Index explanation target axes and preserve sparse storage."""
        if self.shape == ():
            msg = "cannot index a scalar explanation"
            raise IndexError(msg)
        new_shape = indexed_shape(self.shape, key)
        key_tuple = expand_ellipsis(key if isinstance(key, tuple) else (key,), self.ndim)
        new_values = {
            interaction: cast("ValueT", cast("_Indexable", value)[(*key_tuple, Ellipsis)])
            for interaction, value in self.attributions.items()
        }
        new_baseline = (
            None
            if self.baseline is None
            else cast("ValueT", cast("_Indexable", self.baseline)[(*key_tuple, Ellipsis)])
        )
        return type(self)(
            new_values,
            n_players=self.n_players,
            index=self.index,
            order=self.order,
            shape=new_shape,
            orientation=self.orientation,
            default_attribution=self.default_attribution,
            value_shape=self.value_shape,
            baseline=new_baseline,
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
            f"index={self.index!r}, order={self.order!r}, "
            f"orientation={self.orientation!r}, value_shape={self.value_shape!r})"
        )

    def attribution(self, interaction: Sequence[int] | object) -> ValueT:
        """Return attributions for an interaction or fixed-size interaction array."""
        if isinstance(interaction, tuple):
            return self._single_attribution(cast("Sequence[int]", interaction))
        interactions = as_interaction_array(interaction)
        rows = [self._single_attribution(tuple(row)) for row in interaction_rows(interactions)]
        batch_shape = tuple(interactions.shape[:-1])
        if not rows:
            empty = jnp.zeros((*self.shape, *batch_shape, *self.value_shape))
            return cast("ValueT", empty)
        stacked = jnp.stack(cast("list[jnp.ndarray]", rows), axis=len(self.shape))
        return cast(
            "ValueT",
            jnp.reshape(stacked, (*self.shape, *batch_shape, *self.value_shape)),
        )

    def has(self, interaction: Sequence[int] | object) -> object:
        """Return where attributions are available for an interaction."""
        if isinstance(interaction, tuple):
            available = self._is_available(cast("Sequence[int]", interaction))
            return jnp.full(self.shape, available, dtype=bool)
        interactions = as_interaction_array(interaction)
        mask = jnp.asarray(
            [self._is_available(tuple(row)) for row in interaction_rows(interactions)],
            dtype=bool,
        )
        mask = jnp.reshape(mask, interactions.shape[:-1])
        return jnp.broadcast_to(mask, broadcast_shapes(self.shape, interactions.shape[:-1]))

    def _single_attribution(self, interaction: Sequence[int]) -> ValueT:
        normalized = self._normalize_valid(interaction)
        if normalized in self.attributions:
            return self.attributions[normalized]
        if self.default_attribution is not None:
            return self.default_attribution(normalized)
        msg = (
            f"no attribution is stored for {normalized} and this sparse explanation "
            "has no default; probe availability with has()"
        )
        raise KeyError(msg)

    def _is_available(self, interaction: Sequence[int]) -> bool:
        try:
            normalized = self._normalize_valid(interaction)
        except (TypeError, ValueError, KeyError):
            return False
        return normalized in self.attributions or self.default_attribution is not None

    def _normalize_valid(self, interaction: Sequence[int]) -> Interaction:
        normalized = normalize_interaction(
            interaction,
            orientation=self.orientation,
            n_players=self.n_players,
        )
        check_represented_window(self.index, len(normalized), self.order)
        return normalized


class _Indexable(Protocol):
    """Minimal protocol for indexable attribution storage."""

    def __getitem__(self, key: object) -> object:
        """Return indexed data."""
