from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import jax.numpy as jnp

from shapiq._shape import Shape, normalize_shape, validate_n_players
from shapiq.explanations._base import ExplanationArray, validate_explained_index
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
            interaction_index=self.index.name,
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
            actual = jnp.shape(cast("jnp.ndarray", value))
            if actual != expected:
                msg = (
                    f"attributions for {interaction} have shape {actual}, expected "
                    f"{expected} (targets, then value_shape)"
                )
                raise ValueError(msg)
        if self.baseline is not None:
            actual = jnp.shape(cast("jnp.ndarray", self.baseline))
            if actual != expected:
                msg = f"the baseline has shape {actual}, expected {expected}"
                raise ValueError(msg)

    def __getitem__(self, key: object) -> SparseExplanationArray[ValueT]:
        """Index explanation target axes and preserve sparse storage."""
        if self.shape == ():
            msg = "cannot index a scalar explanation"
            raise IndexError(msg)
        key_tuple = key if isinstance(key, tuple) else (key,)
        new_shape = tuple(int(dim) for dim in jnp.empty(self.shape)[key].shape)
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
        """Return attributions for an interaction."""
        if not isinstance(interaction, tuple):
            if self.default_attribution is None:
                msg = "array interaction lookup is unavailable without a default"
                raise KeyError(msg)
            return self.default_attribution(interaction)
        normalized = self._normalize_valid(cast("Sequence[int]", interaction))
        if normalized in self.attributions:
            return self.attributions[normalized]
        if self.default_attribution is not None:
            return self.default_attribution(normalized)
        msg = "interaction attribution is missing"
        raise KeyError(msg)

    def has(self, interaction: Sequence[int] | object) -> object:
        """Return where attributions are available for an interaction."""
        if not isinstance(interaction, tuple):
            return jnp.full(self.shape, self.default_attribution is not None, dtype=bool)
        try:
            normalized = self._normalize_valid(cast("Sequence[int]", interaction))
        except (TypeError, ValueError):
            return jnp.zeros(self.shape, dtype=bool)
        available = normalized in self.attributions or self.default_attribution is not None
        return jnp.full(self.shape, available, dtype=bool)

    def _normalize_valid(self, interaction: Sequence[int]) -> Interaction:
        normalized = normalize_interaction(
            interaction,
            orientation=self.orientation,
            n_players=self.n_players,
        )
        if len(normalized) > self.order:
            msg = "interaction is not represented"
            raise KeyError(msg)
        return normalized


class _Indexable(Protocol):
    """Minimal protocol for indexable attribution storage."""

    def __getitem__(self, key: object) -> object:
        """Return indexed data."""
