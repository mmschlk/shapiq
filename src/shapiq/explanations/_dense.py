from __future__ import annotations

from dataclasses import dataclass
from itertools import repeat
from math import comb
from typing import TYPE_CHECKING, Protocol, cast

import jax.numpy as jnp
import numpy as np

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
    validate_explained_index,
)
from shapiq.interactions import (
    Interaction,
    InteractionIndex,
    InteractionOrientation,
    normalize_interaction,
    validate_interaction_metadata,
)
from shapiq.interactions._ranks import host_interaction_ranks, interaction_rank

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True, repr=False)
class DenseExplanationArray[ValueT](ExplanationArray[ValueT]):
    """Explanation array with dense attribution storage by interaction order."""

    attributions_by_order: Mapping[int, ValueT]
    n_players: int
    index: InteractionIndex
    order: int
    shape: Shape = ()
    orientation: InteractionOrientation = "undirected"
    value_shape: Shape = ()
    baseline: ValueT | None = None

    def __post_init__(self) -> None:
        """Normalize and validate metadata, then validate attribution blocks."""
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
        for size, block in self.attributions_by_order.items():
            expected = (*shape, comb(n_players, size), *value_shape)
            actual = shape_of(block)
            if actual != expected:
                msg = (
                    f"order-{size} attributions have shape {actual}, expected "
                    f"{expected} (targets, then interactions, then value_shape)"
                )
                raise ValueError(msg)
        if self.baseline is not None:
            expected = (*shape, *value_shape)
            actual = shape_of(self.baseline)
            if actual != expected:
                msg = f"the baseline has shape {actual}, expected {expected}"
                raise ValueError(msg)

    def __getitem__(self, key: object) -> DenseExplanationArray[ValueT]:
        """Index explanation target axes and preserve dense storage."""
        if self.shape == ():
            msg = "cannot index a scalar explanation"
            raise IndexError(msg)
        new_shape = indexed_shape(self.shape, key)
        key_tuple = expand_ellipsis(key if isinstance(key, tuple) else (key,), self.ndim)
        new_values = {
            order: _slice_attributions(values, key_tuple)
            for order, values in self.attributions_by_order.items()
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
            normalized = self._normalize_represented(cast("Sequence[int]", interaction))
            values = self.attributions_by_order[len(normalized)]
            key = (*repeat(slice(None), len(self.shape)), self._position(normalized), Ellipsis)
            return cast("ValueT", cast("_Indexable", values)[key])
        interactions = as_interaction_array(interaction)
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
        interactions = as_interaction_array(interaction)
        rows = interactions.reshape(-1, interactions.shape[-1])
        available = (
            self._valid_players(rows)
            if self._represents_size(int(interactions.shape[-1]))
            else np.zeros(rows.shape[0], dtype=bool)
        )
        mask = jnp.asarray(available.reshape(interactions.shape[:-1]))
        return jnp.broadcast_to(mask, broadcast_shapes(self.shape, interactions.shape[:-1]))

    def _represents_size(self, size: int) -> bool:
        """Return whether lookups of one interaction size can succeed."""
        try:
            check_represented_window(self.index, size, self.order)
        except KeyError:
            return False
        return size in self.attributions_by_order

    def _normalize_represented(self, interaction: Sequence[int]) -> Interaction:
        normalized = normalize_interaction(
            interaction,
            orientation=self.orientation,
            n_players=self.n_players,
        )
        check_represented_window(self.index, len(normalized), self.order)
        if len(normalized) not in self.attributions_by_order:
            msg = f"no order-{len(normalized)} attributions are stored on this explanation"
            raise KeyError(msg)
        return normalized

    def _valid_players(self, rows: np.ndarray) -> np.ndarray:
        """Return per-interaction validity: players in range, no repeats."""
        in_bounds = np.all((rows >= 0) & (rows < self.n_players), axis=-1)
        if rows.shape[-1] < 2:
            return in_bounds
        ordered = np.sort(rows, axis=-1)
        return in_bounds & np.all(ordered[..., 1:] > ordered[..., :-1], axis=-1)

    def _position(self, interaction: Interaction) -> int:
        # every constructible explanation is undirected; a directed rank
        # would need its own closed form next to interaction_rank
        return interaction_rank(interaction, self.n_players)

    def _positions(self, interactions: np.ndarray) -> np.ndarray:
        """Resolve block positions of a lookup array, teaching on bad rows.

        Positions are host-side index math and resolve in plain NumPy, so
        bulk lookups run at full speed on the first call with no kernel
        compilation.
        """
        rows = interactions.reshape(-1, interactions.shape[-1])
        invalid = ~self._valid_players(rows)
        if bool(invalid.any()):
            offender = tuple(int(player) for player in rows[int(np.argmax(invalid))])
            normalize_interaction(
                offender,
                orientation=self.orientation,
                n_players=self.n_players,
            )
            msg = "invalid interaction row"  # unreachable: the checks above mirror it
            raise ValueError(msg)
        size = int(interactions.shape[-1])
        check_represented_window(self.index, size, self.order)
        if size not in self.attributions_by_order:
            msg = f"no order-{size} attributions are stored on this explanation"
            raise KeyError(msg)
        return host_interaction_ranks(rows, self.n_players).reshape(interactions.shape[:-1])


def _slice_attributions(value: object, key: tuple[object, ...]) -> object:
    return cast("_Indexable", value)[(*key, slice(None), Ellipsis)]


class _Indexable(Protocol):
    """Minimal protocol for indexable attribution storage."""

    def __getitem__(self, key: object) -> object:
        """Return indexed data."""
