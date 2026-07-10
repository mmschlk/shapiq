from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

import jax.numpy as jnp

from shapiq._shape import Shape, logical_size
from shapiq.interactions import (
    Interaction,
    InteractionIndex,
    InteractionOrientation,
    iter_interactions,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from jax import Array


class ExplanationArray[ValueT](ABC):
    """Array-like collection whose logical elements are explanations.

    Attributions are defined on the centered game, following the
    game-theoretic convention that the empty coalition has value zero; the
    empty-coalition value itself travels separately as the ``baseline``.
    """

    n_players: int
    shape: Shape
    index: InteractionIndex
    order: int
    orientation: InteractionOrientation
    baseline: ValueT | None

    @property
    def interaction_index(self) -> str:
        """Return the name of the explained interaction index."""
        return self.index.name

    @property
    def ndim(self) -> int:
        """Return the number of logical dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Return the number of logical explanation elements."""
        return logical_size(self.shape)

    @abstractmethod
    def __getitem__(self, key: object) -> Self:
        """Index explanation target axes and return an explanation array."""

    @abstractmethod
    def __iter__(self) -> object:
        """Iterate over the first logical axis."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the first logical axis length."""

    def __bool__(self) -> bool:
        """Reject ambiguous truth-value testing."""
        msg = f"truth value of {type(self).__name__} is ambiguous"
        raise TypeError(msg)

    def __call__(self, interaction: Sequence[int] | object) -> ValueT:
        """Return attributions for an interaction."""
        return self.attribution(interaction)

    @abstractmethod
    def attribution(self, interaction: Sequence[int] | object) -> ValueT:
        """Return attributions for an interaction."""

    @abstractmethod
    def has(self, interaction: Sequence[int] | object) -> object:
        """Return where attributions are available for an interaction."""

    def iter_interactions(self, min_order: int | None = None) -> Iterator[Interaction]:
        """Iterate represented interactions using this explanation's metadata.

        Args:
            min_order: The smallest interaction order to yield. Defaults to
                the smallest size the explained index represents, so every
                yielded interaction can be looked up on this explanation.

        Returns:
            An iterator over interactions by increasing order.
        """
        resolved = self.index.min_interaction_size if min_order is None else min_order
        return iter_interactions(
            self.n_players,
            self.order,
            min_order=resolved,
            orientation=self.orientation,
        )


def check_represented_window(index: InteractionIndex, size: int, order: int) -> None:
    """Raise a teaching ``KeyError`` when a size lies outside the represented window.

    The window is ``index.min_interaction_size .. order``: the explained
    index defines no attributions below its smallest represented size, and
    the explanation carries none above its order.
    """
    min_size = index.min_interaction_size
    if min_size <= size <= order:
        return
    if size == 0:
        msg = (
            f"{index.name} defines no order-0 attribution; the empty-coalition "
            "value travels as the explanation's baseline"
        )
    else:
        msg = (
            f"{index.name} explanations at order {order} represent interaction "
            f"sizes {min_size} to {order}, not {size}"
        )
    raise KeyError(msg)


def as_interaction_array(value: object) -> Array:
    """Coerce an array-of-interactions lookup argument, teaching on misuse."""
    array = jnp.asarray(value)
    if array.ndim < 1:
        msg = (
            "interactions must be tuples of player indices, e.g. explanation((0,)) "
            "for player 0; array lookups need a final interaction-members axis"
        )
        raise TypeError(msg)
    if array.dtype == jnp.bool_ or not jnp.issubdtype(array.dtype, jnp.integer):
        msg = "interaction arrays must have integer dtype"
        raise TypeError(msg)
    return array


def interaction_rows(array: Array) -> list[list[int]]:
    """Return the interactions of a lookup array as flat host rows."""
    return jnp.reshape(array, (-1, array.shape[-1])).tolist()


def validate_explained_index(index: object, *, order: int) -> InteractionIndex:
    """Validate the index recorded on an explanation and its order consistency."""
    if isinstance(index, str):
        msg = (
            "explanations carry the interaction index object, not its name: "
            f"pass index=shapiq.SV() (or the index the explainer used), "
            f"not the string {index!r}"
        )
        raise TypeError(msg)
    if not isinstance(index, InteractionIndex):
        msg = f"index must be an interaction index object, got {type(index).__name__}"
        raise TypeError(msg)
    if index.order is not None and index.order != order:
        msg = (
            f"the explanation records order {order} but its index declares "
            f"order {index.order}; explanations carry the index they were computed with"
        )
        raise ValueError(msg)
    return index
