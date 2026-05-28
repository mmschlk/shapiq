from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

from shapiq._shape import Shape, logical_size
from shapiq.interactions import (
    Interaction,
    InteractionIndexName,
    InteractionOrientation,
    iter_interactions,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class ExplanationArray[ValueT](ABC):
    """Array-like collection whose logical elements are explanations."""

    n_players: int
    shape: Shape
    interaction_index: InteractionIndexName
    order: int
    orientation: InteractionOrientation

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

    def iter_interactions(self, min_order: int = 0) -> Iterator[Interaction]:
        """Iterate represented interactions using this explanation's metadata."""
        return iter_interactions(
            self.n_players,
            self.order,
            min_order=min_order,
            orientation=self.orientation,
        )
