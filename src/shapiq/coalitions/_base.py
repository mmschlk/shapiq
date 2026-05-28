from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from shapiq._shape import Shape, logical_size


class CoalitionArray(ABC):
    """Array-like collection whose logical elements are coalitions."""

    @property
    @abstractmethod
    def n_players(self) -> int:
        """Return the fixed number of players."""

    @property
    @abstractmethod
    def shape(self) -> Shape:
        """Return the logical coalition-array shape."""

    @property
    def ndim(self) -> int:
        """Return the number of logical dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Return the number of logical coalition elements."""
        return logical_size(self.shape)

    @abstractmethod
    def __getitem__(self, key: object) -> Self:
        """Index logical coalition axes and return a coalition array."""

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

    @abstractmethod
    def to_dense(self) -> object:
        """Return a dense boolean representation."""
