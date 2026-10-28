"""The minimal interface for a suspended computation stage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class Partial[T](ABC):
    """A reusable suspended stage whose immediate result has type ``T``.

    Each family defines its own call signature and implements ``map`` to
    repackage itself without executing the stage. Concrete ``map`` methods
    should return their own family with its result parameter replaced.
    """

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Supply this family's inputs and execute exactly one stage."""

    @abstractmethod
    def map[U](self, function: Callable[[T], U]) -> Partial[U]:
        """Defer postcomposition, preserving the family and callback result.

        Calling the mapped stage must yield ``function(self(*args, **kwargs))``.
        A callback's result is retained intact, including any new partial stages.
        The original partial remains reusable.
        """
