"""This module contains utility functions for dealing with sets, coalitions and game theory."""

from itertools import chain, combinations
from typing import Iterable, Any, Optional, Union, Callable, TypeVar, Tuple


__all__ = [
    "powerset",
]


def powerset(
        iterable: Iterable[Any],
        min_size: Optional[int] = 0,
        max_size: Optional[int] = None
) -> Iterable[tuple[Any, ...]]:
    """Return a powerset of an iterable as tuples with optional size limits.

    Args:
        iterable: Iterable.
        min_size: Minimum size of the subsets. Defaults to 0 (start with the empty set).
        max_size: Maximum size of the subsets. Defaults to None (all possible sizes).

    Returns:
        iterable: Powerset of the iterable.

    Example:
        >>> list(powerset([1, 2, 3]))
        [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

        >>> list(powerset([1, 2, 3], min_size=1))
        [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

        >>> list(powerset([1, 2, 3], max_size=2))
        [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3)]
    """
    s = list(iterable)
    max_size = len(s) if max_size is None else min(max_size, len(s))
    return chain.from_iterable(combinations(s, r) for r in range(max(min_size, 0), max_size + 1))
