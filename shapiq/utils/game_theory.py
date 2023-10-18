"""This module contains utility functions for dealing with sets, coalitions and game theory."""

from itertools import chain, combinations
from typing import Iterable, Any, Optional, Union, Callable, TypeVar, Tuple


__all__ = [
    "powerset",
]


def powerset(
        iterable: Iterable[Any],
        min_size: int = -1,
        max_size: Optional[int] = None
) -> Iterable[Any]:
    """Return a powerset of the iterable with optional size limits.

    Args:
        iterable (iterable): Iterable.
        min_size (int, optional): Minimum size of the subsets. Defaults to -1.
        max_size (int, optional): Maximum size of the subsets. Defaults to None.

    Returns:
        iterable: Powerset of the iterable.
    """
    if max_size is None and min_size > -1:
        max_size = min_size
    s = list(iterable)
    if max_size is None:
        max_size = len(s)
    else:
        max_size = min(max_size, len(s))
    return chain.from_iterable(combinations(s, r) for r in range(max(min_size, 0), max_size + 1))
