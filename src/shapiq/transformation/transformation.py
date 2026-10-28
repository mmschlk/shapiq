"""The callable contract for an ordinary unary transformation."""

from __future__ import annotations

from typing import Protocol


class Transformation[A, B](Protocol):
    """An ordinary ``A -> B`` callable, mapped explicitly over partial stages."""

    def __call__(self, value: A, /) -> B:
        """Transform one immediate input, without automatic partial traversal."""
        ...
