"""Metrics for evaluating performances of different approximation methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shapiq import InteractionValues


EQUALITY_TOLERANCE = 1e-10  # Tolerance for floating point arithmetic


def iv_equal_values(iv_one: InteractionValues, iv_second: InteractionValues) -> bool:
    """Check if two interaction values are equal."""
    iv_diff = iv_one - iv_second
    return sum(abs(iv_diff)) < EQUALITY_TOLERANCE
