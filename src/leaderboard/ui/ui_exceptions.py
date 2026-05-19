"""Exceptions for the UI module."""

from __future__ import annotations


class UnknownDataLoadingMethodError(Exception):
    """Raised when an unknown data loading method is specified."""

    def __init__(self, method: str) -> None:
        """Initialize with a message indicating the unknown data loading method."""
        super().__init__(f"Unknown data loading method: {method}")
