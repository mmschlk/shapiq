"""Exceptions for the UI module."""

from __future__ import annotations


class UnknownDataLoadingMethodException(Exception):
    """Raised when an unknown data loading method is specified."""

    def __init__(self, method: str):
        super().__init__(f"Unknown data loading method: {method}")
