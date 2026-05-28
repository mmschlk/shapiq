"""Domain errors for shapiq."""

from __future__ import annotations


class InsufficientSamplesError(RuntimeError):
    """Raised when an approximator cannot explain from its current state."""


class UnsupportedGameError(TypeError):
    """Raised when an explainer cannot work with a supplied game."""


class SamplingError(RuntimeError):
    """Raised when a sampler cannot produce requested coalitions."""


class HistoryError(IndexError):
    """Raised when approximation history is unavailable or invalid."""
