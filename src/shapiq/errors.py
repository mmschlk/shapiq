# Copyright (c) 2026 shapiq contributors
"""Domain errors for shapiq."""

from __future__ import annotations


class InsufficientSamplesError(RuntimeError):
    """Raised when an approximation lacks enough evidence to produce a game."""


class UnsupportedGameError(TypeError):
    """Raised when a transformation has no applicable implementation for a game."""


class SamplingError(RuntimeError):
    """Raised when a sampler cannot produce requested coalitions."""


class HistoryError(IndexError):
    """Raised when approximation history is unavailable or invalid."""
