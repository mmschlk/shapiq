"""Domain errors for shapiq."""

from __future__ import annotations


class InsufficientSamplesError(RuntimeError):
    """Raised when an approximator cannot produce coefficients from its current evidence."""


class UnsupportedGameError(TypeError):
    """Raised when an explainer cannot work with a supplied game."""


class SamplingStallWarning(UserWarning):
    """Warned when deduplicated sampling cannot spend its remaining budget.

    Issued during ``estimate`` and ``refine`` when deduplication is enabled and
    the sampler keeps producing only previously evaluated coalitions, which
    happens once the budget approaches the number of distinct coalitions of
    the game. The budget spent before the stall remains valid evidence, and
    estimates stay available.
    """
