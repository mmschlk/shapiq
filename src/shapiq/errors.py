"""Domain errors for shapiq."""

from __future__ import annotations


class InsufficientSamplesError(RuntimeError):
    """Raised when an approximator cannot explain from its current state."""


class UnsupportedGameError(TypeError):
    """Raised when an explainer cannot work with a supplied game."""


class SamplingStallWarning(UserWarning):
    """Warned when deduplicated sampling cannot spend its remaining budget.

    Issued during ``Approximator.sample`` when deduplication is enabled and
    the sampler keeps producing only previously evaluated coalitions, which
    happens once the budget approaches the number of distinct coalitions of
    the game. The budget spent before the stall remains valid evidence, and
    explanations stay available.
    """
