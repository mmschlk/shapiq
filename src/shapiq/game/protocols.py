# Copyright (c) 2026 shapiq contributors
"""Semantic game capabilities, independent of backing representations."""

from __future__ import annotations

from typing import Literal, Protocol


class KAdditive(Protocol):
    """A game with no nonzero Möbius coefficients above a declared order.

    This capability is used alongside ``Game``. Implementations guarantee the
    mathematics; structural conformance alone does not establish k-additivity.
    An empty-coalition coefficient is allowed.
    """

    @property
    def additivity(self) -> int:
        """Return a nonnegative upper bound on the nonzero interaction order."""


class Additive(KAdditive, Protocol):
    """A zero-empty game whose coalition values sum its singleton values."""

    @property
    def additivity(self) -> Literal[1]:
        """Return the singleton interaction-order bound."""
