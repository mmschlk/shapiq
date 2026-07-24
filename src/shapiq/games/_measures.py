"""Measures over coalitions: the inner product of the game space.

A measure assigns every coalition a positive weight by its size. It is
the second half of the design's center — the pair (game, measure): a
projection index is a subspace plus a measure, a sampling law targets a
measure, and fidelity is distance under one. Projections compose into a
tower only under a shared measure.

Weights are host float64 and normalized so the full coalition table sums
to one (exactness is semantic here; evaluation precision is the game
boundary's business).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Measure:
    """A size-symmetric probability measure over the coalition lattice."""

    weight_of_size: np.ndarray  # (n_players + 1,) float64, one coalition each
    n_players: int

    def row_weights(self, masks: ArrayLike) -> np.ndarray:
        """Return one probability per mask row ``(..., m, n) -> (..., m)``."""
        sizes = np.asarray(masks, dtype=bool).sum(axis=-1)
        return self.weight_of_size[sizes]


def _normalized(raw_of_size: np.ndarray, n_players: int) -> Measure:
    counts = np.array([math.comb(n_players, size) for size in range(n_players + 1)])
    total = float((raw_of_size * counts).sum())
    return Measure(weight_of_size=raw_of_size / total, n_players=n_players)


def uniform_measure(n_players: int) -> Measure:
    """The uniform measure: every coalition equally likely (Banzhaf world).

    The Fourier basis is orthonormal under it; degree-k truncation is the
    best k-additive approximation (Hammer-Holzman).
    """
    return _normalized(np.ones(n_players + 1), n_players)


def soft_shapley_measure(n_players: int, boost: float = 1e6) -> Measure:
    """The Shapley kernel with its degenerate endpoints softened.

    The exact kernel puts infinite weight on the empty and grand
    coalition — its projections live as constrained least squares, not an
    orthonormal basis. ``boost`` approximates the constraints while
    keeping a proper inner product, so the projection tower applies.
    """
    raw = np.empty(n_players + 1)
    for size in range(n_players + 1):
        if size in (0, n_players):
            raw[size] = boost
        else:
            raw[size] = (n_players - 1) / (math.comb(n_players, size) * size * (n_players - size))
    return _normalized(raw, n_players)


def product_measure(n_players: int, p: float) -> Measure:
    """Independent membership with probability ``p`` (weighted-Banzhaf world)."""
    if not 0.0 < p < 1.0:
        msg = "p must lie strictly between 0 and 1"
        raise ValueError(msg)
    raw = np.array(
        [p**size * (1.0 - p) ** (n_players - size) for size in range(n_players + 1)],
    )
    return _normalized(raw, n_players)
