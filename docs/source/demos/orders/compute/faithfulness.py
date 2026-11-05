"""FSII of every order and its R^2 beyond size under the Shapley distribution, for these scripts.

Every order of the demo is FSII, the Faithful Shapley Interactions (Tsai et al., 2023): of all
explanations with values for groups of up to k players, the one that fits the game best in a
weighted least squares over its coalitions, with the weights of the Shapley values (those of
KernelSHAP), under two constraints: its value of the empty group is the game's value of the empty
coalition, and its values add up to the game's value of all players. ``shapiq`` converts the Möbius
transform of a game into FSII of any order exactly; FSII of order 1 is the Shapley values, and FSII
of the highest order with a Möbius value is the Möbius transform, the full decomposition.

The weights of this least squares are a distribution of the coalitions, the Shapley distribution:
a size s from 1 to n - 1 with probability proportional to 1 / (s (n - s)), and every coalition of
that size equally likely. The empty and the full coalition have no weight: the constraints fix the
explanation there. The scripts score every order under this distribution: exactly over all
coalitions for 8 players, and on coalitions drawn from it for more.

They score it by its R^2 beyond size. Under the Shapley distribution, most coalitions hold only a
few players or lack only a few, and a game's value mostly follows how many players a coalition
holds: plain R^2, against the mean of all values, credits any explanation with that trend, which
counting the players already gives (for the molecule, the size alone explains 98 % of the values'
variance, and the Shapley values reach R^2 = 0.98). R^2 beyond size measures the explanation's
error against the spread of the values among coalitions of the same size instead,

    R^2 = 1 - sum w (v(S) - v_hat(S))^2 / sum w (v(S) - mean of v over coalitions of size |S|)^2,

how much of what the number of players alone does not predict the explanation reproduces: which
players are there, not how many. It is 1 for an exact explanation, and below 0 for one worse than
the mean of every size. Its numerator is the weighted least squares FSII minimizes, and its
denominator the same for every explanation: FSII of every order is still the best of its order.
FSII of order k + 1 could keep the values of FSII of order k and give every group of k + 1 players
0, so it fits at least as well: the R^2 does not fall with the order (on drawn coalitions, up to the
chance of the draw), and reaches 1 with the full decomposition.

The explanation predicts a coalition's value as the sum of the values of its subsets, the empty
one's (the game's value of the empty coalition) included: no constant is fit to the coalitions.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import comb

if TYPE_CHECKING:
    from shapiq.game_theory.moebius_converter import MoebiusConverter


def size_weights(n: int) -> np.ndarray:
    """Return the probability of every coalition size, 0 to n, under the Shapley distribution."""
    sizes = np.arange(1, n)
    weights = np.zeros(n + 1)
    weights[1:n] = 1 / (sizes * (n - sizes))
    return weights / weights.sum()


def all_coalitions(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return every coalition of n players but the empty and the full one, with its probability."""
    coalitions = np.array(list(itertools.product([False, True], repeat=n)))
    sizes = coalitions.sum(axis=1)
    proper = (sizes > 0) & (sizes < n)
    probabilities = size_weights(n)[sizes] / comb(n, sizes)
    return coalitions[proper], probabilities[proper]


def sample(n: int, count: int, seed: int) -> np.ndarray:
    """Return ``count`` coalitions of n players drawn from the Shapley distribution."""
    rng = np.random.default_rng(seed)
    sizes = rng.choice(n + 1, size=count, p=size_weights(n))
    order = rng.random((count, n)).argsort(axis=1)  # the players in a random order, per coalition
    return order < sizes[:, None]  # the first players of that order, as many as the size


def predictions(values: dict[tuple[int, ...], float], coalitions: np.ndarray) -> np.ndarray:
    """Return the sum of the values of every coalition's subsets: the explanation's prediction.

    The empty group's value, the explanation's baseline, is part of every coalition's sum. The
    groups are summed in chunks, so that millions of them fit in memory.
    """
    groups = [group for group in values if group]
    out = np.full(len(coalitions), values.get((), 0.0))
    for start in range(0, len(groups), 100_000):
        chunk = groups[start : start + 100_000]
        members = np.zeros((len(chunk), coalitions.shape[1]))
        for row, group in enumerate(chunk):
            members[row, list(group)] = 1
        inside = coalitions.astype(float) @ members.T == members.sum(axis=1)
        out += inside @ np.array([values[group] for group in chunk])
    return out


def size_means(
    y: np.ndarray, sizes: np.ndarray, weights: np.ndarray | None = None
) -> dict[int, float]:
    """Return the mean value of the coalitions of every size, weighted by their probability."""
    weights = np.ones(len(y)) if weights is None else weights
    return {
        int(size): float(np.average(y[sizes == size], weights=weights[sizes == size]))
        for size in np.unique(sizes)
    }


def r2(
    y: np.ndarray,
    y_hat: np.ndarray,
    sizes: np.ndarray,
    weights: np.ndarray | None = None,
    means: dict[int, float] | None = None,
) -> float:
    """Return the R^2 beyond size of a prediction, its coalitions weighted by their probability.

    The prediction's error is measured against the spread of the values around the mean of their
    coalitions' size (``means``, by default the means of these coalitions). Without ``weights``,
    the coalitions are equally likely: drawn from the distribution.
    """
    weights = np.ones(len(y)) if weights is None else weights
    means = size_means(y, sizes, weights) if means is None else means
    baseline = np.array([means[int(size)] for size in sizes])
    return float(1 - np.sum(weights * (y - y_hat) ** 2) / np.sum(weights * (y - baseline) ** 2))


def fsii(converter: MoebiusConverter, order: int) -> dict[tuple[int, ...], float]:
    """Return the values of FSII of an order, by group: the empty group's and all others not 0."""
    explanation = converter("FSII", order)
    return {
        group: float(explanation[group])
        for group in explanation.interaction_lookup
        if not group or abs(explanation[group]) > 1e-9
    }
