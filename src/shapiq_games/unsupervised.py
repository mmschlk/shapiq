"""Unsupervised data games: how much statistical dependence a coalition of features shares."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import entr
from sklearn.preprocessing import KBinsDiscretizer

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix, GameValues

__all__ = ["UnsupervisedData", "total_correlation"]


def total_correlation(data: np.ndarray) -> float:
    r"""Return the total correlation of discrete data.

    .. math::
        TC(X_1, \dots, X_k) = \sum_i H(X_i) - H(X_1, \dots, X_k)

    Args:
        data: Discrete data of shape ``(n_samples, n_features)``.

    Returns:
        The total correlation in nats (zero for a single feature).
    """
    if data.shape[1] < 2:
        return 0.0
    codes, sizes, entropies = _code_columns(data)
    return sum(entropies) - _joint_entropy(codes, sizes)


def _code_columns(data: np.ndarray) -> tuple[np.ndarray, list[int], list[float]]:
    """Return every column coded as ``0, ..., size - 1`` in sorted order, the sizes, and entropies."""
    columns = [np.unique(column, return_inverse=True, return_counts=True) for column in data.T]
    codes = np.column_stack([inverse for _, inverse, _ in columns])
    sizes = [values.shape[0] for values, _, _ in columns]
    entropies = [_entropy(counts) for _, _, counts in columns]
    return codes, sizes, entropies


def _entropy(counts: np.ndarray) -> float:
    """Return the entropy, in nats, of the distribution given by counts.

    This is the arithmetic of :func:`scipy.stats.entropy` (normalize, then sum ``-p log p``)
    without its overhead of about 0.3 ms per call; the tests check that the two agree.
    """
    return float(np.sum(entr(counts / np.sum(counts))))


def _joint_entropy(codes: np.ndarray, sizes: list[int]) -> float:
    """Return the joint entropy, in nats, of discrete data given as codes.

    Args:
        codes: The data with every column coded as ``0, ..., size - 1`` in sorted order, of shape
            ``(n_samples, n_features)``.
        sizes: The number of distinct values of every column.

    Returns:
        The entropy of the distinct rows. A row's number in the row-major (lexicographic) order of
        all possible rows sorts the rows as ``np.unique(data, axis=0)`` does, which is much
        slower; it is only needed when the numbers would not fit into 64 bits.
    """
    if math.prod(sizes) <= np.iinfo(np.intp).max:
        rows = np.ravel_multi_index(tuple(codes.T), sizes)
        return _entropy(np.unique(rows, return_counts=True)[1])
    return _entropy(np.unique(codes, axis=0, return_counts=True)[1])


class UnsupervisedData(Game):
    """The unsupervised data game: the total correlation of a coalition of features.

    The players are the features, discretized into equal-width bins. The value of a coalition is
    the total correlation of its features, which is zero for the empty coalition and for single
    features (Balestra et al., 2022).

    Attributes:
        data_discrete: The discretized data.

    Examples:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, n_features=5, random_state=0)
        >>> game = UnsupervisedData(X)
        >>> game.n_players
        5
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        n_bins: int = 20,
        verbose: bool = False,
    ) -> None:
        """Initialize the unsupervised data game.

        Args:
            data: The data of shape ``(n_samples, n_features)``.
            n_bins: The number of equal-width bins per feature. Defaults to ``20``.
            verbose: Whether to show a progress bar when evaluating the game.
        """
        data = np.asarray(data, dtype=float)
        discretizer = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy="uniform", subsample=None
        )
        self.data_discrete = np.column_stack(
            [
                discretizer.fit_transform(data[:, [i]]).ravel().astype(int)
                for i in range(data.shape[1])
            ]
        )
        # what does not depend on the coalition: every column's codes and entropy
        self._codes, self._sizes, self._entropies = _code_columns(self.data_discrete)
        super().__init__(data.shape[1], normalize=False, verbose=verbose)

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the total correlation of the features of each coalition."""
        coalitions = as_bool_coalitions(coalitions)
        values = np.zeros(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            features = np.flatnonzero(coalition)
            if features.shape[0] < 2:
                continue  # zero for the empty coalition and single features
            marginal = sum(self._entropies[j] for j in features)
            sizes = [self._sizes[j] for j in features]
            values[i] = marginal - _joint_entropy(self._codes[:, features], sizes)
        return values
