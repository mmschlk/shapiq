"""Unsupervised data games: how much statistical dependence a coalition of features shares."""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import KBinsDiscretizer

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions

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
    marginal = sum(
        float(entropy(np.unique(data[:, i], return_counts=True)[1])) for i in range(data.shape[1])
    )
    joint = float(entropy(np.unique(data, axis=0, return_counts=True)[1]))
    return marginal - joint


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
        super().__init__(data.shape[1], normalize=False, verbose=verbose)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the total correlation of the features of each coalition."""
        coalitions = as_bool_coalitions(coalitions)
        return np.array(
            [total_correlation(self.data_discrete[:, coalition]) for coalition in coalitions]
        )
