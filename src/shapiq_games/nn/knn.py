"""The data valuation game of a k-nearest-neighbor classifier."""

from __future__ import annotations

import numpy as np

from shapiq_games._base import as_bool_coalitions

from ._base import KNNGameBase, keep_first_n

__all__ = ["KNNGame"]


class KNNGame(KNNGameBase):
    r"""The utility game of a (uniformly weighted) k-nearest-neighbor classifier.

    The players are the training points. The value of a coalition :math:`S` is the fraction of the
    :math:`k` nearest neighbors of :math:`x` within :math:`S` that belong to the explained class
    (Jia et al., 2019). Its exact Shapley values are computed by
    :class:`~shapiq.explainer.nn.KNNExplainer`.

    Examples:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, n_features=5, random_state=0)
        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> model = KNeighborsClassifier(n_neighbors=3).fit(X[:10], y[:10])
        >>> game = KNNGame(model, x=X[10])  # the players are the 10 training points
        >>> game.n_players
        10
    """

    _model_name = "knn"

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the share of the explained class among the coalition's k nearest neighbors."""
        coalitions = as_bool_coalitions(coalitions)
        utilities = np.zeros(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            k_nearest = keep_first_n(coalition[self.sortperm], n=self.k)
            utilities[i] = np.sum(self.y_train_sorted[k_nearest] == self.class_index) / self.k
        return utilities
