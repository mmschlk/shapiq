"""The data valuation game of a k-nearest-neighbor classifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq_games._base import as_bool_coalitions

from ._base import KNNGameBase, keep_first_n

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix, GameValues

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

    model_weights = "uniform"

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the share of the explained class among the coalition's k nearest neighbors."""
        coalitions = as_bool_coalitions(coalitions)
        k_nearest = keep_first_n(coalitions[:, self.sortperm], n=self.k)
        n_of_class = np.sum(k_nearest & (self.y_train_sorted == self.class_index), axis=1)
        return n_of_class / self.k
