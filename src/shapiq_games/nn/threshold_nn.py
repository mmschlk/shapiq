"""The data valuation game of a threshold (radius) nearest-neighbor classifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.neighbors import RadiusNeighborsClassifier

from shapiq_games._base import as_bool_coalitions

from ._base import NNGameBase

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix, FloatVector, GameValues

__all__ = ["ThresholdNNGame"]


class ThresholdNNGame(NNGameBase[RadiusNeighborsClassifier]):
    """The utility game of a threshold (radius) nearest-neighbor classifier.

    The players are the training points. The value of a coalition is the share of the explained
    class among the coalition's training points within the radius of ``x``, and ``1 / n_classes``
    if there are none (equation 3 of Wang et al., 2023). Its exact Shapley values are computed by
    :class:`~shapiq.explainer.nn.ThresholdNNExplainer`.

    Examples:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, n_features=5, random_state=0)
        >>> from sklearn.neighbors import RadiusNeighborsClassifier
        >>> model = RadiusNeighborsClassifier(radius=3.0).fit(X[:10], y[:10])
        >>> game = ThresholdNNGame(model, x=X[10])
        >>> game.n_players
        10
    """

    def __init__(
        self,
        model: RadiusNeighborsClassifier,
        x: FloatVector,
        *,
        class_index: int | None = None,
    ) -> None:
        """Initialize the game.

        Args:
            model: The fitted radius nearest-neighbor classifier.
            x: The explained point.
            class_index: The explained class. Defaults to ``None``, which means class ``1``.
        """
        super().__init__(model, x, class_index=class_index)
        neighbors = model.radius_neighbors(self.x.reshape(1, -1), return_distance=False)[0]
        self.in_neighborhood = np.zeros(self.n_train, dtype=bool)
        self.in_neighborhood[neighbors] = True
        self.is_class = self.y_train_indices == self.class_index

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the share of the explained class among the coalition's points in the radius."""
        in_radius = as_bool_coalitions(coalitions) & self.in_neighborhood
        n_in_radius = np.sum(in_radius, axis=1)
        n_of_class = np.sum(in_radius & self.is_class, axis=1)
        share = n_of_class / np.maximum(n_in_radius, 1)
        return np.where(n_in_radius == 0, 1 / self.n_classes, share)
