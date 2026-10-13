"""The data valuation game of a distance-weighted k-nearest-neighbor classifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq_games._base import as_bool_coalitions

from ._base import KNNGameBase, keep_first_n

if TYPE_CHECKING:
    from sklearn.neighbors import KNeighborsClassifier

    from shapiq.typing import CoalitionMatrix, FloatVector, GameValues

__all__ = ["WeightedKNNGame"]


def quantize_weights(weights: FloatVector, n_bits: int) -> FloatVector:
    """Round weights to multiples of ``2**-n_bits``.

    This is the weight discretization of :class:`~shapiq.explainer.nn.WeightedKNNExplainer`
    (discretizing and undiscretizing a weight rounds it to the grid of ``2**-n_bits``). It is part
    of the definition of the explained model and is reimplemented here so that the game does not
    depend on the explainer it checks.
    """
    scale = 2**n_bits
    return np.round(weights * scale) / scale


class WeightedKNNGame(KNNGameBase):
    """The utility game of a distance-weighted k-nearest-neighbor classifier.

    The players are the training points. The game averages the binary games of the explained
    class against every other class (Wang et al., 2024): the value of a coalition in a binary game
    is one if, among the coalition's ``k`` nearest neighbors of the two classes, the summed weight
    of the explained class is at least the summed weight of the other class, and zero otherwise
    (equation 15, with zero for the empty coalition). Its exact Shapley values are computed by
    :class:`~shapiq.explainer.nn.WeightedKNNExplainer`.

    Attributes:
        n_bits: The number of bits of the weight discretization, or ``None`` for exact weights.
        weights: The weights of the sorted training points in ``[0, 1]`` (quantized to
            ``n_bits``).
        other_classes: The classes the explained class is compared with.

    Examples:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, n_features=5, random_state=0)
        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> model = KNeighborsClassifier(n_neighbors=3, weights="distance").fit(X[:10], y[:10])
        >>> game = WeightedKNNGame(model, x=X[10], n_bits=3)  # weights rounded to 3 bits
        >>> game.n_players
        10
    """

    model_weights = "distance"

    def __init__(
        self,
        model: KNeighborsClassifier,
        x: FloatVector,
        *,
        class_index: int | None = None,
        n_bits: int | None = None,
    ) -> None:
        """Initialize the game.

        Args:
            model: The fitted distance-weighted nearest-neighbor classifier.
            x: The explained point.
            class_index: The explained class. Defaults to ``None``, which means class ``1``.
            n_bits: The number of bits of the weight discretization of the explainer, or ``None``
                for exact weights.

        Raises:
            ValueError: If the model does not use ``weights="distance"``, or its training data has
                one class only.
        """
        super().__init__(model, x, class_index=class_index)
        self.n_bits = n_bits
        if self.n_classes == 1:
            msg = "Training data must include at least two classes, but got only one."
            raise ValueError(msg)
        # as in scikit-learn: points at distance zero get weight one and all others zero
        zero_distance = np.isclose(self.distances, 0)
        if np.any(zero_distance):
            weights = zero_distance.astype(float)
        else:
            weights = self.distances[0] / self.distances
        self.weights = weights if n_bits is None else quantize_weights(weights, n_bits)
        self.other_classes = [c for c in range(self.n_classes) if c != self.class_index]

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Average the binary games of the explained class against every other class."""
        coalitions = as_bool_coalitions(coalitions)
        values = [self._binary_utilities(coalitions, other) for other in self.other_classes]
        return np.sum(values, axis=0) / (self.n_classes - 1)

    def _binary_utilities(self, coalitions: CoalitionMatrix, other: int) -> GameValues:
        """Return whether the explained class outweighs ``other`` among each coalition's neighbors."""
        is_class = self.y_train_sorted == self.class_index
        is_other = self.y_train_sorted == other
        relevant = coalitions[:, self.sortperm] & (is_class | is_other)
        nearest = keep_first_n(relevant, self.k)
        class_weight = np.sum(np.where(nearest & is_class, self.weights, 0.0), axis=1)
        other_weight = np.sum(np.where(nearest & is_other, self.weights, 0.0), axis=1)
        # at least as heavy, allowing for floating point error; the empty coalition has value zero
        wins = (class_weight >= other_weight) | np.isclose(class_weight, other_weight)
        return (wins & relevant.any(axis=1)).astype(float)
