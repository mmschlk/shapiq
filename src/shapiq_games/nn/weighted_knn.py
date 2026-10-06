"""The data valuation game of a distance-weighted k-nearest-neighbor classifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq_games._base import as_bool_coalitions

from ._base import KNNGameBase, keep_first_n

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.neighbors import KNeighborsClassifier

__all__ = ["BinaryWeightedKNNGame", "WeightedKNNGame"]


def _greater_or_close(a: float, b: float) -> bool:
    """Return ``a >= b``, allowing for floating point error."""
    return bool(a >= b or np.isclose(a, b))


def quantize_weights(weights: np.ndarray, n_bits: int) -> np.ndarray:
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
    class against every other class (Wang et al., 2024). Its exact Shapley values are computed by
    :class:`~shapiq.explainer.nn.WeightedKNNExplainer`.

    Attributes:
        n_bits: The number of bits of the weight discretization, or ``None`` for exact weights.
        binary_games: The binary games, keyed by the other class.
    """

    _model_name = "weighted_knn"

    def __init__(
        self,
        model: KNeighborsClassifier,
        x: npt.NDArray[np.floating],
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
        """
        super().__init__(model, x, class_index)
        self.n_bits = n_bits
        if self.n_classes == 1:
            msg = "Training data must include at least two classes, but got only one."
            raise ValueError(msg)
        self.binary_games = {
            other: BinaryWeightedKNNGame(model, self.x, self.class_index, other, n_bits)
            for other in range(self.n_classes)
            if other != self.class_index
        }

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Average the binary games of the explained class against every other class."""
        coalitions = as_bool_coalitions(coalitions)
        values = [game.value_function(coalitions) for game in self.binary_games.values()]
        return np.sum(values, axis=0) / (self.n_classes - 1)


class BinaryWeightedKNNGame(KNNGameBase):
    """The binary weighted k-nearest-neighbor game of the explained class against one other class.

    The value of a coalition is one if, among the coalition's ``k`` nearest neighbors of the two
    classes, the summed weight of the explained class is at least the summed weight of the other
    class, and zero otherwise (equation 15 of Wang et al., 2024, with zero for the empty
    coalition).
    """

    _model_name = "weighted_knn"

    def __init__(
        self,
        model: KNeighborsClassifier,
        x: npt.NDArray[np.floating],
        class_index: int,
        class_index_other: int,
        n_bits: int | None = None,
    ) -> None:
        """Initialize the binary game.

        Args:
            model: The fitted distance-weighted nearest-neighbor classifier.
            x: The explained point.
            class_index: The explained class.
            class_index_other: The other class of the binary game.
            n_bits: The number of bits of the weight discretization, or ``None``.
        """
        super().__init__(model, x, class_index)
        self.class_index_other = class_index_other
        self.n_bits = n_bits

    def _normalized_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the training points sorted by decreasing weight and their weights in [0, 1]."""
        distances, sortperm = self.knn_model.kneighbors(
            self.x.reshape(1, -1), n_neighbors=self.X_train.shape[0], return_distance=True
        )
        distances, sortperm = distances[0], sortperm[0]
        # as in scikit-learn: points at distance zero get weight one and all others zero
        zero_distance = np.isclose(distances, 0)
        if np.any(zero_distance):
            weights = np.zeros_like(distances)
            weights[zero_distance] = 1
        else:
            weights = distances[0] / distances
        return sortperm, weights

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return whether the explained class outweighs the other class among the neighbors."""
        coalitions = as_bool_coalitions(coalitions)
        sortperm, weights = self._normalized_weights()
        if self.n_bits is not None:
            weights = quantize_weights(weights, self.n_bits)
        y_sorted = self.y_train_indices[sortperm]
        is_class = y_sorted == self.class_index
        is_other = y_sorted == self.class_index_other
        utilities = np.zeros(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            relevant = coalition[sortperm] & (is_class | is_other)
            if not np.any(relevant):
                continue  # the empty coalition has value zero
            nearest = keep_first_n(relevant, self.k)
            utilities[i] = int(
                _greater_or_close(
                    np.sum(weights[is_class & nearest]), np.sum(weights[is_other & nearest])
                )
            )
        return utilities
