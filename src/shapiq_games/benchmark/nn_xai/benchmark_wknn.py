"""Implementation of the benchmark for WeightedKNNExplainer."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from typing_extensions import override

import numpy as np

from shapiq.explainer.nn.weighted_knn import WeightedKNNExplainer
from shapiq_games.benchmark.nn_xai.base import NNBenchmarkBase

from ._util import keep_first_n

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.neighbors import KNeighborsClassifier

    from shapiq.typing import GameValues


class WeightedKNNExplainerXAI(NNBenchmarkBase):
    """Benchmark game for the WeightedKNNExplainer."""

    def __init__(
        self,
        model: KNeighborsClassifier,
        x: npt.NDArray[np.floating],
        class_index: int,
        n_bits: int | None = None,
    ) -> None:
        """Initialize the class.

        Args:
            model: The model to explain.
            x: The data point to explain.
            class_index: Index of the class to explain.
            n_bits: Number of bits to use for discretization.
        """
        super().__init__(model, x, class_index)
        self.n_bits = n_bits
        self.n_classes = self.y_train_classes.shape[0]
        if self.n_classes == 1:
            msg = "Training data must include at least two classes, but got only one."
            raise ValueError(msg)

        self.binary_games = {
            other_class_index: BinaryWeightedKNNExplainerXAI(
                model, x, class_index, other_class_index, n_bits
            )
            for other_class_index in range(self.n_classes)
            if class_index != other_class_index
        }

    @override
    def value_function(self, coalitions: npt.NDArray[np.bool]) -> GameValues:
        binary_game_values = [
            game.value_function(coalitions) for game in self.binary_games.values()
        ]

        return sum(binary_game_values) / (self.n_classes - 1)


class BinaryWeightedKNNExplainerXAI(NNBenchmarkBase):
    """Benchmark game for the WeightedKNNExplainer in the binary case."""

    def __init__(
        self,
        model: KNeighborsClassifier,
        x: npt.NDArray[np.floating],
        class_index: int,
        class_index_other: int,
        n_bits: int | None = None,
    ) -> None:
        """Initialize the class.

        The binary game considers only two out of all (possibly more than two) class labels, specified by ``class_index`` and
        ``class_index_other``, where the former is the one for which utilities will be returned.

        Args:
            model: The model to explain.
            x: The data point to explain.
            class_index: Index of the class to explain.
            class_index_other: Index of the other class in the binary game.
            n_bits: Number of bits to use for discretization.
        """
        super().__init__(model, x, class_index)
        self.class_index_other = class_index_other
        self.n_bits = n_bits

    @override
    def value_function(self, coalitions: npt.NDArray[np.bool]) -> GameValues:
        n_classes = len(self.y_train_classes)
        # TODO(Zaphoood): Allow n_classes == 1 # noqa: TD003
        expected_classes = 2
        if n_classes != expected_classes:
            msg = f"Expected exactly {expected_classes} classes but got {n_classes}"
            raise ValueError(msg)

        sortperm, weights = _get_normalized_weights(self.model, self.X_train, self.x)

        if self.n_bits is not None:
            _explainer = WeightedKNNExplainer(self.model, self.class_index, n_bits=self.n_bits)
            weights = _explainer._undiscretize_weight(  # noqa: SLF001
                _explainer._discretize_weight(weights)  # noqa: SLF001
            )

        y_train_sorted = self.y_train_indices[sortperm]
        y_val_mask = y_train_sorted == self.class_index
        y_other_mask = y_train_sorted == self.class_index_other

        # This is the utility function according to equation (15) in Wang et al. (2024), with the modification that the utility of the empty set is zero.
        utilities = np.zeros(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            coalition_relevant_class = coalition & (y_val_mask | y_other_mask)
            if not np.any(coalition_relevant_class):
                # Empty coalition must be zero
                utilities[i] = 0
                continue

            # Mask of k nearest training points of current coalition with class y_val or y_other
            k_nearest_with_relevant_class = keep_first_n(coalition_relevant_class, self.k)
            y_val_nearest = y_val_mask & k_nearest_with_relevant_class
            y_other_nearest = y_other_mask & k_nearest_with_relevant_class
            utilities[i] = int(
                _greater_or_close(np.sum(weights[y_val_nearest]), np.sum(weights[y_other_nearest]))
            )

        return utilities


def _greater_or_close(a: np.floating, b: np.floating) -> np.bool:
    """Returns ``a >= b`` but allows for floating point error.

    That is, if ``a < b`` but ``np.isclose(a, b)``, ``True`` will be returned.
    """
    return cast("np.bool", a >= b) or np.isclose(a, b)


def _get_normalized_weights(
    model: KNeighborsClassifier, X_train: npt.NDArray, x_val: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
    """Calculate normalized weights of training data points with respect to a validation data point.

    Args:
        model: The KNN model fitted to the training data.
        X_train: The training data.
        x_val: The validation data point.

    Returns:
        A tuple ``(sortperm, weights)``, where both are ``numpy.ndarray``s with dimensions ``(n_training_samples,)`` and
            - ``sortperm`` is a permutation that sorts the training data points by decreasing weight
            - ``weights`` contains the weights for each training data point, normalized to the interval [0, 1]
    """
    distances, sortperm = model.kneighbors(
        x_val.reshape(1, -1), n_neighbors=X_train.shape[0], return_distance=True
    )
    distances = distances[0]
    sortperm = sortperm[0]

    # Replicate sklearn behavior: if any training points are zero distance
    # from the test point, those points get weight 1 and all others 0
    zero_dist = np.isclose(distances, 0)
    if np.any(zero_dist):
        weights = np.zeros_like(distances)
        weights[zero_dist] = 1
    else:
        weights = distances[0] / distances

    return sortperm, weights
