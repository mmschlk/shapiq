"""KNN Classifier Explainer."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np

from ._common_knn import _CommonKNNExplainer
from ._lookup_game import LookupGame
from ._util import interaction_values_from_array, keep_first_n, warn_ignored_parameters

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.neighbors import KNeighborsClassifier

    from shapiq import InteractionValues
    from shapiq.explainer.custom_types import ExplainerIndices


class _BruteForceNormalKNNExplainer(_CommonKNNExplainer):
    """Brute force approach to computing Shapley values for normal (unweighted) KNN models."""

    @override
    def __init__(
        self,
        model: KNeighborsClassifier,
        class_index: int | None = None,
    ) -> None:
        super().__init__(model, class_index=class_index)

    @override
    def explain_function(self, x: npt.NDArray[np.floating]) -> InteractionValues:
        utilities = {}

        sortperm = self.knn_model.kneighbors(
            x.reshape(1, -1), n_neighbors=self.X_train.shape[0], return_distance=False
        )
        sortperm = sortperm[0]
        y_train_sorted = self.y_train_indices[sortperm]

        for coalition_generator in product([False, True], repeat=self.X_train.shape[0]):
            coalition = np.array(list(coalition_generator))
            coalition_first_k = keep_first_n(coalition, n=self.k)
            utility = np.sum(y_train_sorted[coalition_first_k] == self.class_index) / self.k

            coalition_tuple = tuple(sorted(sortperm[coalition]))
            utilities[coalition_tuple] = utility

        game = LookupGame(n_players=self.X_train.shape[0], utilities=utilities)
        return game.exact_values("SV", order=1)


class KNNExplainer(_CommonKNNExplainer):
    r"""Explainer for unweighted KNN models.

    Implements the algorithm proposed by Jia et al. (2019) [Jia19]_ to efficiently calculate Shapley values for unweighted KNN models.
    The algorithm itself has a linear time complexity, but expects a sorted array of training points as input, resulting in a time complexity of :math:`O(N \log N)` for explaining a single data point.
    """

    @override
    def __init__(
        self,
        model: KNeighborsClassifier,
        class_index: int | None = None,
        data: np.ndarray | None = None,
        index: ExplainerIndices = "SV",
        max_order: int = 1,
    ) -> None:
        # TODO(Zaphoood): Check that index and max_order are valid (only first-order etc.)  # noqa: TD003
        warn_ignored_parameters(locals(), ["data"], self.__class__.__name__)

        super().__init__(model, class_index=class_index)
        model_weights = model.weights  # type: ignore [attr-defined]
        if model_weights != "uniform":
            msg = f"KNeighborsClassifier must use weights='uniform', but has weights='{model_weights}'"
            raise ValueError(msg)

    @override
    def explain_function(
        self, x: npt.NDArray[np.floating], class_index: int | None = None
    ) -> InteractionValues:
        if class_index is None:
            class_index = self.class_index

        n = len(self.X_train)
        sv = np.zeros(n)

        sortperm = self.knn_model.kneighbors(x.reshape(1, -1), n_neighbors=n, return_distance=False)
        sortperm = sortperm[0]

        y_train_indices_sorted = self.y_train_indices[sortperm]
        # Compute indicator function of whether a training point's class agrees with the class to explain
        y_train_is_class_index = (y_train_indices_sorted == class_index).astype(int)

        sv[-1] = y_train_is_class_index[-1] / n

        for i in range(n - 2, -1, -1):
            sv[i] = sv[i + 1] + (
                (y_train_is_class_index[i] - y_train_is_class_index[i + 1]) / self.k
            ) * (min(self.k, (i + 1)) / (i + 1))

        inv_sortperm = np.zeros_like(sortperm)
        inv_sortperm[sortperm] = np.arange(sortperm.shape[0])

        return interaction_values_from_array(sv[inv_sortperm])
