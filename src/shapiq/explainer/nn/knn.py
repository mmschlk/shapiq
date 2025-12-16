"""KNN Classifier Explainer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np

from shapiq.explainer.nn.base import NNExplainerBase

from ._util import (
    assert_valid_index_and_order,
    interaction_values_from_array,
    warn_ignored_parameters,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.neighbors import KNeighborsClassifier

    from shapiq import InteractionValues
    from shapiq.explainer.custom_types import ExplainerIndices


class KNNExplainer(NNExplainerBase):
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
        assert_valid_index_and_order(index, max_order)
        warn_ignored_parameters(locals(), ["data"], self.__class__.__name__)

        super().__init__(model, class_index=class_index)
        self.knn_model = model
        self.k: int = self.knn_model.n_neighbors  # type: ignore[attr-defined]
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
