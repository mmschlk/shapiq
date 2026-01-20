"""KNN Classifier Explainer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np

from shapiq import InteractionValues
from shapiq.explainer.nn.base import NNExplainerBase

from ._util import (
    assert_valid_index_and_order,
    warn_ignored_parameters,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.neighbors import KNeighborsClassifier

    from shapiq.explainer.custom_types import ExplainerIndices


class KNNExplainer(NNExplainerBase):
    r"""Explainer for unweighted KNN models.

    Implements the algorithm proposed by :footcite:t:`Jia.2019` to efficiently calculate Shapley values for unweighted KNN models.
    The algorithm itself has a linear time complexity, but requires sorting training points by distance to the test
    point, resulting in a time complexity of :math:`O(N \log N)` for explaining a single data point.

    References:
        .. footbibliography::
    """

    model: KNeighborsClassifier

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
        if model.weights != "uniform":
            msg = f"KNeighborsClassifier must use weights='uniform', but has weights='{model.weights}'"
            raise ValueError(msg)
        if not isinstance(model.n_neighbors, int):
            msg = f"Expected KNeighborsClassifier.n_neighbors to be int but got {type(model.n_neighbors)}"
            raise TypeError(msg)

        super().__init__(model, class_index=class_index)
        self.k = model.n_neighbors

    @override
    def explain_function(self, x: npt.NDArray[np.floating]) -> InteractionValues:
        n = len(self.X_train)
        sv = np.zeros(n)

        sortperm = self.model.kneighbors(x.reshape(1, -1), n_neighbors=n, return_distance=False)
        sortperm = sortperm[0]

        y_train_indices_sorted = self.y_train_indices[sortperm]
        # Compute indicator function of whether a training point's class agrees with the class to explain
        y_train_is_class_index = (y_train_indices_sorted == self.class_index).astype(int)

        sv[-1] = y_train_is_class_index[-1] / n

        for i in range(n - 2, -1, -1):
            sv[i] = sv[i + 1] + (
                (y_train_is_class_index[i] - y_train_is_class_index[i + 1]) / self.k
            ) * (min(self.k, (i + 1)) / (i + 1))

        inv_sortperm = np.zeros_like(sortperm)
        inv_sortperm[sortperm] = np.arange(sortperm.shape[0])

        return InteractionValues.from_first_order_array(sv[inv_sortperm], index="SV")
