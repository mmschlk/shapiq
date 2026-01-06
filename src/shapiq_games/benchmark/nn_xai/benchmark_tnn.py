"""Implementation of the benchmark for TNNExplainer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np

from .base import NNBenchmarkBase

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.neighbors import RadiusNeighborsClassifier

    from shapiq.typing import GameValues


class TNNExplainerXAI(NNBenchmarkBase):
    """Benchmark game for the KNNExplainer."""

    @override
    def __init__(
        self, model: RadiusNeighborsClassifier, x: npt.NDArray[np.floating], class_index: int
    ) -> None:
        super().__init__(model, x, class_index)
        n_train = self.X_train.shape[0]
        # Reassign in order to narrow type to RadiusNeighborsClassifier
        self.model = model

        self.neighbor_indices = self.model.radius_neighbors(x.reshape(1, -1), return_distance=False)
        self.neighbor_indices = self.neighbor_indices[0]
        self.in_neighborhood = np.zeros((n_train,), dtype=bool)
        self.in_neighborhood[self.neighbor_indices] = True

        self.y_train_is_class_index = self.y_train_indices == self.class_index

    @override
    def value_function(self, coalitions: npt.NDArray[np.bool]) -> GameValues:
        utilities = np.zeros(coalitions.shape[0])

        for i, coalition in enumerate(coalitions):
            coal_nhood = coalition & self.in_neighborhood
            coal_nhood_with_class_index = coal_nhood & self.y_train_is_class_index
            n_coal_nhood = np.sum(coal_nhood)

            # Utility function according to equation (3) in Wang et al. (2023)
            if n_coal_nhood == 0:
                utilities[i] = 1 / self.n_classes
            else:
                utilities[i] = np.sum(coal_nhood_with_class_index) / n_coal_nhood

        return utilities
