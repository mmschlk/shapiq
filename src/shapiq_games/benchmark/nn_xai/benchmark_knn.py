"""Implementation of the benchmark for KNNExplainer."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from typing_extensions import override

import numpy as np

from shapiq import Game

from ._util import keep_first_n

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.neighbors import KNeighborsClassifier


class KNNExplainerXAI(Game):
    """Benchmark game for the KNNExplainer."""

    def __init__(
        self,
        model: KNeighborsClassifier,
        x: npt.NDArray[np.floating],
        class_index: int,
    ) -> None:
        """Initialize the class.

        Args:
            model: The KNN model to base explanations on.
            x: The data point to explain.
            class_index: The index of the class to explain.
        """
        self.model = model
        self.class_index = class_index

        X_train = self.model._fit_X  # noqa: SLF001
        if not isinstance(X_train, np.ndarray):
            msg = (
                f"Expected model's training data (_fit_X) to be np.ndarray but got {type(X_train)}"
            )
            raise TypeError(msg)
        self.X_train = cast("np.ndarray", X_train)

        y_train = model._y  # noqa: SLF001
        if not isinstance(y_train, np.ndarray):
            msg = f"Expected model's training data labels (_y) to be np.ndarray but got {type(X_train)}"
            raise TypeError(msg)

        self.y_train_indices = cast("np.ndarray", y_train)
        self.k: int = self.knn_model.n_neighbors  # type: ignore[attr-defined]

        self.sortperm = self.model.kneighbors(
            x.reshape(1, -1), n_neighbors=self.X_train.shape[0], return_distance=False
        )[0]
        self.y_train_sorted = self.y_train_indices[self.sortperm]

    @override
    def value_function(self, coalitions: npt.NDArray[np.bool]) -> npt.NDArray[np.floating]:
        utilities = np.zeros(coalitions.shape[0])

        for coalition in coalitions:
            coalition_first_k = keep_first_n(coalition, n=self.k)
            # TODO(Zaphoood): Handle case where N < k  # noqa: TD003
            utility = np.sum(self.y_train_sorted[coalition_first_k] == self.class_index) / self.k
            coalition_tuple = tuple(sorted(self.sortperm[coalition]))
            utilities[coalition_tuple] = utility

        return utilities
