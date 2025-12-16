"""Base for all nearest-neighbor explainer benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from typing_extensions import override

import numpy as np

from shapiq import Game

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier


class NNBenchmarkBase(Game):
    """Base class for all nearest-neighbor explainer benchmarks."""

    def __init__(
        self,
        model: KNeighborsClassifier | RadiusNeighborsClassifier,
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
        self.x = x
        self.class_index = class_index

        X_train = self.model._fit_X  # noqa: SLF001
        if not isinstance(X_train, np.ndarray):
            msg = f"Expected model's training data (model._fit_X) to be np.ndarray but got {type(X_train)}"
            raise TypeError(msg)
        self.X_train = cast("np.ndarray", X_train)

        y_train = model._y  # noqa: SLF001
        if not isinstance(y_train, np.ndarray):
            msg = f"Expected model's training data class indices (model._y) to be np.ndarray but got {type(X_train)}"
            raise TypeError(msg)
        self.y_train_indices = cast("np.ndarray", y_train)

        y_train_classes = model.classes_
        if not isinstance(y_train, np.ndarray):
            msg = f"Expected model's training data class (model.classes_) to be np.ndarray but got {type(X_train)}"
            raise TypeError(msg)
        self.y_train_classes = cast("npt.NDArray[np.object_]", y_train_classes)
        self.n_classes = self.y_train_classes.shape[0]

        self.k: int = self.model.n_neighbors  # type: ignore[attr-defined]

        super().__init__(n_players=self.X_train.shape[0])


class KNNBenchmarkBase(NNBenchmarkBase):
    """Base class for k nearest-neighbor explainer benchmarks."""

    @override
    def __init__(
        self,
        model: KNeighborsClassifier,
        x: npt.NDArray[np.floating],
        class_index: int,
    ) -> None:
        super().__init__(model, x, class_index)
        # Reassign in order to narrow type to KNeighborsClassifier
        self.model = model

        self.sortperm = self.model.kneighbors(
            x.reshape(1, -1), n_neighbors=self.X_train.shape[0], return_distance=False
        )[0]
        self.y_train_sorted = self.y_train_indices[self.sortperm]
