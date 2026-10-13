"""Shared base of the nearest-neighbor data valuation games."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

from shapiq.explainer.nn._util import assert_enough_training_samples
from shapiq.game import Game
from shapiq_games._base import resolve_class_index

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix, FloatVector

__all__ = ["KNNGameBase", "NNGameBase", "keep_first_n"]


def keep_first_n(masks: CoalitionMatrix, n: int) -> CoalitionMatrix:
    """Return ``masks`` with only the first ``n`` ``True`` entries of every row kept.

    Args:
        masks: A boolean matrix of shape ``(n_rows, n_columns)``.
        n: The number of ``True`` entries to keep per row.

    Returns:
        A new boolean matrix of the same shape.
    """
    return masks & (np.cumsum(masks, axis=1, dtype=np.int32) <= n)


class NNGameBase[M: KNeighborsClassifier | RadiusNeighborsClassifier](Game):
    """Base of the games whose players are the training points of a nearest-neighbor classifier.

    Attributes:
        model: The fitted nearest-neighbor classifier.
        x: The explained (validation) point.
        class_index: The explained class.
        n_train: The number of training points (the players).
        y_train_indices: The class index of every training point.
        n_classes: The number of classes.
    """

    def __init__(
        self,
        model: M,
        x: FloatVector,
        *,
        class_index: int | None = None,
    ) -> None:
        """Initialize the game.

        Args:
            model: The fitted nearest-neighbor classifier.
            x: The explained point of shape ``(n_features,)``.
            class_index: The explained class. Defaults to ``None``, which means class ``1``.

        Raises:
            TypeError: If the model's class indices (``model._y``) are not integers.
            ValueError: If the model has several outputs.
        """
        self.model: M = model
        self.x = np.asarray(x).reshape(-1)
        self.class_index = cast("int", resolve_class_index(model, class_index))
        self.n_train = int(model.n_samples_fit_)
        y_train_indices = np.asarray(model._y)  # noqa: SLF001
        if not np.issubdtype(y_train_indices.dtype, np.integer):
            msg = (
                "Expected dtype of model's training class indices (model._y) to be a subtype of "
                f"np.integer, but got {y_train_indices.dtype}"
            )
            raise TypeError(msg)
        if y_train_indices.ndim != 1:
            msg = (
                "Multi-output nearest neighbor classifiers are not supported. Make sure to pass "
                "the training labels as a 1D vector when calling `model.fit()`."
            )
            raise ValueError(msg)
        self.y_train_indices = y_train_indices
        self.n_classes = len(model.classes_)
        super().__init__(n_players=self.n_train, normalize=False)


class KNNGameBase(NNGameBase[KNeighborsClassifier]):
    """Base of the games of k-nearest-neighbor classifiers.

    Attributes:
        k: The number of neighbors of the model.
        sortperm: The training points sorted by increasing distance to ``x``.
        distances: The distances of the sorted training points to ``x``.
        y_train_sorted: The class indices of the sorted training points.
    """

    model_weights: ClassVar[str]
    """The ``weights`` of the classifiers the game describes (as the matching explainer)."""

    def __init__(
        self,
        model: KNeighborsClassifier,
        x: FloatVector,
        *,
        class_index: int | None = None,
    ) -> None:
        """Initialize the game (see :class:`NNGameBase`).

        Raises:
            ValueError: If the model weights its neighbors otherwise than the game.
        """
        if model.weights != self.model_weights:
            msg = (
                f"{type(self).__name__} describes a KNeighborsClassifier with "
                f"weights={self.model_weights!r}, but the model has weights={model.weights!r}."
            )
            raise ValueError(msg)
        super().__init__(model, x, class_index=class_index)
        self.k: int = int(model.n_neighbors)  # type: ignore[arg-type]
        assert_enough_training_samples(self.k, self.n_train)
        distances, sortperm = model.kneighbors(
            self.x.reshape(1, -1), n_neighbors=self.n_train, return_distance=True
        )
        self.distances, self.sortperm = distances[0], sortperm[0]
        self.y_train_sorted = self.y_train_indices[self.sortperm]
