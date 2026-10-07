"""Shared base of the nearest-neighbor data valuation games."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from shapiq.explainer.nn._util import assert_enough_training_samples
from shapiq.game import Game
from shapiq_games._base import resolve_class_index

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

__all__ = ["KNNGameBase", "NNGameBase", "keep_first_n"]


def keep_first_n(mask: npt.NDArray[np.bool_], n: int) -> npt.NDArray[np.bool_]:
    """Return a copy of ``mask`` in which only the first ``n`` ``True`` entries are kept.

    Returns ``mask`` itself if it has at most ``n`` ``True`` entries.
    """
    if n == 0:
        return np.zeros_like(mask)
    n_true = 0
    for i, value in enumerate(mask):
        n_true += int(value)
        if n_true == n:
            out = np.zeros_like(mask)
            out[: i + 1] = mask[: i + 1]
            return out
    return mask


class NNGameBase(Game):
    """Base of the games whose players are the training points of a nearest-neighbor classifier.

    Attributes:
        model: The fitted nearest-neighbor classifier.
        x: The explained (validation) point.
        class_index: The explained class.
        X_train: The training points (the players).
        y_train_indices: The class index of every training point.
        n_classes: The number of classes.
    """

    def __init__(
        self,
        model: KNeighborsClassifier | RadiusNeighborsClassifier,
        x: npt.NDArray[np.floating],
        class_index: int | None = None,
    ) -> None:
        """Initialize the game.

        Args:
            model: The fitted nearest-neighbor classifier.
            x: The explained point of shape ``(n_features,)``.
            class_index: The explained class. Defaults to ``None``, which means class ``1``.
        """
        self.model = model
        self.x = np.asarray(x).reshape(-1)
        self.class_index = cast("int", resolve_class_index(model, class_index))

        x_train = model._fit_X  # noqa: SLF001
        if not isinstance(x_train, np.ndarray):
            msg = (
                "Expected model's training data (model._fit_X) to be np.ndarray but got "
                f"{type(x_train)}"
            )
            raise TypeError(msg)
        if not (
            np.issubdtype(x_train.dtype, np.floating) or np.issubdtype(x_train.dtype, np.integer)
        ):
            msg = (
                "Expected dtype of model's training features (model._fit_X) to be a subtype of "
                f"np.floating or np.integer, but got {x_train.dtype}"
            )
            raise TypeError(msg)
        if np.issubdtype(x_train.dtype, np.integer):
            x_train = x_train.astype(np.float32)
        self.X_train = x_train

        y_train_indices = model._y  # noqa: SLF001
        if not isinstance(y_train_indices, np.ndarray):
            msg = (
                "Expected model's training data class indices (model._y) to be np.ndarray but got "
                f"{type(y_train_indices)}"
            )
            raise TypeError(msg)
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
        self.y_train_classes = np.asarray(model.classes_)
        self.n_classes = self.y_train_classes.shape[0]
        super().__init__(n_players=self.X_train.shape[0], normalize=False)


class KNNGameBase(NNGameBase):
    """Base of the games of k-nearest-neighbor classifiers.

    Attributes:
        k: The number of neighbors of the model.
        sortperm: The training points sorted by increasing distance to ``x``.
        y_train_sorted: The class indices of the sorted training points.
    """

    def __init__(
        self,
        model: KNeighborsClassifier,
        x: npt.NDArray[np.floating],
        class_index: int | None = None,
    ) -> None:
        """Initialize the game (see :class:`NNGameBase`)."""
        super().__init__(model, x, class_index)
        self.k: int = int(model.n_neighbors)  # type: ignore[arg-type]
        assert_enough_training_samples(self.k, self.X_train.shape[0])
        self.knn_model = model
        self.sortperm = model.kneighbors(
            self.x.reshape(1, -1), n_neighbors=self.X_train.shape[0], return_distance=False
        )[0]
        self.y_train_sorted = self.y_train_indices[self.sortperm]
