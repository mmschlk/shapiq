"""Shared base of the nearest-neighbor data valuation games."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np

from shapiq.explainer.nn._util import assert_enough_training_samples
from shapiq.game import Game
from shapiq_games._base import ConfigMixin, resolve_class_index, resolve_x
from shapiq_games._setup import configure

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


class NNGameBase(ConfigMixin, Game):
    """Base of the games whose players are the training points of a nearest-neighbor classifier.

    Attributes:
        model: The fitted nearest-neighbor classifier.
        x: The explained (validation) point.
        class_index: The explained class.
        X_train: The training points (the players).
        y_train_indices: The class index of every training point.
        n_classes: The number of classes.
    """

    #: The name of the nearest-neighbor model in :mod:`shapiq_games.models` used by ``from_config``.
    _model_name: str = "knn"

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

    @classmethod
    def from_config(
        cls,
        *,
        dataset: str,
        n_train: int = 10,
        x: int = 0,
        class_index: int | None = None,
        random_state: int = 42,
        test_size: float = 0.2,
        model_params: dict[str, Any] | None = None,
        **game_params: Any,
    ) -> Self:
        """Build the game for a registered classification dataset.

        The players are ``n_train`` training points drawn (seeded, stratified where possible)
        from the training split; the model is fitted on them and the explained point is taken
        from the test split.

        Args:
            dataset: The name of a classification dataset.
            n_train: The number of training points, i.e. players. Defaults to ``10``.
            x: The index of the explained point in the test split. Defaults to ``0``.
            class_index: The explained class (``None`` means class ``1``).
            random_state: The seed of the split and the training-point sample.
            test_size: The fraction of the data used as test set. Defaults to ``0.2``.
            model_params: Parameters of the nearest-neighbor model (e.g. ``n_neighbors``).
            **game_params: Further parameters of the game (e.g. ``n_bits``).

        Returns:
            The configured game.
        """
        from sklearn.model_selection import train_test_split

        from shapiq_games.models import build_model

        setup = configure(
            dataset=dataset, model=None, random_state=random_state, test_size=test_size
        )
        split = setup.split
        if split.task != "classification":
            msg = f"{cls.__name__} needs a classification dataset, got '{dataset}'."
            raise ValueError(msg)
        model_params = dict(model_params or {})
        indices = np.arange(split.x_train.shape[0])
        if n_train < indices.shape[0]:
            try:
                indices, _ = train_test_split(
                    indices, train_size=n_train, random_state=random_state, stratify=split.y_train
                )
            except ValueError:  # too few points per class to stratify
                indices, _ = train_test_split(
                    indices, train_size=n_train, random_state=random_state
                )
        indices = np.sort(indices)
        model = build_model(cls._model_name, "classification", **model_params)
        model.fit(split.x_train[indices], split.y_train[indices])
        game = cls(model, resolve_x(x, split.x_test), class_index, **game_params)
        return game._set_config(
            **setup.config,
            n_train=n_train,
            x=x,
            class_index=class_index,
            nn_model_params=model_params,
            **game_params,
        )


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
