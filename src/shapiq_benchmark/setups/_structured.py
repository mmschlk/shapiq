"""Setups of the model-specific games, whose exact values the structured computers compute."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from shapiq_benchmark.models import ModelName, TreeModelName, build_model
from shapiq_games import (
    InterventionalTreeGame,
    KNNGame,
    PathDependentTreeGame,
    ProductKernelGame,
    ThresholdNNGame,
    WeightedKNNGame,
)
from shapiq_games._base import resolve_x

from ._base import ModelSetup, TabularSetup

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "InterventionalTreeSetup",
    "KNNSetup",
    "PathDependentTreeSetup",
    "ProductKernelSetup",
    "ThresholdNNSetup",
    "WeightedKNNSetup",
]


@dataclass(frozen=True, kw_only=True)
class PathDependentTreeSetup(ModelSetup, name="path_dependent_tree"):
    """A :class:`~shapiq_games.PathDependentTreeGame` of a tree model trained on a dataset.

    The explained point is taken from the test split.

    Attributes:
        model: A tree model name: ``"decision_tree"`` (default), ``"random_forest"``,
            ``"xgboost"``, ``"lightgbm"``, or ``"catboost"``.
        x: The index of the explained point in the test split. Defaults to ``0``.
        class_index: The explained class for classifiers (``None`` means class ``1``).
        normalize: Whether to center the game. Defaults to ``True``.

    Examples:
        >>> setup = PathDependentTreeSetup(dataset="xor", model="random_forest")
        >>> setup.build().n_players
        4
    """

    model: TreeModelName = "decision_tree"
    x: int = 0
    class_index: int | None = None
    normalize: bool = True

    def build(self) -> PathDependentTreeGame:
        """Train the model and build the game."""
        split = self.load_split()
        return PathDependentTreeGame(
            self.fit(split),
            resolve_x(self.x, split.x_test),
            class_index=self.class_index,
            normalize=self.normalize,
        )


@dataclass(frozen=True, kw_only=True)
class InterventionalTreeSetup(ModelSetup, name="interventional_tree"):
    """An :class:`~shapiq_games.InterventionalTreeGame` of a tree model trained on a dataset.

    The reference data is a seeded random subset of the training split; the explained point is
    taken from the test split.

    Attributes:
        model: A tree model name (see :class:`PathDependentTreeSetup`). Defaults to
            ``"decision_tree"``.
        x: The index of the explained point in the test split. Defaults to ``0``.
        n_reference: The number of reference rows. Defaults to ``100``.
        class_index: The explained class for classifiers (``None`` means class ``1``).
        normalize: Whether to center the game. Defaults to ``False``.

    Examples:
        >>> setup = InterventionalTreeSetup(dataset="xor", n_reference=20)
        >>> setup.build().reference_data.shape
        (20, 4)
    """

    model: TreeModelName = "decision_tree"
    x: int = 0
    n_reference: int = 100
    class_index: int | None = None
    normalize: bool = False

    def build(self) -> InterventionalTreeGame:
        """Train the model and build the game."""
        split = self.load_split()
        rows = self.sample_rows(split.x_train.shape[0], self.n_reference)
        return InterventionalTreeGame(
            self.fit(split),
            split.x_train[rows],
            resolve_x(self.x, split.x_test),
            class_index=self.class_index,
            normalize=self.normalize,
        )


@dataclass(frozen=True, kw_only=True)
class _NearestNeighborSetup(TabularSetup):
    """Shared fields: ``n_train`` training points as players, a point from the test split."""

    model_name: ClassVar[ModelName]
    tasks = ("classification",)

    n_train: int = 10
    x: int = 0
    class_index: int | None = None
    model_params: dict[str, Any] = field(default_factory=dict)

    def _fit(self) -> tuple[Any, np.ndarray]:
        """Fit the nearest-neighbor model on the players; return it and the explained point."""
        split = self.load_split()
        indices = self.stratified_rows(split.y_train, self.n_train, "classification")
        model = build_model(self.model_name, "classification", **self.model_params)
        model.fit(split.x_train[indices], split.y_train[indices])
        return model, resolve_x(self.x, split.x_test)


@dataclass(frozen=True, kw_only=True)
class KNNSetup(_NearestNeighborSetup, name="knn"):
    """A :class:`~shapiq_games.KNNGame` with ``n_train`` training points of a classification dataset.

    The players are drawn from the training split (seeded, stratified where possible); the
    explained point is taken from the test split.

    Attributes:
        n_train: The number of training points, i.e. players. Defaults to ``10``.
        x: The index of the explained point in the test split. Defaults to ``0``.
        class_index: The explained class (``None`` means class ``1``).
        model_params: Parameters of the model (e.g. ``{"n_neighbors": 3}``).

    Examples:
        >>> KNNSetup(dataset="xor", n_train=8).build().n_players
        8
    """

    model_name = "knn"

    def build(self) -> KNNGame:
        """Fit the model on the players and build the game."""
        model, point = self._fit()
        return KNNGame(model, point, class_index=self.class_index)


@dataclass(frozen=True, kw_only=True)
class WeightedKNNSetup(_NearestNeighborSetup, name="weighted_knn"):
    """A :class:`~shapiq_games.WeightedKNNGame` with ``n_train`` training points.

    Attributes:
        n_train: The number of training points, i.e. players. Defaults to ``10``.
        x: The index of the explained point in the test split. Defaults to ``0``.
        class_index: The explained class (``None`` means class ``1``).
        model_params: Parameters of the model (e.g. ``{"n_neighbors": 3}``).
        n_bits: The weight discretization of the game (``None`` for exact weights); set it to
            compare with the weighted KNN explainer.

    Examples:
        >>> WeightedKNNSetup(dataset="breast_cancer", n_train=8, n_bits=3).build().n_players
        8
    """

    model_name = "weighted_knn"

    n_bits: int | None = None

    def build(self) -> WeightedKNNGame:
        """Fit the model on the players and build the game."""
        model, point = self._fit()
        return WeightedKNNGame(model, point, class_index=self.class_index, n_bits=self.n_bits)


@dataclass(frozen=True, kw_only=True)
class ThresholdNNSetup(_NearestNeighborSetup, name="threshold_nn"):
    """A :class:`~shapiq_games.ThresholdNNGame` with ``n_train`` training points.

    Attributes:
        n_train: The number of training points, i.e. players. Defaults to ``10``.
        x: The index of the explained point in the test split. Defaults to ``0``.
        class_index: The explained class (``None`` means class ``1``).
        model_params: Parameters of the model (e.g. ``{"radius": 1.0}``).

    Examples:
        >>> setup = ThresholdNNSetup(dataset="xor", n_train=8, model_params={"radius": 1.0})
        >>> setup.build().n_players
        8
    """

    model_name = "threshold_nn"

    def build(self) -> ThresholdNNGame:
        """Fit the model on the players and build the game."""
        model, point = self._fit()
        return ThresholdNNGame(model, point, class_index=self.class_index)


@dataclass(frozen=True, kw_only=True)
class ProductKernelSetup(ModelSetup, name="product_kernel"):
    """A :class:`~shapiq_games.ProductKernelGame` of a kernel model trained on a dataset.

    The model is fitted on ``n_train`` seeded training points (kernel models scale poorly with the
    number of training points); the explained point is taken from the test split.

    Attributes:
        model: ``"svm"`` (default) or ``"gaussian_process"`` (regression only).
        x: The index of the explained point in the test split. Defaults to ``0``.
        n_train: The number of training points of the model. Defaults to ``500``.
        normalize: Whether to center the game. Defaults to ``False``.

    Examples:
        >>> ProductKernelSetup(dataset="breast_cancer", n_train=100).build().n_players
        30
    """

    model: Literal["svm", "gaussian_process"] = "svm"
    x: int = 0
    n_train: int = 500
    normalize: bool = False

    def build(self) -> ProductKernelGame:
        """Fit the model on the sampled training points and build the game."""
        split = self.load_split()
        rows = self.sample_rows(split.x_train.shape[0], self.n_train)
        model = self.estimator(split.task)
        model.fit(split.x_train[rows], split.y_train[rows])
        return ProductKernelGame(model, resolve_x(self.x, split.x_test), normalize=self.normalize)
