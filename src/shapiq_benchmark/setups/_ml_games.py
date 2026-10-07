"""Setups of the machine learning games: explanation, selection, valuation, and data games."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shapiq_benchmark.models import build_model
from shapiq_games import (
    ClusterExplanation,
    DatasetValuation,
    DataValuation,
    EnsembleSelection,
    FeatureSelection,
    GlobalExplanation,
    LocalExplanation,
    RandomForestEnsembleSelection,
    UncertaintyExplanation,
    UnsupervisedData,
)
from shapiq_games._base import is_classifier, resolve_class_index, resolve_x

from ._base import ModelSetup, TabularSetup

if TYPE_CHECKING:
    from shapiq.imputer.base import Imputer
    from shapiq_games._training import MetricName
    from shapiq_games.uncertainty import Uncertainty

__all__ = [
    "DEFAULT_MEMBER_POOL",
    "ClusterExplanationSetup",
    "DataValuationSetup",
    "DatasetValuationSetup",
    "EnsembleSelectionSetup",
    "FeatureSelectionSetup",
    "GlobalExplanationSetup",
    "LocalExplanationSetup",
    "RandomForestEnsembleSelectionSetup",
    "UncertaintyExplanationSetup",
    "UnsupervisedDataSetup",
]

DEFAULT_MEMBER_POOL: tuple[str, ...] = (
    "linear",
    "decision_tree",
    "random_forest",
    "svm",
    "knn",
    "xgboost",
)
"""The model names :class:`EnsembleSelectionSetup` draws its members from by default."""


@dataclass(frozen=True, kw_only=True)
class LocalExplanationSetup(ModelSetup, name="local_explanation"):
    """A :class:`~shapiq_games.LocalExplanation` of a model trained on a dataset.

    The background data is a seeded random subset of ``n_background`` training rows; the explained
    point is taken from the test split. With ``imputer="tabpfn"`` the model must be ``"tabpfn"``,
    and the background rows are TabPFN's context (remove-and-recontextualize).

    Attributes:
        model: The model name. Defaults to ``"random_forest"``.
        x: The index of the explained point in the test split. Defaults to ``0``.
        imputer: ``"marginal"`` (default), ``"conditional"``, ``"baseline"``, or ``"tabpfn"``.
        n_background: The number of background (or context) rows. Defaults to ``100``.
        class_index: The explained class for classifiers (``None`` means class ``1``).
        normalize: Whether to center the game. Defaults to ``True``.

    Examples:
        >>> setup = LocalExplanationSetup(dataset="breast_cancer", model="decision_tree")
        >>> setup.build().n_players
        30
    """

    x: int = 0
    imputer: Literal["marginal", "conditional", "baseline", "tabpfn"] = "marginal"
    n_background: int = 100
    class_index: int | None = None
    normalize: bool = True

    def __post_init__(self) -> None:
        """Check that the TabPFN imputer gets a TabPFN model."""
        super().__post_init__()
        if self.imputer == "tabpfn" and self.model != "tabpfn":
            msg = f"imputer='tabpfn' needs model='tabpfn', got model={self.model!r}."
            raise ValueError(msg)

    def build(self) -> LocalExplanation:
        """Train the model, draw the background rows, and build the game."""
        split = self.load_split()
        model = self.fit(split)
        rows = self.sample_rows(split.x_train.shape[0], self.n_background)
        background = split.x_train[rows]
        point = resolve_x(self.x, split.x_test)
        imputer: str | Imputer = self.imputer
        if self.imputer == "tabpfn":
            imputer = _tabpfn_imputer(
                model, background, split.y_train[rows], split.x_test, point, self.class_index
            )
        return LocalExplanation(
            model,
            background,
            point,
            imputer=imputer,  # type: ignore[arg-type]
            class_index=self.class_index,
            random_state=self.random_state,
            normalize=self.normalize,
        )


def _tabpfn_imputer(
    model: Any,  # noqa: ANN401
    context_x: np.ndarray,
    context_y: np.ndarray,
    x_test: np.ndarray,
    x: np.ndarray,
    class_index: int | None,
) -> Imputer:
    """Build the remove-and-recontextualize imputer of a fitted TabPFN model."""
    from shapiq.explainer.utils import get_predict_function_and_model_type
    from shapiq.imputer import TabPFNImputer

    resolved = resolve_class_index(model, class_index) if is_classifier(model) else None
    predict_function, _ = get_predict_function_and_model_type(model, class_index=resolved)
    if isinstance(predict_function, Exception):
        raise predict_function
    imputer = TabPFNImputer(
        model=model,
        x_train=context_x,
        y_train=context_y,
        x_test=x_test,
        predict_function=predict_function,
    )
    imputer.fit(x)
    return imputer


@dataclass(frozen=True, kw_only=True)
class GlobalExplanationSetup(ModelSetup, name="global_explanation"):
    """A :class:`~shapiq_games.GlobalExplanation` of a model trained on a dataset.

    The evaluation rows are drawn from the test split.

    Attributes:
        model: The model name. Defaults to ``"random_forest"``.
        class_index: The explained class for classifiers (``None`` means class ``1``).
        loss: ``"mse"`` (default) or ``"mae"``.
        n_samples: The number of evaluation rows. Defaults to ``100``.
        normalize: Whether to center the game. Defaults to ``True``.

    Examples:
        >>> GlobalExplanationSetup(dataset="xor", model="decision_tree").build().n_players
        4
    """

    class_index: int | None = None
    loss: Literal["mse", "mae"] = "mse"
    n_samples: int = 100
    normalize: bool = True

    def build(self) -> GlobalExplanation:
        """Train the model and build the game on the test split."""
        split = self.load_split()
        return GlobalExplanation(
            self.fit(split),
            split.x_test,
            class_index=self.class_index,
            loss=self.loss,
            n_samples=self.n_samples,
            random_state=self.random_state,
            normalize=self.normalize,
        )


@dataclass(frozen=True, kw_only=True)
class FeatureSelectionSetup(ModelSetup, name="feature_selection"):
    """A :class:`~shapiq_games.FeatureSelection` game: a model retrained on feature subsets.

    Attributes:
        model: The model name. Defaults to ``"decision_tree"``.
        metric: The metric name, or ``None`` for the default of the task.
        n_train: Use a seeded subset of this many training rows (``None`` for all).
        normalize: Whether to center the game. Defaults to ``True``.

    Examples:
        >>> FeatureSelectionSetup(dataset="breast_cancer", n_train=100).build().n_players
        30
    """

    model: str = "decision_tree"
    metric: MetricName | None = None
    n_train: int | None = None
    normalize: bool = True

    def build(self) -> FeatureSelection:
        """Build the game with the unfitted model; the game trains it per coalition."""
        split = self.load_split()
        rows = self.sample_rows(split.x_train.shape[0], self.n_train)
        return FeatureSelection(
            self.estimator(split.task),
            split.x_train[rows],
            split.y_train[rows],
            split.x_test,
            split.y_test,
            task=split.task,
            metric=self.metric,
            normalize=self.normalize,
        )


def _stratified_rows(y: np.ndarray, n: int | None, task: str, random_state: int) -> np.ndarray:
    """Draw ``n`` row indices, stratified by class for classification where possible."""
    indices = np.arange(y.shape[0])
    if n is None or n >= indices.shape[0]:
        return indices
    stratify = y if task == "classification" else None
    try:
        rows, _ = train_test_split(
            indices, train_size=n, random_state=random_state, stratify=stratify
        )
    except ValueError:  # too few points per class to stratify
        rows, _ = train_test_split(indices, train_size=n, random_state=random_state)
    return np.sort(rows)


@dataclass(frozen=True, kw_only=True)
class DataValuationSetup(ModelSetup, name="data_valuation"):
    """A :class:`~shapiq_games.DataValuation` game with ``n_players`` training points.

    The points are drawn from the training split, stratified by class where possible; the test
    split is the test set.

    Attributes:
        model: The model name. Defaults to ``"decision_tree"``.
        n_players: The number of training points, i.e. players. Defaults to ``10``.
        metric: The metric name, or ``None`` for the default of the task.
        empty_value: The value of the empty coalition. Defaults to ``0``.
        normalize: Whether to center the game. Defaults to ``True``.

    Examples:
        >>> DataValuationSetup(dataset="xor", n_players=8).build().n_players
        8
    """

    model: str = "decision_tree"
    n_players: int = 10
    metric: MetricName | None = None
    empty_value: float = 0.0
    normalize: bool = True

    def build(self) -> DataValuation:
        """Draw the players and build the game with the unfitted model."""
        split = self.load_split()
        rows = _stratified_rows(split.y_train, self.n_players, split.task, self.random_state)
        return DataValuation(
            self.estimator(split.task),
            split.x_train[rows],
            split.y_train[rows],
            split.x_test,
            split.y_test,
            task=split.task,
            metric=self.metric,
            empty_value=self.empty_value,
            normalize=self.normalize,
        )


@dataclass(frozen=True, kw_only=True)
class DatasetValuationSetup(ModelSetup, name="dataset_valuation"):
    """A :class:`~shapiq_games.DatasetValuation` game: the training split divided into groups.

    Attributes:
        model: The model name. Defaults to ``"decision_tree"``.
        n_players: The number of groups. Defaults to ``10``.
        player_sizes: ``"uniform"`` (default), ``"increasing"``, or ``"random"``.
        n_train: Use a seeded subset of this many training rows (``None`` for all).
        metric: The metric name, or ``None`` for the default of the task.
        empty_value: The value of the empty coalition. Defaults to ``0``.
        normalize: Whether to center the game. Defaults to ``True``.

    Examples:
        >>> DatasetValuationSetup(dataset="xor", n_players=4).build().n_players
        4
    """

    model: str = "decision_tree"
    n_players: int = 10
    player_sizes: Literal["uniform", "increasing", "random"] = "uniform"
    n_train: int | None = None
    metric: MetricName | None = None
    empty_value: float = 0.0
    normalize: bool = True

    def build(self) -> DatasetValuation:
        """Split the training rows into groups and build the game with the unfitted model."""
        split = self.load_split()
        rows = _stratified_rows(split.y_train, self.n_train, split.task, self.random_state)
        return DatasetValuation(
            self.estimator(split.task),
            split.x_train[rows],
            split.y_train[rows],
            split.x_test,
            split.y_test,
            task=split.task,
            n_players=self.n_players,
            player_sizes=self.player_sizes,
            metric=self.metric,
            empty_value=self.empty_value,
            random_state=self.random_state,
            normalize=self.normalize,
        )


@dataclass(frozen=True, kw_only=True)
class EnsembleSelectionSetup(TabularSetup, name="ensemble_selection"):
    """An :class:`~shapiq_games.EnsembleSelection` game with members trained on a dataset.

    Attributes:
        members: The model name of every member. If ``None``, ``n_members`` names are drawn
            (seeded) from :data:`DEFAULT_MEMBER_POOL`.
        n_members: The number of members when ``members`` is ``None``. Defaults to ``10``.
        metric: The metric name, or ``None`` for the default of the task.
        empty_value: The value of the empty coalition. Defaults to ``0``.
        normalize: Whether to center the game. Defaults to ``True``.

    Member ``i`` is seeded with ``random_state + i``.

    Examples:
        >>> setup = EnsembleSelectionSetup(dataset="xor", members=("linear", "decision_tree"))
        >>> setup.build().n_players
        2
    """

    members: tuple[str, ...] | None = None
    n_members: int = 10
    metric: MetricName | None = None
    empty_value: float = 0.0
    normalize: bool = True

    def __post_init__(self) -> None:
        """Store the members as a tuple, so that a setup read from JSON equals the original."""
        if self.members is not None:
            object.__setattr__(self, "members", tuple(self.members))
        super().__post_init__()

    def build(self) -> EnsembleSelection:
        """Train the members and build the game on the test split."""
        split = self.load_split()
        members = self.members
        if members is None:
            rng = np.random.default_rng(self.random_state)
            members = tuple(str(name) for name in rng.choice(DEFAULT_MEMBER_POOL, self.n_members))
        fitted = []
        for i, name in enumerate(members):
            params = {"n_neighbors": 3} if name == "knn" else {}
            model = build_model(name, split.task, random_state=self.random_state + i, **params)
            fitted.append(model.fit(split.x_train, split.y_train))
        return EnsembleSelection(
            fitted,
            split.x_test,
            split.y_test,
            task=split.task,
            metric=self.metric,
            empty_value=self.empty_value,
            member_names=[f"{i}_{name}" for i, name in enumerate(members)],
            normalize=self.normalize,
        )


@dataclass(frozen=True, kw_only=True)
class RandomForestEnsembleSelectionSetup(TabularSetup, name="random_forest_ensemble_selection"):
    """A :class:`~shapiq_games.RandomForestEnsembleSelection` game: the trees of a forest.

    Attributes:
        n_members: The number of trees, i.e. players. Defaults to ``10``.
        metric: The metric name, or ``None`` for the default of the task.
        empty_value: The value of the empty coalition. Defaults to ``0``.
        model_params: Further hyperparameters of the forest.
        normalize: Whether to center the game. Defaults to ``True``.

    Examples:
        >>> RandomForestEnsembleSelectionSetup(dataset="xor", n_members=5).build().n_players
        5
    """

    n_members: int = 10
    metric: MetricName | None = None
    empty_value: float = 0.0
    model_params: dict[str, Any] = field(default_factory=dict)
    normalize: bool = True

    def build(self) -> RandomForestEnsembleSelection:
        """Train the forest and build the game on the test split."""
        split = self.load_split()
        forest = build_model(
            "random_forest",
            split.task,
            random_state=self.random_state,
            **{**self.model_params, "n_estimators": self.n_members},
        ).fit(split.x_train, split.y_train)
        return RandomForestEnsembleSelection.from_forest(
            forest,
            split.x_test,
            split.y_test,
            task=split.task,
            metric=self.metric,
            empty_value=self.empty_value,
            normalize=self.normalize,
        )


@dataclass(frozen=True, kw_only=True)
class UncertaintyExplanationSetup(TabularSetup, name="uncertainty_explanation"):
    """An :class:`~shapiq_games.UncertaintyExplanation` of a random forest on a classification dataset.

    Attributes:
        x: The index of the explained point in the test split. Defaults to ``0``.
        uncertainty: ``"total"`` (default), ``"aleatoric"``, or ``"epistemic"``.
        imputer: ``"marginal"`` (default), ``"conditional"``, or ``"baseline"``.
        n_background: The number of background rows (from the training split).
        model_params: Hyperparameters of the forest.
        normalize: Whether to center the game. Defaults to ``True``.

    Examples:
        >>> UncertaintyExplanationSetup(dataset="breast_cancer").build().n_players
        30
    """

    x: int = 0
    uncertainty: Uncertainty = "total"
    imputer: Literal["marginal", "conditional", "baseline"] = "marginal"
    n_background: int = 100
    model_params: dict[str, Any] = field(default_factory=dict)
    normalize: bool = True

    def build(self) -> UncertaintyExplanation:
        """Train the forest and build the game."""
        split = self.load_split()
        if split.task != "classification":
            msg = f"UncertaintyExplanation needs a classification dataset, got '{self.dataset}'."
            raise ValueError(msg)
        forest = build_model(
            "random_forest", split.task, random_state=self.random_state, **self.model_params
        ).fit(split.x_train, split.y_train)
        rows = self.sample_rows(split.x_train.shape[0], self.n_background)
        return UncertaintyExplanation(
            forest,
            split.x_train[rows],
            resolve_x(self.x, split.x_test),
            uncertainty=self.uncertainty,
            imputer=self.imputer,
            random_state=self.random_state,
            normalize=self.normalize,
        )


@dataclass(frozen=True, kw_only=True)
class ClusterExplanationSetup(TabularSetup, name="cluster_explanation"):
    """A :class:`~shapiq_games.ClusterExplanation` on a seeded, standardized sample of a dataset.

    Attributes:
        n_samples: The number of rows to cluster (scores like the silhouette are quadratic in the
            number of rows). Defaults to ``1000``.
        method: ``"kmeans"`` (default) or ``"agglomerative"``.
        n_clusters: The number of clusters. Defaults to ``3``.
        score: ``"calinski_harabasz"`` (default) or ``"silhouette"``.
        empty_value: The value of the empty coalition. Defaults to ``0``.
        normalize: Whether to center the game. Defaults to ``True``.

    Examples:
        >>> ClusterExplanationSetup(dataset="group", n_samples=200).build().n_players
        4
    """

    n_samples: int = 1000
    method: Literal["kmeans", "agglomerative"] = "kmeans"
    n_clusters: int = 3
    score: Literal["calinski_harabasz", "silhouette"] = "calinski_harabasz"
    empty_value: float = 0.0
    normalize: bool = True

    def build(self) -> ClusterExplanation:
        """Sample and standardize the rows and build the game."""
        data = self.load().x
        data = StandardScaler().fit_transform(data[self.sample_rows(data.shape[0], self.n_samples)])
        return ClusterExplanation(
            data,
            method=self.method,
            n_clusters=self.n_clusters,
            score=self.score,
            empty_value=self.empty_value,
            random_state=self.random_state,
            normalize=self.normalize,
        )


@dataclass(frozen=True, kw_only=True)
class UnsupervisedDataSetup(TabularSetup, name="unsupervised_data"):
    """An :class:`~shapiq_games.UnsupervisedData` game on (a seeded sample of) a dataset.

    Attributes:
        n_samples: Use a seeded sample of this many rows (``None`` for all).
        n_bins: The number of equal-width bins per feature. Defaults to ``20``.

    Examples:
        >>> UnsupervisedDataSetup(dataset="xor").build().n_players
        4
    """

    n_samples: int | None = None
    n_bins: int = 20

    def build(self) -> UnsupervisedData:
        """Sample the rows and build the game."""
        data = self.load().x
        return UnsupervisedData(
            data[self.sample_rows(data.shape[0], self.n_samples)], n_bins=self.n_bins
        )
