"""Setups of the machine learning games: explanation, selection, valuation, and data games."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shapiq_benchmark.models import ModelName, build_model
from shapiq_games import (
    ClusterExplanation,
    DatasetValuation,
    DataValuation,
    EnsembleSelection,
    FeatureSelection,
    RandomForestEnsembleSelection,
    TabularGlobalExplanation,
    TabularLocalExplanation,
    TabularUncertaintyExplanation,
    UnsupervisedData,
)
from shapiq_games._base import is_classifier, resolve_class_index, resolve_x
from shapiq_games._tabpfn import require_inf_passthrough
from shapiq_games.typing import (  # noqa: TC001  (resolved by the field checks)
    ClusterMethod,
    ClusterScore,
    ImputerName,
    LossName,
    MetricName,
    PlayerSizes,
    Uncertainty,
)

from ._base import ModelSetup, TabularSetup

if TYPE_CHECKING:
    from shapiq.imputer.base import Imputer
    from shapiq.typing import IntVector
    from shapiq_games.typing import Task

__all__ = [
    "DEFAULT_MEMBER_POOL",
    "MISSING_VALUE_MODELS",
    "ClusterExplanationSetup",
    "DataValuationSetup",
    "DatasetValuationSetup",
    "EnsembleSelectionSetup",
    "FeatureSelectionSetup",
    "RandomForestEnsembleSelectionSetup",
    "TabularGlobalExplanationSetup",
    "TabularLocalExplanationSetup",
    "TabularUncertaintyExplanationSetup",
    "UnsupervisedDataSetup",
]

MISSING_VALUE_MODELS: tuple[ModelName, ...] = (
    "catboost",
    "decision_tree",
    "lightgbm",
    "random_forest",
    "tabpfn",
    "xgboost",
)
"""The registry models that read missing values natively (for ``baseline="missing"``)."""

DEFAULT_MEMBER_POOL: tuple[ModelName, ...] = (
    "linear",
    "decision_tree",
    "random_forest",
    "svm",
    "knn",
    "xgboost",
)
"""The model names :class:`EnsembleSelectionSetup` draws its members from by default."""


@dataclass(frozen=True, kw_only=True)
class TabularLocalExplanationSetup(ModelSetup, name="tabular_local_explanation"):
    """A :class:`~shapiq_games.TabularLocalExplanation` of a model trained on a dataset.

    The model is trained on the training split (or a seeded subset of ``n_train`` rows, e.g. for
    TabPFN's context limit). The background data is a seeded random subset of ``n_background``
    training rows; the explained point is taken from the test split.

    - With ``imputer="tabpfn"`` the model must be ``"tabpfn"``, and the background rows are
      TabPFN's context (remove-and-recontextualize).
    - With ``imputer="baseline"``, absent features take the background mean (mode for categorical
      features), or with ``baseline="missing"`` a missing value the model reads natively: NaN for
      the tree models, and ``+inf`` for ``"tabpfn"``, which is then built with
      ``inference_config={"PASSTHROUGH_INF": True}`` (``tabpfn>=8.1``).
    - ``"tabpfn"`` is TabPFN v2 unless ``model_params`` chooses a ``version``, e.g.
      ``{"version": "v3"}`` as in the benchmarking paper (the versions after v2 need a Prior Labs
      license token to download).

    Attributes:
        model: The model name. Defaults to ``"random_forest"``.
        x: The index of the explained point in the test split. Defaults to ``0``.
        imputer: ``"marginal"`` (default), ``"conditional"``, ``"baseline"``, or ``"tabpfn"``.
        baseline: For ``imputer="baseline"``: ``"mean"`` (default) or ``"missing"`` (see above).
        n_background: The number of background (or context) rows. Defaults to ``100``.
        n_train: Train the model on a seeded subset of this many training rows (``None`` for all).
        class_index: The explained class for classifiers (``None`` means class ``1``).
        normalize: Whether to center the game. Defaults to ``True``.

    Examples:
        >>> setup = TabularLocalExplanationSetup(dataset="breast_cancer", model="decision_tree")
        >>> setup.build().n_players
        30
        >>> # TabPFN v3 with absent features masked as +inf, trained on 1,040 rows (on a CPU,
        >>> # tabpfn takes more than 1,000 only with ignore_pretraining_limits):
        >>> setup = TabularLocalExplanationSetup(
        ...     dataset="adult_census",
        ...     model="tabpfn",
        ...     model_params={"version": "v3", "ignore_pretraining_limits": True},
        ...     imputer="baseline",
        ...     baseline="missing",
        ...     n_train=1040,
        ... )
    """

    x: int = 0
    imputer: ImputerName | Literal["tabpfn"] = "marginal"
    baseline: Literal["mean", "missing"] = "mean"
    n_background: int = 100
    n_train: int | None = None
    class_index: int | None = None
    normalize: bool = True

    def __post_init__(self) -> None:
        """Check that the TabPFN imputer gets TabPFN and missing values a model that reads them."""
        super().__post_init__()
        if self.imputer == "tabpfn" and self.model != "tabpfn":
            msg = f"imputer='tabpfn' needs model='tabpfn', got model={self.model!r}."
            raise ValueError(msg)
        if self.baseline != "mean" and self.imputer != "baseline":
            msg = f"baseline applies to imputer='baseline', got imputer={self.imputer!r}."
            raise ValueError(msg)
        if self.baseline == "missing" and self.model not in MISSING_VALUE_MODELS:
            msg = (
                f"baseline='missing' needs a model that reads missing values, one of "
                f"{MISSING_VALUE_MODELS}; got model={self.model!r}."
            )
            raise ValueError(msg)

    def build(self) -> TabularLocalExplanation:
        """Train the model, draw the background rows, and build the game."""
        split = self.load_split()
        params: dict[str, Any] = dict(self.model_params)
        baseline: float | None = None
        if self.baseline == "missing":
            baseline = np.nan
            if self.model == "tabpfn":  # TabPFN reads +inf as missing
                require_inf_passthrough()
                config = dict(params.get("inference_config", {}))
                params["inference_config"] = {**config, "PASSTHROUGH_INF": True}
                baseline = np.inf
        train = self.sample_rows(split.x_train.shape[0], self.n_train)
        model = build_model(
            self.model,
            split.task,
            random_state=self.random_state,
            preset=self.preset,
            dataset=self.dataset,
            **params,
        ).fit(split.x_train[train], split.y_train[train])
        rows = self.sample_rows(split.x_train.shape[0], self.n_background)
        background = split.x_train[rows]
        point = resolve_x(self.x, split.x_test)
        imputer: ImputerName | Imputer
        if self.imputer == "tabpfn":
            imputer = _tabpfn_imputer(
                model, background, split.y_train[rows], split.x_test, point, self.class_index
            )
        else:
            imputer = self.imputer
        return TabularLocalExplanation(
            model,
            background,
            point,
            imputer=imputer,
            class_index=self.class_index,
            sample_size=self.n_background,
            baseline=baseline,
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
class TabularGlobalExplanationSetup(ModelSetup, name="tabular_global_explanation"):
    """A :class:`~shapiq_games.TabularGlobalExplanation` of a model trained on a dataset.

    The evaluation rows are drawn from the test split.

    Attributes:
        model: The model name. Defaults to ``"random_forest"``.
        class_index: The explained class for classifiers (``None`` means class ``1``).
        loss: ``"mse"`` (default) or ``"mae"``.
        n_samples: The number of evaluation rows. Defaults to ``100``.
        normalize: Whether to center the game. Defaults to ``True``.

    Examples:
        >>> TabularGlobalExplanationSetup(dataset="xor", model="decision_tree").build().n_players
        4
    """

    class_index: int | None = None
    loss: LossName = "mse"
    n_samples: int = 100
    normalize: bool = True

    def build(self) -> TabularGlobalExplanation:
        """Train the model and build the game on the test split."""
        split = self.load_split()
        return TabularGlobalExplanation(
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

    model: ModelName = "decision_tree"
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
            random_state=self.random_state,
            normalize=self.normalize,
        )


def _stratified_rows(y: np.ndarray, n: int | None, task: Task, random_state: int) -> IntVector:
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

    model: ModelName = "decision_tree"
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
            random_state=self.random_state,
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

    model: ModelName = "decision_tree"
    n_players: int = 10
    player_sizes: PlayerSizes = "uniform"
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

    members: tuple[ModelName, ...] | None = None
    n_members: int = 10
    metric: MetricName | None = None
    empty_value: float = 0.0
    normalize: bool = True

    def build(self) -> EnsembleSelection:
        """Train the members and build the game on the test split."""
        split = self.load_split()
        members = self.members
        if members is None:
            rng = np.random.default_rng(self.random_state)
            draws = rng.choice(len(DEFAULT_MEMBER_POOL), self.n_members)
            members = tuple(DEFAULT_MEMBER_POOL[i] for i in draws)
        fitted = []
        for i, name in enumerate(members):
            params: dict[str, Any] = {"n_neighbors": 3} if name == "knn" else {}
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
        params: dict[str, Any] = {**self.model_params, "n_estimators": self.n_members}
        forest = build_model(
            "random_forest", split.task, random_state=self.random_state, **params
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
class TabularUncertaintyExplanationSetup(TabularSetup, name="tabular_uncertainty_explanation"):
    """An :class:`~shapiq_games.TabularUncertaintyExplanation` of a random forest on a classification dataset.

    Attributes:
        x: The index of the explained point in the test split. Defaults to ``0``.
        uncertainty: ``"total"`` (default), ``"aleatoric"``, or ``"epistemic"``.
        imputer: ``"marginal"`` (default), ``"conditional"``, or ``"baseline"``.
        n_background: The number of background rows (from the training split).
        model_params: Hyperparameters of the forest.
        normalize: Whether to center the game. Defaults to ``True``.

    Examples:
        >>> TabularUncertaintyExplanationSetup(dataset="breast_cancer").build().n_players
        30
    """

    tasks = ("classification",)

    x: int = 0
    uncertainty: Uncertainty = "total"
    imputer: ImputerName = "marginal"
    n_background: int = 100
    model_params: dict[str, Any] = field(default_factory=dict)
    normalize: bool = True

    def build(self) -> TabularUncertaintyExplanation:
        """Train the forest and build the game."""
        split = self.load_split()
        forest = build_model(
            "random_forest", split.task, random_state=self.random_state, **self.model_params
        ).fit(split.x_train, split.y_train)
        rows = self.sample_rows(split.x_train.shape[0], self.n_background)
        return TabularUncertaintyExplanation(
            forest,
            split.x_train[rows],
            resolve_x(self.x, split.x_test),
            uncertainty=self.uncertainty,
            imputer=self.imputer,
            sample_size=self.n_background,
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
    method: ClusterMethod = "kmeans"
    n_clusters: int = 3
    score: ClusterScore = "calinski_harabasz"
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
