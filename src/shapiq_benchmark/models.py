"""Seeded model registry of the benchmark setups (:mod:`shapiq_benchmark.setups`).

Every model is built with an explicit ``random_state`` and single-threaded where the library
allows it, so that a game built by a setup is reproducible. Optional backends (``xgboost``,
``lightgbm``, ``catboost``, ``tabpfn``) are imported lazily.

``"tabpfn"`` builds TabPFN v2, whose checkpoints download without a license, with the version's
default settings. Choose another version with ``version`` (e.g.
``build_model("tabpfn", "regression", version="v3")``; the versions after v2 need a Prior Labs
license token to download) or a checkpoint of your own with ``model_path``.

Examples:
    >>> from shapiq_benchmark.datasets import load_dataset
    >>> from shapiq_benchmark.models import fit_model
    >>> split = load_dataset("breast_cancer").split(random_state=42)
    >>> model = fit_model("random_forest", split, random_state=42)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, get_args

from shapiq_games._optional import require
from shapiq_games._tabpfn import DEFAULT_TABPFN_VERSION, build_tabpfn

if TYPE_CHECKING:
    from shapiq_games.typing import Task

    from .datasets import DatasetSplit

__all__ = ["MODEL_NAMES", "TUNED_PRESETS", "ModelName", "Preset", "build_model", "fit_model"]

type ModelName = Literal[
    "catboost",
    "decision_tree",
    "gaussian_process",
    "knn",
    "lightgbm",
    "linear",
    "mlp",
    "random_forest",
    "svm",
    "tabpfn",
    "threshold_nn",
    "weighted_knn",
    "xgboost",
]
"""The names of the models :func:`build_model` builds."""

type Preset = Literal["tuned"]
"""A hyperparameter preset: ``"tuned"`` takes the hyperparameters of :data:`TUNED_PRESETS`."""

MODEL_NAMES: tuple[ModelName, ...] = get_args(ModelName.__value__)
"""All model names understood by :func:`build_model`."""

# The LightGBM presets were tuned with a subsample that LightGBM ignores without
# subsample_freq; it is left out, which builds the same models.
TUNED_PRESETS: dict[tuple[ModelName, str], dict[str, Any]] = {
    ("lightgbm", "adult_census"): {
        "n_estimators": 944,
        "num_leaves": 47,
        "learning_rate": 0.030338722763452043,
        "max_depth": 4,
        "min_child_samples": 9,
        "colsample_bytree": 0.7155127263223596,
        "reg_alpha": 0.00013816182430361382,
        "reg_lambda": 0.0004958490081190707,
    },
    ("lightgbm", "california_housing"): {
        "n_estimators": 876,
        "num_leaves": 213,
        "learning_rate": 0.02568211645857779,
        "max_depth": 12,
        "min_child_samples": 7,
        "colsample_bytree": 0.7907540024230353,
        "reg_alpha": 6.829010029601757e-06,
        "reg_lambda": 5.2528698290206714e-05,
    },
    ("random_forest", "adult_census"): {
        "n_estimators": 891,
        "max_depth": 18,
        "min_samples_split": 8,
        "min_samples_leaf": 2,
        "max_features": 0.34835985964965094,
        "bootstrap": True,
    },
    ("random_forest", "california_housing"): {
        "n_estimators": 942,
        "max_depth": 23,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 0.4742092568563367,
        "bootstrap": False,
    },
    ("xgboost", "adult_census"): {
        "n_estimators": 123,
        "max_depth": 5,
        "learning_rate": 0.18043561943195366,
        "subsample": 0.9085787661658077,
        "colsample_bytree": 0.6887648967210234,
        "min_child_weight": 2.7918525297388306,
        "reg_alpha": 0.0003945389391237688,
        "reg_lambda": 0.00010284791765246808,
    },
    ("xgboost", "california_housing"): {
        "n_estimators": 702,
        "max_depth": 10,
        "learning_rate": 0.0206655363533331,
        "subsample": 0.894841518751018,
        "colsample_bytree": 0.7209125840196445,
        "min_child_weight": 6.662699149338694,
        "reg_alpha": 0.0006797261661911757,
        "reg_lambda": 0.0008137614330078947,
    },
}
"""Hyperparameters tuned with 50 Optuna trials and 5-fold cross-validation, keyed by
``(model, dataset)``. Produced by ``shapiq_benchmark/optimization/optuna_optimization.py``."""


def _sklearn_model(
    name: ModelName, task: Task, random_state: int | None, params: dict[str, Any]
) -> Any:  # noqa: ANN401
    """Build one of the scikit-learn models."""
    classification = task == "classification"
    if name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        cls = DecisionTreeClassifier if classification else DecisionTreeRegressor
        return cls(**{"random_state": random_state, **params})
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cls = RandomForestClassifier if classification else RandomForestRegressor
        return cls(**{"n_estimators": 10, "n_jobs": 1, "random_state": random_state, **params})
    if name == "linear":
        from sklearn.linear_model import LinearRegression, LogisticRegression

        if classification:
            return LogisticRegression(**{"max_iter": 1000, "random_state": random_state, **params})
        return LinearRegression(**params)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        cls = MLPClassifier if classification else MLPRegressor
        return cls(**{"max_iter": 500, "random_state": random_state, **params})
    if name == "svm":
        from sklearn.svm import SVC, SVR

        if classification:
            # probabilities are needed by the games that explain class probabilities
            return SVC(
                **{"kernel": "rbf", "probability": True, "random_state": random_state, **params}
            )
        return SVR(**{"kernel": "rbf", **params})
    if name == "gaussian_process":
        if classification:
            msg = "The 'gaussian_process' model is only available for regression."
            raise ValueError(msg)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF

        return GaussianProcessRegressor(**{"kernel": RBF(), "random_state": random_state, **params})
    if name in {"knn", "weighted_knn"}:
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

        cls = KNeighborsClassifier if classification else KNeighborsRegressor
        weights = "distance" if name == "weighted_knn" else "uniform"
        return cls(**{"weights": weights, **params})
    if name == "threshold_nn":
        if not classification:
            msg = "The 'threshold_nn' model is only available for classification."
            raise ValueError(msg)
        from sklearn.neighbors import RadiusNeighborsClassifier

        return RadiusNeighborsClassifier(**params)
    msg = f"Unknown model '{name}'."
    raise ValueError(msg)


def build_model(
    name: ModelName,
    task: Task,
    *,
    random_state: int | None = 42,
    preset: Preset | None = None,
    dataset: str | None = None,
    **params: Any,
) -> Any:  # noqa: ANN401
    """Build an unfitted, seeded model.

    Args:
        name: The model name, one of :data:`MODEL_NAMES`.
        task: ``"classification"`` or ``"regression"``.
        random_state: The seed passed to the model. Defaults to ``42``.
        preset: ``"tuned"`` to use the hyperparameters in :data:`TUNED_PRESETS` (requires
            ``dataset``), or ``None`` for the defaults of the registry.
        dataset: The dataset name, needed for ``preset="tuned"``.
        **params: Hyperparameters overriding the defaults and the preset; for ``"tabpfn"``, also
            the TabPFN ``version`` (default ``"v2"``).

    Returns:
        The unfitted model.

    Raises:
        ValueError: If the model, task, or preset is unknown.
    """
    if task not in {"classification", "regression"}:
        msg = f"task must be 'classification' or 'regression', got {task!r}."
        raise ValueError(msg)
    if preset is not None:
        if preset != "tuned":
            msg = f"Unknown preset {preset!r}; the only preset is 'tuned'."
            raise ValueError(msg)
        tuned = TUNED_PRESETS.get((name, dataset)) if dataset is not None else None
        if tuned is None:
            available = ", ".join(f"{m}/{d}" for m, d in sorted(TUNED_PRESETS))
            msg = f"No tuned preset for model '{name}' on '{dataset}'. Available: {available}."
            raise ValueError(msg)
        params = {**tuned, **params}

    classification = task == "classification"
    if name == "xgboost":
        xgboost = require("xgboost", purpose="the 'xgboost' model", extra="benchmark")
        cls = xgboost.XGBClassifier if classification else xgboost.XGBRegressor
        return cls(**{"n_jobs": 1, "random_state": random_state, **params})
    if name == "lightgbm":
        lightgbm = require("lightgbm", purpose="the 'lightgbm' model", extra="benchmark")
        cls = lightgbm.LGBMClassifier if classification else lightgbm.LGBMRegressor
        return cls(**{"n_jobs": 1, "verbose": -1, "random_state": random_state, **params})
    if name == "catboost":
        catboost = require("catboost", purpose="the 'catboost' model", extra="benchmark")
        cls = catboost.CatBoostClassifier if classification else catboost.CatBoostRegressor
        return cls(**{"thread_count": 1, "verbose": 0, "random_seed": random_state, **params})
    if name == "tabpfn":
        version = params.pop("version", DEFAULT_TABPFN_VERSION)
        return build_tabpfn(
            task, version, **{"device": "cpu", "random_state": random_state, **params}
        )
    return _sklearn_model(name, task, random_state, params)


def fit_model(
    name: ModelName,
    split: DatasetSplit,
    *,
    random_state: int | None = 42,
    preset: Preset | None = None,
    **params: Any,
) -> Any:  # noqa: ANN401
    """Build a seeded model and fit it on the training part of a dataset split.

    Args:
        name: The model name, one of :data:`MODEL_NAMES`.
        split: The dataset split to train on.
        random_state: The seed passed to the model. Defaults to ``42``.
        preset: ``"tuned"`` to use the tuned hyperparameters for this dataset.
        **params: Hyperparameters overriding the defaults and the preset.

    Returns:
        The fitted model.
    """
    model = build_model(
        name,
        split.task,
        random_state=random_state,
        preset=preset,
        dataset=split.dataset.name,
        **params,
    )
    model.fit(split.x_train, split.y_train)
    return model
