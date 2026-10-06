"""Seeded model registry used to configure the games of :mod:`shapiq_games` from strings.

Every model is built with an explicit ``random_state`` and single-threaded where the library
allows it, so that a configured game is reproducible. Optional backends (``xgboost``,
``lightgbm``, ``catboost``, ``tabpfn``) are imported lazily.

Examples:
    >>> from shapiq_games.datasets import load_dataset
    >>> from shapiq_games.models import fit_model
    >>> split = load_dataset("california_housing").split(random_state=42)
    >>> model = fit_model("random_forest", split, random_state=42)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ._optional import require

if TYPE_CHECKING:
    from .datasets import DatasetSplit

__all__ = ["MODEL_NAMES", "TUNED_PRESETS", "build_model", "fit_model"]

type Task = Literal["classification", "regression"]

MODEL_NAMES: tuple[str, ...] = (
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
)
"""All model names understood by :func:`build_model`."""

TUNED_PRESETS: dict[tuple[str, str], dict[str, Any]] = {
    ("lightgbm", "adult_census"): {
        "n_estimators": 944,
        "num_leaves": 47,
        "learning_rate": 0.030338722763452043,
        "max_depth": 4,
        "min_child_samples": 9,
        "subsample": 0.9396824635118503,
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
        "subsample": 0.8486821062041688,
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


def _sklearn_model(name: str, task: Task, random_state: int | None, params: dict[str, Any]) -> Any:  # noqa: ANN401
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
    name: str,
    task: Task,
    *,
    random_state: int | None = 42,
    preset: str | None = None,
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
        **params: Hyperparameters overriding the defaults and the preset.

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
        if (name, dataset) not in TUNED_PRESETS:
            available = ", ".join(f"{m}/{d}" for m, d in sorted(TUNED_PRESETS))
            msg = f"No tuned preset for model '{name}' on '{dataset}'. Available: {available}."
            raise ValueError(msg)
        params = {**TUNED_PRESETS[(name, dataset)], **params}

    classification = task == "classification"
    if name == "xgboost":
        xgboost = require("xgboost", purpose="the 'xgboost' model")
        cls = xgboost.XGBClassifier if classification else xgboost.XGBRegressor
        return cls(**{"n_jobs": 1, "random_state": random_state, **params})
    if name == "lightgbm":
        lightgbm = require("lightgbm", purpose="the 'lightgbm' model")
        cls = lightgbm.LGBMClassifier if classification else lightgbm.LGBMRegressor
        return cls(**{"n_jobs": 1, "verbose": -1, "random_state": random_state, **params})
    if name == "catboost":
        catboost = require("catboost", purpose="the 'catboost' model")
        cls = catboost.CatBoostClassifier if classification else catboost.CatBoostRegressor
        return cls(**{"thread_count": 1, "verbose": 0, "random_seed": random_state, **params})
    if name == "tabpfn":
        tabpfn = require("tabpfn", purpose="the 'tabpfn' model")
        cls = tabpfn.TabPFNClassifier if classification else tabpfn.TabPFNRegressor
        return cls(**{"device": "cpu", "random_state": random_state, **params})
    return _sklearn_model(name, task, random_state, params)


def fit_model(
    name: str,
    split: DatasetSplit,
    *,
    random_state: int | None = 42,
    preset: str | None = None,
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
