"""Optimize hyperparameters for various models and datasets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna

# Model imports
from lightgbm import LGBMClassifier, LGBMRegressor
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

from shapiq_benchmark.setup import load_data_from_str

if TYPE_CHECKING:
    from shapiq_benchmark.bench_types import BenchmarkDataset

IndexArray = NDArray[np.integer]
DataArray = NDArray[np.number]

logger = logging.getLogger(__name__)

# Map datasets to their respective data types
DATASET_TASKS = {
    "adult_census": "classification",
    "california_housing": "regression",
}


def _subset(data: Any, indices: IndexArray) -> Any: # noqa: ANN401
    """Return a subset of data for the provided indices."""
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def get_hyperparameters(
    trial: optuna.Trial, model_name: str, task: str, random_state: int
) -> dict[str, Any]:
    """Return the hyperparameter search space based on the model.

    Args:
        trial: Optuna trial object for suggesting hyperparameters.
        model_name: Name of the model to optimize (e.g. "lightgbm").
        task: Type of machine learning task ("classification" or "regression").
        random_state: Random seed for reproducibility.
    """
    params: dict[str, Any] = {"random_state": random_state, "n_jobs": -1}

    if model_name == "lightgbm":
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "num_leaves": trial.suggest_int("num_leaves", 16, 255),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", -1, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
            }
        )
    elif model_name == "random_forest":
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_float("max_features", 0.3, 1.0),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            }
        )
    elif model_name == "xgboost":
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
            }
        )
        if task == "classification":
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
        else:
            params["objective"] = "reg:squarederror"
    else:
        msg = f"Unsupported model: {model_name}"
        raise ValueError(msg)

    return params


def get_model_instance(model_name: str, task: str, params: dict[str, Any]) -> Any: # noqa: ANN401
    """Instantiate the correct model class based on name and task.

    Args:
        model_name: Name of the model to instantiate (e.g. "lightgbm").
        task: Type of machine learning task ("classification" or "regression").
        params: Hyperparameters to initialize the model with.
    """
    if model_name == "lightgbm":
        return LGBMClassifier(**params) if task == "classification" else LGBMRegressor(**params)
    if model_name == "random_forest":
        return (
            RandomForestClassifier(**params)
            if task == "classification"
            else RandomForestRegressor(**params)
        )
    if model_name == "xgboost":
        return XGBClassifier(**params) if task == "classification" else XGBRegressor(**params)
    msg = f"Unsupported model: {model_name}"
    raise ValueError(msg)


def objective(
    trial: optuna.Trial,
    dataset: BenchmarkDataset,
    model_name: str,
    task: str,
    random_state: int,
    n_splits: int,
) -> float:
    """Run cross-validated optimization for a trial.

    Args:
        trial: Optuna trial object for suggesting hyperparameters.
        dataset: Benchmark dataset for training and validation.
        model_name: Name of the model to optimize (e.g. "lightgbm").
        task: Type of machine learning task ("classification" or "regression").
        random_state: Random seed for reproducibility.
        n_splits: Number of cross-validation folds.

    Returns:
        Average cross-validation score.
    """
    params = get_hyperparameters(trial, model_name, task, random_state)

    x_all = dataset.x_train
    y_all = dataset.y_train

    # Configure cross-validation and metric based on the task
    if task == "classification":
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        metric_func = accuracy_score
    else:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        metric_func = r2_score

    scores = []
    # Both StratifiedKFold and KFold accept (X, y) in their split method in scikit-learn
    for train_idx, valid_idx in kfold.split(x_all, y_all):
        model = get_model_instance(model_name, task, params)
        x_train = _subset(x_all, train_idx)
        y_train = _subset(y_all, train_idx)
        x_valid = _subset(x_all, valid_idx)
        y_valid = _subset(y_all, valid_idx)

        model.fit(x_train, y_train)
        preds = model.predict(x_valid)
        scores.append(metric_func(y_valid, preds))

    return sum(scores) / len(scores)


def save_results(
    output_path: Path,
    dataset_name: str,
    model_name: str,
    task: str,
    study: optuna.Study,
    n_trials: int,
    n_splits: int,
) -> None:
    """Persist study results to disk.

    Args:
        output_path: Path to save the results JSON file.
        dataset_name: Name of the dataset used in the study.
        model_name: Name of the model optimized in the study.
        task: Type of machine learning task ("classification" or "regression").
        study: Optuna study object containing the optimization results.
        n_trials: Number of optimization trials.
        n_splits: Number of cross-validation folds.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_name = "accuracy" if task == "classification" else "r2"

    payload = {
        "dataset": dataset_name,
        "model": model_name,
        "metric": metric_name,
        "cv_folds": n_splits,
        "n_trials": n_trials,
        f"best_{metric_name}": study.best_value,
        "best_params": study.best_params,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), "utf-8")


def main() -> None:
    """Run Optuna optimization dynamically based on arguments."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Optimize ML models using Optuna.")

    # Core Arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lightgbm", "random_forest", "xgboost"],
        help="Model to optimize",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["adult_census", "california_housing"],
        help="Dataset to evaluate on",
    )

    # Tuning Arguments
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")

    # Output path mapping
    parser.add_argument("--output", type=Path, default=None, help="Custom output JSON path")
    args = parser.parse_args()

    # Determine default path if not explicitly provided
    if args.output is None:
        args.output = Path(f"results/{args.model}_{args.dataset}.json")

    task = DATASET_TASKS[args.dataset]
    logger.info("Initializing %s optimization for %s (%s task).", args.model, args.dataset, task)

    dataset = load_data_from_str(args.dataset)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, dataset, args.model, task, args.random_state, args.folds),
        n_trials=args.trials,
    )

    metric_name = "accuracy" if task == "classification" else "R^2"
    logger.info("Best %s: %s", metric_name, study.best_value)
    logger.info("Best params:")
    for key, value in study.best_params.items():
        logger.info("  %s: %s", key, value)

    save_results(
        args.output,
        args.dataset,
        args.model,
        task,
        study,
        args.trials,
        args.folds,
    )
    logger.info("Saved results to: %s", args.output)


if __name__ == "__main__":
    main()
