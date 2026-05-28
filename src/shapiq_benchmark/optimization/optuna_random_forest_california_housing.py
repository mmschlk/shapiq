"""Optimize RandomForest hyperparameters for California Housing (R^2)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import optuna
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from shapiq_benchmark.bench_types import BenchmarkDataset
from shapiq_benchmark.setup import load_data_from_str

IndexArray = NDArray[np.integer]
DataArray = NDArray[np.number]

logger = logging.getLogger(__name__)


def _subset(data: DataArray, indices: IndexArray) -> DataArray:
    """Return a subset of data for the provided indices."""
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def objective(
    trial: optuna.Trial,
    dataset: BenchmarkDataset,
    random_state: int,
    n_splits: int,
) -> float:
    """Run cross-validated optimization for a trial."""
    params: dict[str, object] = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": random_state,
        "n_jobs": -1,
    }

    x_all = dataset.x_train
    y_all = dataset.y_train
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = []
    for train_idx, valid_idx in kfold.split(x_all):
        model = RandomForestRegressor(**params)
        x_train = _subset(x_all, train_idx)
        y_train = _subset(y_all, train_idx)
        x_valid = _subset(x_all, valid_idx)
        y_valid = _subset(y_all, valid_idx)

        model.fit(x_train, y_train)
        preds = model.predict(x_valid)
        scores.append(r2_score(y_valid, preds))

    return sum(scores) / len(scores)


def save_results(
    output_path: Path,
    dataset_name: str,
    model_name: str,
    study: optuna.Study,
    n_trials: int,
    n_splits: int,
) -> None:
    """Persist study results to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "model": model_name,
        "metric": "r2",
        "cv_folds": n_splits,
        "n_trials": n_trials,
        "best_r2": study.best_value,
        "best_params": study.best_params,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), "utf-8")


def main() -> None:
    """Run Optuna optimization for RandomForest on California Housing."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/random_forest_california_housing.json"),
    )
    args = parser.parse_args()

    dataset = load_data_from_str("california_housing")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, dataset, args.random_state, args.folds),
        n_trials=args.trials,
    )

    logger.info("Best R^2: %s", study.best_value)
    logger.info("Best params:")
    for key, value in study.best_params.items():
        logger.info("  %s: %s", key, value)
    save_results(
        args.output,
        "california_housing",
        "random_forest",
        study,
        args.trials,
        args.folds,
    )
    logger.info("Saved results to: %s", args.output)


if __name__ == "__main__":
    main()
