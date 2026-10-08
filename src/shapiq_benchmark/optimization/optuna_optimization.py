"""Tune model hyperparameters with Optuna to produce the presets of :mod:`shapiq_benchmark.models`.

Example:
    python -m shapiq_benchmark.optimization.optuna_optimization \
        --model xgboost --dataset adult_census --trials 50 --output tuned.json

The best parameters are written to the output file. To make them available as
``preset="tuned"``, add them to ``TUNED_PRESETS`` in ``shapiq_benchmark/models.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold

from shapiq_benchmark.datasets import list_datasets, load_dataset
from shapiq_benchmark.models import build_model
from shapiq_games._optional import require

if TYPE_CHECKING:
    import optuna

    from shapiq_benchmark.datasets import DatasetSplit
    from shapiq_benchmark.models import ModelName

logger = logging.getLogger(__name__)

TUNABLE_MODELS: tuple[ModelName, ...] = ("lightgbm", "random_forest", "xgboost")


def get_hyperparameters(trial: optuna.Trial, model_name: ModelName) -> dict[str, Any]:
    """Return the hyperparameters of a trial (the search space of the model)."""
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "num_leaves": trial.suggest_int("num_leaves", 16, 255),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,  # LightGBM ignores subsample without it
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
        }
    msg = f"Unsupported model: {model_name}. Choose one of {TUNABLE_MODELS}."
    raise ValueError(msg)


def cross_validated_score(
    split: DatasetSplit,
    model_name: ModelName,
    params: dict[str, Any],
    *,
    n_splits: int,
    random_state: int,
) -> float:
    """Return the mean cross-validated accuracy (classification) or R² (regression)."""
    classification = split.task == "classification"
    folds_class = StratifiedKFold if classification else KFold
    folds = folds_class(n_splits=n_splits, shuffle=True, random_state=random_state)
    metric = accuracy_score if classification else r2_score
    scores = []
    for train, valid in folds.split(split.x_train, split.y_train):
        model = build_model(model_name, split.task, random_state=random_state, **params)
        model.fit(split.x_train[train], split.y_train[train])
        scores.append(metric(split.y_train[valid], model.predict(split.x_train[valid])))
    return float(np.mean(scores))


def main() -> None:
    """Tune a model on a dataset and write the best parameters."""
    optuna = require("optuna", extra="benchmark")
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Tune model hyperparameters with Optuna.")
    parser.add_argument("--model", required=True, choices=TUNABLE_MODELS)
    parser.add_argument("--dataset", required=True, choices=list_datasets())
    parser.add_argument("--trials", type=int, default=50, help="number of Optuna trials")
    parser.add_argument("--folds", type=int, default=5, help="number of cross-validation folds")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="output JSON path")
    args = parser.parse_args()
    output = args.output or Path(f"tuned_{args.model}_{args.dataset}.json")

    split = load_dataset(args.dataset).split(random_state=args.random_state)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.random_state)
    )
    study.optimize(
        lambda trial: cross_validated_score(
            split,
            args.model,
            get_hyperparameters(trial, args.model),
            n_splits=args.folds,
            random_state=args.random_state,
        ),
        n_trials=args.trials,
    )
    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "metric": "accuracy" if split.task == "classification" else "r2",
        "cv_folds": args.folds,
        "n_trials": args.trials,
        "random_state": args.random_state,
        "best_score": study.best_value,
        "best_params": study.best_params,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Best score %.4f; parameters written to %s", study.best_value, output)
    logger.info(
        "Add to TUNED_PRESETS in shapiq_benchmark/models.py: (%r, %r): %r",
        args.model,
        args.dataset,
        study.best_params,
    )


if __name__ == "__main__":
    main()
