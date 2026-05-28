"""Optimize XGBoost hyperparameters for Adult Census classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from shapiq_benchmark.setup import load_data_from_str


def _subset(data: Any, indices: Any) -> Any:
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def objective(
    trial: optuna.Trial,
    dataset: Any,
    random_state: int,
    n_splits: int,
) -> float:
    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
        "random_state": random_state,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": -1,
    }

    x_all = dataset.x_train
    y_all = dataset.y_train
    kfold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    scores = []
    for train_idx, valid_idx in kfold.split(x_all, y_all):
        model = XGBClassifier(**params)
        x_train = _subset(x_all, train_idx)
        y_train = _subset(y_all, train_idx)
        x_valid = _subset(x_all, valid_idx)
        y_valid = _subset(y_all, valid_idx)

        model.fit(x_train, y_train)
        preds = model.predict(x_valid)
        scores.append(accuracy_score(y_valid, preds))

    return sum(scores) / len(scores)


def save_results(
    output_path: Path,
    dataset_name: str,
    model_name: str,
    study: optuna.Study,
    n_trials: int,
    n_splits: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "model": model_name,
        "metric": "accuracy",
        "cv_folds": n_splits,
        "n_trials": n_trials,
        "best_accuracy": study.best_value,
        "best_params": study.best_params,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), "utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/xgboost_adult_census.json"),
    )
    args = parser.parse_args()

    dataset = load_data_from_str("adult_census")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, dataset, args.random_state, args.folds),
        n_trials=args.trials,
    )

    print("Best accuracy:", study.best_value)
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    save_results(
        args.output,
        "adult_census",
        "xgboost",
        study,
        args.trials,
        args.folds,
    )
    print("Saved results to:", args.output)


if __name__ == "__main__":
    main()
