"""Shared, frozen prediction models for the quality-focused benchmark.

One profile/dataset/feature-set/seed produces one model, reused across game
constructions. Dataset loaders own their preprocessing; additional missing-value
repair uses fitting rows only. Held-out rows never tune a model.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import Any

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import balanced_accuracy_score, log_loss, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from shapiq_benchmark.datasets import DATASETS, dataset_details, load_raw_dataset
from shapiq_benchmark.setup import _resolve_model_builder

MODEL_PROFILES: dict = {
    "random_forest": {
        "label": "Random forest",
        "parameters": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 5, "n_jobs": 1},
    },
    "xgboost": {
        "label": "XGBoost",
        "parameters": {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.05,
            "early_stopping_rounds": 20,
            "tree_method": "hist",
            "n_jobs": 1,
        },
    },
}
TRAINING_PROFILE: dict = {
    "id": "quality-v1",
    "fit_rows_max": 5000,
    "validation_rows_max": 1000,
    "test_rows_max": 1000,
    "test_fraction": 0.2,
    "validation_fraction_of_remainder": 0.2,
    "missing_values": "fitting-row medians",
    "classification_output": "probability of the second sorted original class",
}


@dataclass
class PreparedModel:
    """A fitted predictor, disjoint data partitions, and public reproduction metadata."""

    model: Any
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    metadata: dict

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Use a fixed scalar output, including for multiclass classification."""
        if self.metadata["task"] == "classification":
            return self.model.predict_proba(x)[:, 1]
        return self.model.predict(x)


def _split(
    rows: np.ndarray, y: np.ndarray, size: float, seed: int, *, classification: bool
) -> tuple[np.ndarray, np.ndarray, str]:
    """Stratify when feasible; explicitly record a fallback for rare-class splits."""
    labels = y[rows] if classification else None
    try:
        left, right = train_test_split(rows, test_size=size, random_state=seed, stratify=labels)
    except ValueError:
        if not classification:
            raise
        left, right = train_test_split(rows, test_size=size, random_state=seed)
        return left, right, "shuffled: stratification infeasible"
    else:
        return left, right, "stratified" if classification else "shuffled"


def _bounded(
    rows: np.ndarray, y: np.ndarray, limit: int, seed: int, *, classification: bool
) -> tuple[np.ndarray, str]:
    if len(rows) <= limit:
        return rows, "all rows"
    _, selected, rule = _split(rows, y, limit, seed, classification=classification)
    return selected, rule


def _scores(
    model: object, x: np.ndarray, y: np.ndarray, classes: np.ndarray, *, classification: bool
) -> dict:
    model = cast("Any", model)
    if classification:
        return {
            "log_loss": float(log_loss(y, model.predict_proba(x), labels=classes)),
            "balanced_accuracy": float(balanced_accuracy_score(y, model.predict(x))),
        }
    return {
        "mse": float(mean_squared_error(y, model.predict(x))),
        "r2": float(r2_score(y, model.predict(x))),
    }


def _tree_diagnostics(model: object, profile: str) -> dict:
    """Describe realized structure, without claiming depth guarantees interaction strength."""
    model = cast("Any", model)
    features, depths, path_widths = set(), [], []
    trees = (
        model.estimators_
        if profile == "random_forest"
        else model.get_booster().get_dump(dump_format="json")
    )
    if profile == "xgboost":
        # The early-stopped predictor ignores trees fitted after its best round.
        classes_per_round = getattr(model, "n_classes_", 2)
        trees_per_round = classes_per_round if classes_per_round > 2 else 1
        trees = trees[: (model.best_iteration + 1) * trees_per_round]
    for estimator in trees:
        tree = estimator.tree_ if profile == "random_forest" else json.loads(estimator)
        stack: list[tuple[Any, int, frozenset]] = [
            (0 if profile == "random_forest" else tree, 0, frozenset())
        ]
        depth_max = width_max = 0
        while stack:
            node, depth, path = stack.pop()
            if profile == "random_forest":
                feature = int(tree.feature[node])
                children = (
                    [int(tree.children_left[node]), int(tree.children_right[node])]
                    if feature >= 0
                    else []
                )
            else:
                feature = int(node["split"].removeprefix("f")) if "split" in node else -1
                children = node.get("children", [])
            if children:
                features.add(feature)
                path = path | {feature}
                stack.extend((child, depth + 1, path) for child in children)
            else:
                depth_max = max(depth_max, depth)
                width_max = max(width_max, len(path))
        depths.append(depth_max)
        path_widths.append(width_max)
    return {
        "tree_count": len(trees),
        "tree_depths": depths,
        "features_used": sorted(features),
        "max_distinct_features_on_path": max(path_widths),
        "note": "Model structure only; game-level interaction diagnostics are separate.",
    }


def _fit(
    dataset: str,
    n_players: int,
    seed: int,
    profile: str,
    x: np.ndarray,
    original_y: np.ndarray,
    names: list,
    identity: dict,
) -> PreparedModel:
    classification = DATASETS[dataset]["task"] == "classification"
    classes, encoded = np.unique(original_y, return_inverse=True)
    y = encoded if classification else original_y
    features = np.sort(np.random.default_rng(seed).choice(x.shape[1], n_players, replace=False))
    fit, test, test_rule = _split(
        np.arange(len(x)), y, TRAINING_PROFILE["test_fraction"], seed, classification=classification
    )
    fit, validation, validation_rule = _split(
        fit,
        y,
        TRAINING_PROFILE["validation_fraction_of_remainder"],
        seed,
        classification=classification,
    )
    fit, fit_cap = _bounded(
        fit, y, TRAINING_PROFILE["fit_rows_max"], seed, classification=classification
    )
    validation, validation_cap = _bounded(
        validation, y, TRAINING_PROFILE["validation_rows_max"], seed, classification=classification
    )
    test, test_cap = _bounded(
        test, y, TRAINING_PROFILE["test_rows_max"], seed, classification=classification
    )
    if classification and len(np.unique(y[fit])) != len(classes):
        message = "Fitting partition omits a class; this seeded instance is unsupported."
        raise ValueError(message)
    selected = np.asarray(x[:, features], dtype=float).copy()
    if np.isinf(selected).any():
        message = "Infinite inputs cannot be repaired by the missing-value protocol."
        raise ValueError(message)
    medians = np.nanmedian(selected[fit], axis=0)
    if not np.isfinite(medians).all():
        message = "A selected feature is missing on every fitting row."
        raise ValueError(message)
    rows, columns = np.where(np.isnan(selected))
    selected[rows, columns] = medians[columns]
    parameters: dict = {**MODEL_PROFILES[profile]["parameters"], "random_state": seed}
    if profile == "xgboost":
        parameters["objective"] = (
            ("binary:logistic" if len(classes) == 2 else "multi:softprob")
            if classification
            else "reg:squarederror"
        )
        parameters["eval_metric"] = (
            "logloss"
            if classification and len(classes) == 2
            else "mlogloss"
            if classification
            else "rmse"
        )
    model = cast("Any", _resolve_model_builder(profile, str(DATASETS[dataset]["task"])))
    model = model(**parameters)
    fit_options = (
        {"eval_set": [(selected[validation], y[validation])], "verbose": False}
        if profile == "xgboost"
        else {}
    )
    model.fit(selected[fit], y[fit], **fit_options)
    dummy = DummyClassifier(strategy="prior") if classification else DummyRegressor(strategy="mean")
    dummy.fit(selected[fit], y[fit])
    metrics = {}
    for label, indices in (("validation", validation), ("test", test)):
        metrics[label] = {
            "model": _scores(
                model,
                selected[indices],
                y[indices],
                np.arange(len(classes)),
                classification=classification,
            ),
            "dummy": _scores(
                dummy,
                selected[indices],
                y[indices],
                np.arange(len(classes)),
                classification=classification,
            ),
        }
    metadata = {
        **identity,
        **dataset_details(dataset),
        "model_parameters": parameters,
        "feature_indices": features.tolist(),
        "feature_names": [str(names[i]) for i in features],
        "train_indices": fit.tolist(),
        "background_indices": fit[:16].tolist(),
        "training_rows": len(fit),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "validation_indices": validation.tolist(),
        "test_indices": test.tolist(),
        "split_rules": {
            "test": test_rule,
            "validation": validation_rule,
            "fit_cap": fit_cap,
            "validation_cap": validation_cap,
            "test_cap": test_cap,
        },
        "imputation_medians": medians.tolist(),
        "output": "class_probability" if classification else "prediction",
        "classes": classes.tolist() if classification else None,
        "output_class": classes[1].item() if classification else None,
        "quality": metrics,
        "structure": _tree_diagnostics(model, profile),
    }
    if profile == "xgboost":
        metadata["best_iteration"] = int(model.best_iteration)
    return PreparedModel(
        model,
        selected[fit],
        y[fit],
        selected[validation],
        y[validation],
        selected[test],
        y[test],
        metadata,
    )


def prepare_model(
    dataset: str, n_players: int, seed: int, profile: str, *, cache_dir: str | Path | None = None
) -> PreparedModel:
    """Fit once, or authenticate and reuse a local model artifact.

    Cache keys include source, backend versions, raw data and the complete named
    training/profile configuration. A lock prevents duplicate concurrent fits.
    Joblib artifacts are private trusted-local files, never user uploads.
    """
    if profile not in MODEL_PROFILES:
        message = f"Unknown implemented model profile: {profile}"
        raise ValueError(message)
    x, y, names = load_raw_dataset(dataset)
    if type(n_players) is not int or not 1 <= n_players <= min(20, x.shape[1]):
        message = "Model feature count must fit the dataset and enumeration limit of twenty."
        raise ValueError(message)
    source = hashlib.sha256()
    for name in ("models.py", "datasets.py", "dataset_catalog.py", "setup.py"):
        source.update(Path(__file__).with_name(name).read_bytes())
    packages = ["numpy", "scikit-learn", "joblib"] + (["xgboost"] if profile == "xgboost" else [])
    identity = {
        "dataset": dataset,
        "dataset_source": DATASETS[dataset]["source"],
        "task": DATASETS[dataset]["task"],
        "n_players": n_players,
        "instance_seed": seed,
        "model_profile": profile,
        "profile": MODEL_PROFILES[profile],
        "training_profile": TRAINING_PROFILE,
        "dataset_feature_names": [str(name) for name in names],
        "data_arrays": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in (("x", np.asarray(x)), ("y", np.asarray(y)))
        },
        "data_sha256": hashlib.sha256(
            np.asarray(x).tobytes() + np.asarray(y).tobytes()
        ).hexdigest(),
        "model_source_sha256": source.hexdigest(),
        "model_packages": {name: importlib.metadata.version(name) for name in packages},
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    identity["model_key"] = key
    if cache_dir is None:
        return _fit(dataset, n_players, seed, profile, x, y, names, identity)
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / f"{key}.joblib"
    checksum = artifact.with_suffix(".sha256")
    with artifact.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if artifact.exists() and checksum.exists():
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if digest != checksum.read_text().strip():
                message = "Cached model artifact checksum mismatch."
                raise ValueError(message)
            prepared = joblib.load(artifact)
            if any(prepared.metadata.get(name) != value for name, value in identity.items()):
                message = "Cached model identity disagrees with the requested training recipe."
                raise ValueError(message)
        else:
            prepared = _fit(dataset, n_players, seed, profile, x, y, names, identity)
            temporary = artifact.with_suffix(".tmp")
            joblib.dump(prepared, temporary)
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            temporary.replace(artifact)
            checksum.write_text(digest + "\n")
        prepared.metadata["model_artifact_sha256"] = digest
        return prepared
