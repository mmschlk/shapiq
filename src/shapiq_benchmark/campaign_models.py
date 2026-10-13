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

    from shapiq_benchmark.models import ModelName
    from shapiq_games.typing import Task


import joblib
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, log_loss, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shapiq_benchmark.campaign_datasets import DATASETS, dataset_details, load_raw_dataset
from shapiq_benchmark.models import build_model as registered_model
from shapiq_benchmark.quality import QUALITY_PROTOCOL, model_validation_check, validate_protocol

MODEL_PROFILES: dict = {
    "random_forest": {
        "label": "Random forest",
        "parameters": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 5, "n_jobs": 1},
    },
    "linear": {"label": "Linear control", "parameters": {}},
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
        classification = self.metadata["task"] == "classification"
        if not len(x):
            return np.empty(0)
        return self.model.predict_proba(x)[:, 1] if classification else self.model.predict(x)


def build_model(
    profile: str,
    task: str,
    seed: int,
    *,
    parameters: dict | None = None,
    device: str = "cpu",
    quality_protocol: str | None = None,
) -> object:
    """Build a compact model with scaling learned only from fitting rows."""
    validate_protocol(quality_protocol)
    if device != "cpu":
        message = "Compact models require CPU execution."
        raise ValueError(message)
    params: dict = {**MODEL_PROFILES[profile]["parameters"], **(parameters or {})}
    if profile == "linear":
        params = {**({"max_iter": 1000} if task == "classification" else {"alpha": 1.0}), **params}
    if profile == "linear":
        constructor = LogisticRegression if task == "classification" else Ridge
        estimator = constructor(random_state=seed, **params)
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    return registered_model(
        cast("ModelName", profile), cast("Task", task), random_state=seed, **params
    )


def coalition_model(prepared: PreparedModel) -> object:
    """Clone the fixed predictor without fitting or tuning on held-out rows."""
    return clone(prepared.model)


def _fit_predictor(
    profile: str,
    task: str,
    seed: int,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    *,
    device: str,
    quality_protocol: str | None = None,
) -> tuple[Any, dict, dict]:
    """Fit the prescribed model; compact profiles have no hyperparameter search."""
    parameters = {**MODEL_PROFILES[profile]["parameters"]}
    model = cast(
        "Any",
        build_model(
            profile,
            task,
            seed,
            parameters=parameters,
            device=device,
            quality_protocol=quality_protocol,
        ),
    )
    model.fit(x_fit, y_fit)
    diagnostics = {}
    if isinstance(model, Pipeline):
        scaler = model.named_steps["scaler"]
        diagnostics.update(
            scaling="fitting-row StandardScaler",
            mean=scaler.mean_.tolist(),
            scale=scaler.scale_.tolist(),
        )
    return model, {**parameters, "random_state": seed}, diagnostics


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


def _tree_diagnostics(model: object) -> dict:
    """Describe the fitted forest without treating depth as game difficulty."""
    model = cast("Any", model)
    features, depths, widths = set(), [], []
    for estimator in model.estimators_:
        tree = estimator.tree_
        stack = [(0, 0, frozenset())]
        max_depth = max_width = 0
        while stack:
            node, depth, path = stack.pop()
            feature = int(tree.feature[node])
            if feature >= 0:
                features.add(feature)
                path = path | {feature}
                stack.extend(
                    (int(child), depth + 1, path)
                    for child in (tree.children_left[node], tree.children_right[node])
                )
            else:
                max_depth, max_width = max(max_depth, depth), max(max_width, len(path))
        depths.append(max_depth)
        widths.append(max_width)
    return {
        "tree_count": len(model.estimators_),
        "tree_depths": depths,
        "features_used": sorted(features),
        "max_distinct_features_on_path": max(widths),
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
    training = identity["training_profile"]
    classes, encoded = np.unique(original_y, return_inverse=True)
    y = encoded if classification else original_y
    rng = np.random.default_rng(seed)
    features = np.sort(
        rng.permutation(x.shape[1])[:n_players]
        if identity.get("feature_rule") == "nested"
        else rng.choice(x.shape[1], n_players, replace=False)
    )
    fit, test, test_rule = _split(
        np.arange(len(x)), y, training["test_fraction"], seed, classification=classification
    )
    fit, validation, validation_rule = _split(
        fit,
        y,
        training["validation_fraction_of_remainder"],
        seed,
        classification=classification,
    )
    fit, fit_cap = _bounded(fit, y, training["fit_rows_max"], seed, classification=classification)
    validation, validation_cap = _bounded(
        validation, y, training["validation_rows_max"], seed, classification=classification
    )
    test, test_cap = _bounded(
        test, y, training["test_rows_max"], seed, classification=classification
    )
    if classification and len(np.unique(y[fit])) != len(classes):
        message = "Fitting partition omits a class; this seeded instance is unsupported."
        raise ValueError(message)
    if identity.get("feature_rule") == "continuous":
        categorical = cast("Any", DATASETS[dataset].get("categorical_features", []))
        eligible = [
            i
            for i, name in enumerate(names)
            if categorical != "all"
            and name not in categorical
            and len(np.unique(x[fit, i][np.isfinite(x[fit, i])])) > 2
        ]
        if len(eligible) < n_players:
            message = "Too few continuous fitting-row features for this game."
            raise ValueError(message)
        features = np.sort(np.random.default_rng(seed).choice(eligible, n_players, replace=False))
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
    model, parameters, diagnostics = _fit_predictor(
        profile,
        str(DATASETS[dataset]["task"]),
        seed,
        selected[fit],
        y[fit],
        device=identity.get("device", "cpu"),
        quality_protocol=identity.get("quality_protocol"),
    )
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
        "structure": (_tree_diagnostics(model) if profile == "random_forest" else diagnostics),
    }
    if identity.get("quality_protocol") == QUALITY_PROTOCOL:
        metadata["model_validation_gate"] = model_validation_check(metadata)
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
    dataset: str,
    n_players: int,
    seed: int,
    profile: str,
    *,
    cache_dir: str | Path | None = None,
    device: str = "cpu",
    feature_rule: str = "all",
    quality_protocol: str | None = None,
) -> PreparedModel:
    """Fit once, or authenticate and reuse a local model artifact.

    Cache keys include source, backend versions, raw data and the complete named
    training/profile configuration. A lock prevents duplicate concurrent fits.
    Joblib artifacts are private trusted-local files, never user uploads.
    """
    validate_protocol(quality_protocol)
    if profile not in MODEL_PROFILES:
        message = f"Unknown implemented model profile: {profile}"
        raise ValueError(message)
    if device != "cpu":
        message = "Compact models require CPU execution."
        raise ValueError(message)
    if feature_rule not in {"all", "continuous", "nested"}:
        message = "Unknown model feature selection rule."
        raise ValueError(message)
    x, y, names = load_raw_dataset(dataset)
    if type(n_players) is not int or not 1 <= n_players <= x.shape[1]:
        message = "Model feature count must fit the dataset."
        raise ValueError(message)
    source = hashlib.sha256()
    for name in (
        "campaign_models.py",
        "campaign_datasets.py",
        "dataset_catalog.py",
        "models.py",
        "quality.py",
    ):
        source.update(Path(__file__).with_name(name).read_bytes())
    # Registry preprocessing participates in the preparation identity as well as
    # the loaded array bytes; published replay keeps its original source pin.
    for path in sorted(Path(__file__).with_name("datasets").glob("*.py")):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    packages = ["numpy", "scikit-learn", "joblib"]
    training = {**TRAINING_PROFILE}
    if "fit_rows_max" in MODEL_PROFILES[profile]:
        training["id"] = f"quality-v1-{profile}"
        training["fit_rows_max"] = MODEL_PROFILES[profile]["fit_rows_max"]
    identity = {
        "dataset": dataset,
        "dataset_source": DATASETS[dataset]["source"],
        "task": DATASETS[dataset]["task"],
        "n_players": n_players,
        "instance_seed": seed,
        "model_profile": profile,
        "profile": MODEL_PROFILES[profile],
        "training_profile": training,
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
    if feature_rule != "all":
        identity["feature_rule"] = feature_rule
    if quality_protocol is not None:
        identity["quality_protocol"] = quality_protocol
        training["id"] = training["id"].replace("quality-v1", quality_protocol)
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
