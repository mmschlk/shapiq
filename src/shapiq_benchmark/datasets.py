"""Dataset loading for benchmark recipes; shipped loaders own encoding and targets."""

from __future__ import annotations

import importlib
from functools import lru_cache

import numpy as np
from sklearn.model_selection import train_test_split

from shapiq_benchmark.dataset_catalog import ADDITIONAL_DATASETS

DATASETS = {
    "california_housing": {
        "task": "regression",
        "n_features": 8,
        "source": "shapiq.datasets.load_california_housing",
    },
    "diabetes": {
        "task": "regression",
        "n_features": 10,
        "source": "sklearn.datasets.load_diabetes",
    },
    "bike_sharing": {
        "task": "regression",
        "n_features": 12,
        "source": "shapiq.datasets.load_bike_sharing",
    },
    "iris": {
        "task": "classification",
        "n_features": 4,
        "n_classes": 3,
        "source": "sklearn.datasets.load_iris",
    },
    "wine": {
        "task": "classification",
        "n_features": 13,
        "n_classes": 3,
        "source": "sklearn.datasets.load_wine",
    },
    "breast_cancer": {
        "task": "classification",
        "n_features": 30,
        "n_classes": 2,
        "source": "shapiq_games.datasets.load_breast_cancer",
    },
    "digits": {
        "task": "classification",
        "n_features": 64,
        "n_classes": 10,
        "source": "sklearn.datasets.load_digits",
    },
}


# Keep the old Wine classification key available to reproduce historical suites.
DATASETS.update(ADDITIONAL_DATASETS)


def load_raw_dataset(name: str) -> tuple:
    """Read the shipped loader's stable representation and validate its catalog identity."""
    info = DATASETS[name]
    module, attribute = info["source"].rsplit(".", 1)
    loader = getattr(importlib.import_module(module), attribute)
    loaded = loader()
    if "openml_id" in info:
        # The first download returns OpenML arrays; subsequent calls parse CSV.
        # Always freeze the cached representation, including its float rounding.
        loaded = loader()
    if isinstance(loaded, tuple):
        frame, target = loaded
        feature_names = list(frame.columns)
        x, y = np.asarray(frame, dtype=float), np.asarray(target)
    else:
        x, y = loaded.data, loaded.target
        feature_names = list(loaded.feature_names)
    if (
        x.shape[1] != info["n_features"]
        or len(x) != info.get("n_samples", len(x))
        or len(y) != len(x)
        or not np.isfinite(y).all()
    ):
        message = f"Dataset {name} has unexpected dimensions or nonfinite targets."
        raise ValueError(message)
    classification = info["task"] == "classification"
    if classification and len(np.unique(y)) != info["n_classes"]:
        message = f"Dataset {name} class count disagrees with its declared task."
        raise ValueError(message)
    return x, y, feature_names


@lru_cache(maxsize=8)
def load_dataset(name: str, instance_seed: int = 0) -> tuple:
    """Reproduce the bounded legacy split and training-only missing-input repair."""
    info = DATASETS[name]
    x, y, feature_names = load_raw_dataset(name)
    classification = info["task"] == "classification"
    train, test = train_test_split(
        np.arange(len(x)),
        test_size=0.2,
        random_state=instance_seed,
        stratify=y if classification else None,
    )
    # Preserve existing recipes. New classification data retain rare classes in
    # the bounded training pool instead of taking an arbitrary shuffled prefix.
    if name in ADDITIONAL_DATASETS and classification and len(train) > 512:
        train, _ = train_test_split(
            train,
            train_size=512,
            stratify=y[train],
            random_state=instance_seed,
        )
    train, test = train[:512], test[:128]
    if name in ADDITIONAL_DATASETS and not np.isfinite(x).all():
        x = np.array(x, dtype=float, copy=True)
        if np.isinf(x).any():
            message = f"Dataset {name} contains infinite inputs."
            raise ValueError(message)
        medians = np.nanmedian(x[train], axis=0)
        if not np.isfinite(medians).all():
            message = f"Dataset {name} has a column missing on every training row."
            raise ValueError(message)
        rows, columns = np.where(np.isnan(x))
        x[rows, columns] = medians[columns]
    return x, y, train, test, feature_names


def feature_limit(recipe: str, dataset: str) -> int:
    """Static upper bound; training-row variation is checked at construction."""
    info = DATASETS[dataset]
    if recipe in ("local_gaussian", "local_copula", "cluster"):
        categorical = info.get("categorical_features", [])
        return 0 if categorical == "all" else info["n_features"] - len(categorical)
    return info["n_features"]


def dataset_details(name: str) -> dict:
    """Describe loader semantics alongside the game's recorded data hash and rows."""
    info = DATASETS[name]
    details = {"dataset_source": info["source"]}
    if name in ADDITIONAL_DATASETS:
        details["dataset_preprocessing"] = (
            "Shipped loader preprocessing; remaining missing inputs use medians of the "
            "recorded training rows. Original target values are retained."
        )
        details["dataset_source_url"] = info["source_url"]
        if "target_note" in info:
            details["dataset_target_note"] = info["target_note"]
    return details


def warm_dataset_caches(suite: dict) -> None:
    """Finish TabArena CSV writes before any parallel preparation children start."""
    names = {spec.get("dataset") for spec in [*suite.get("families", []), *suite.get("games", [])]}
    for name in sorted(names - {None}):
        if name in DATASETS and "openml_id" in DATASETS[name]:
            load_dataset(name)
    load_dataset.cache_clear()
