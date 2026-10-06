"""Dataset loading for benchmark recipes; shipped loaders own encoding and targets."""

from __future__ import annotations

import importlib
import json
import os
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

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
    if directory := os.environ.get("SHAPIQ_BENCHMARK_DATA_CACHE"):
        with np.load(Path(directory) / f"{name}.npz", allow_pickle=False) as cached:
            metadata = json.loads(str(cached["metadata"].item()))
            if metadata["dataset"] != name or metadata["source"] != DATASETS[name]["source"]:
                message = f"Dataset cache identity differs for {name}."
                raise ValueError(message)
            result = cached["x"].copy(), cached["y"].copy(), metadata["feature_names"]
    else:
        result = _load_shipped_dataset(name)
    _validate_raw_dataset(name, *result)
    return result


def _load_shipped_dataset(name: str) -> tuple:
    """Read the loader directly; cache warming uses this before parallel workers start."""
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
    return x, y, feature_names


def _validate_raw_dataset(name: str, x: np.ndarray, y: np.ndarray, feature_names: list) -> None:
    info = DATASETS[name]
    if (
        x.ndim != 2
        or y.ndim != 1
        or x.shape[1] != info["n_features"]
        or len(feature_names) != x.shape[1]
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


def cache_raw_dataset(name: str, directory: str | Path) -> Path:
    """Sequentially warm one raw dataset; workers only read the resulting numeric NPZ.

    This deliberately bypasses the optional read cache. TabArena data are read
    again from their loader's CSV before freezing; Wine Quality is fetched once
    by this warmup instead of once per parallel model preparation.
    """
    x, y, feature_names = _load_shipped_dataset(name)
    _validate_raw_dataset(name, x, y, feature_names)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{name}.npz"
    metadata = json.dumps(
        {"dataset": name, "source": DATASETS[name]["source"], "feature_names": feature_names}
    )
    with NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, x=x, y=y, metadata=metadata)
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return output


@lru_cache(maxsize=8)
def load_dataset(name: str, instance_seed: int = 0, *, train_limit: int = 512) -> tuple:
    """Reproduce the bounded legacy split and training-only missing-input repair."""
    if type(train_limit) is not int or train_limit < 1:
        message = "train_limit must be a positive integer."
        raise ValueError(message)
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
    if (
        (name in ADDITIONAL_DATASETS or train_limit != 512)
        and classification
        and len(train) > train_limit
    ):
        train, _ = train_test_split(
            train,
            train_size=train_limit,
            stratify=y[train],
            random_state=instance_seed,
        )
    train, test = train[:train_limit], test[:128]
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


def nested_stratified_rows(rows: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    """Order rows once so every larger prefix contains the smaller player set.

    Start with one example per class, then fill the largest proportional deficit.
    Each class has its own seeded shuffled queue; no held-out labels are used.
    """
    classes, counts = np.unique(labels[rows], return_counts=True)
    rng = np.random.default_rng(seed)
    queues = [rng.permutation(rows[labels[rows] == label]) for label in classes]
    used = np.zeros(len(classes), dtype=int)
    order = []
    for k in range(len(rows)):
        if k < len(classes):
            chosen = k
        else:
            deficit = counts * ((k + 1) / len(rows)) - used
            deficit[used == counts] = -np.inf
            chosen = int(np.argmax(deficit))
        order.append(queues[chosen][used[chosen]])
        used[chosen] += 1
    return np.asarray(order, dtype=rows.dtype)


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
