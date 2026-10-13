"""Compact dataset adapters preserve raw labels and training-only imputation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from shapiq_benchmark import campaign_datasets as datasets
from shapiq_benchmark.campaign_datasets import DATASETS, load_dataset
from shapiq_benchmark.datasets import get_dataset_spec

if TYPE_CHECKING:
    from pathlib import Path


def test_compact_registry_has_all_non_sklearn_loaders() -> None:
    for name, entry in DATASETS.items():
        if not entry["source"].startswith("sklearn."):
            assert callable(get_dataset_spec(name).loader)
    assert DATASETS["wine"]["task"] == "classification"
    assert DATASETS["wine_quality"]["task"] == "regression"


def test_missing_inputs_use_only_training_rows_and_keep_signed_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, test = train_test_split(np.arange(40), test_size=0.2, random_state=2)
    x = np.arange(80, dtype=float).reshape(40, 2)
    x[test] = 1e9
    x[train[0], 0] = x[test[0], 0] = np.nan
    y = np.linspace(-10, 10, 40)
    monkeypatch.setitem(
        DATASETS, "nhanesi", {**DATASETS["nhanesi"], "n_features": 2, "n_samples": 40}
    )
    monkeypatch.setattr(
        datasets,
        "get_dataset_spec",
        lambda _: SimpleNamespace(loader=lambda: (pd.DataFrame(x), pd.Series(y))),
    )
    load_dataset.cache_clear()
    try:
        actual, labels, used_train, used_test, _ = load_dataset("nhanesi", 2)
        assert actual[train[0], 0] == actual[test[0], 0] == np.nanmedian(x[train, 0])
        np.testing.assert_array_equal(labels, y)
        np.testing.assert_array_equal(used_train, train)
        np.testing.assert_array_equal(used_test, test)
        assert np.isnan(x[train[0], 0])
    finally:
        load_dataset.cache_clear()


def test_catalog_dimension_drift_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        datasets,
        "get_dataset_spec",
        lambda _: SimpleNamespace(
            loader=lambda: (pd.DataFrame(np.ones((40, 21))), pd.Series([0, 1] * 20))
        ),
    )
    load_dataset.cache_clear()
    try:
        with pytest.raises(ValueError, match="unexpected dimensions"):
            load_dataset("mushroom")
    finally:
        load_dataset.cache_clear()


def test_optional_raw_cache_never_falls_back_to_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A warmed NPZ preserves bytes/names and missing entries fail before loader calls."""
    from shapiq_benchmark import campaign_datasets as datasets

    calls = []
    x = np.arange(40, dtype=float).reshape(20, 2)
    x[0, 0] = np.nan
    y = np.arange(20, dtype=np.int64)
    names = ["first", "second"]
    monkeypatch.setitem(
        DATASETS, "fixture", {"source": "fixture.loader", "task": "regression", "n_features": 2}
    )

    def loader(name: str) -> tuple:
        calls.append(name)
        return x, y, names

    monkeypatch.setattr(datasets, "_load_shipped_dataset", loader)
    path = datasets.cache_raw_dataset("fixture", tmp_path)
    monkeypatch.setenv("SHAPIQ_BENCHMARK_DATA_CACHE", str(tmp_path))
    actual, targets, columns = datasets.load_raw_dataset("fixture")
    np.testing.assert_array_equal(actual, x)
    np.testing.assert_array_equal(targets, y)
    assert actual.dtype == x.dtype and targets.dtype == y.dtype and columns == names
    assert calls == ["fixture"]
    monkeypatch.setitem(DATASETS, "fixture", {**DATASETS["fixture"], "source": "another.loader"})
    with pytest.raises(ValueError, match="identity differs"):
        datasets.load_raw_dataset("fixture")
    path.unlink()
    with pytest.raises(FileNotFoundError):
        datasets.load_raw_dataset("fixture")
    assert calls == ["fixture"]
    monkeypatch.delenv("SHAPIQ_BENCHMARK_DATA_CACHE")
    datasets.load_raw_dataset("fixture")
    assert calls == ["fixture", "fixture"]


def test_registry_text_labels_are_encoded_without_imputing_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 4.0, 6.0]})
    labels = pd.Series(["z", "a", "z"])
    monkeypatch.setattr(
        datasets, "get_dataset_spec", lambda _: SimpleNamespace(loader=lambda: (frame, labels))
    )
    x, y, names = datasets._load_shipped_dataset("adult_census")
    np.testing.assert_array_equal(y, [1, 0, 1])
    assert np.isnan(x[1, 0])
    assert names == ["a", "b"]
