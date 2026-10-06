"""Shipped datasets keep their targets and use reproducible, training-only repair."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import shapiq_games.datasets as shipped
from shapiq_benchmark.dataset_catalog import ADDITIONAL_DATASETS
from shapiq_benchmark.datasets import DATASETS, dataset_details, feature_limit, load_dataset
from shapiq_benchmark.families import make_family
from shapiq_benchmark.matrix import expand_matrix


def test_matrix_uses_every_shipped_tabarena_loader_and_wine_quality() -> None:
    """An exported TabArena loader cannot quietly disappear from the requested matrix."""
    expected = {
        name.removeprefix("load_") for name in dir(shipped) if name.startswith("load_tabarena_")
    }
    directory = Path(__file__).resolve().parents[2] / "benchmark/suites"
    config = json.loads((directory / "matrix.json").read_text())
    assert len(expected) == 51
    assert expected <= set(config["datasets"])
    assert set(ADDITIONAL_DATASETS) <= set(config["datasets"])
    assert "wine_quality" in config["datasets"] and "wine" not in config["datasets"]
    assert DATASETS["wine_quality"]["task"] == "regression"
    assert DATASETS["wine"]["task"] == "classification"  # historical reproduction
    for name in ADDITIONAL_DATASETS:
        assert callable(getattr(shipped, DATASETS[name]["source"].rsplit(".", 1)[1]))
    suite = expand_matrix(config, json.loads((directory / config["base_suite"]).read_text()))
    rows = {
        (row["recipe"], row["dataset"], row["n_players"]): row
        for row in suite["matrix_coverage"]["candidates"]
    }
    for recipe in ("local_gaussian", "local_copula", "cluster"):
        assert rows[recipe, "mushroom", 11]["reason"] == "insufficient_continuous_features"
        assert feature_limit(recipe, "wine_quality") == 11
    assert rows["local_gaussian", "wine_quality", 11]["status"] == "selected"
    assert rows["local_gaussian", "wine_quality", 12]["status"] == "excluded"
    assert rows["knn", "wine_quality", 11]["reason"] == "requires_class_labels"
    assert "regression surrogate" in dataset_details("nhanesi")["dataset_target_note"]


def test_missing_inputs_use_only_training_rows_and_do_not_modify_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Held-out outliers cannot affect imputation; signed NHANES targets remain signed."""
    train, test = train_test_split(np.arange(40), test_size=0.2, random_state=2)
    x = np.arange(80, dtype=float).reshape(40, 2)
    x[test] = 1e9
    x[train[0], 0] = np.nan
    x[test[0], 0] = np.nan
    y = np.linspace(-10, 10, 40)
    monkeypatch.setitem(
        DATASETS, "nhanesi", {**DATASETS["nhanesi"], "n_features": 2, "n_samples": 40}
    )
    monkeypatch.setattr(shipped, "load_nhanesi", lambda: (pd.DataFrame(x), pd.Series(y)))
    load_dataset.cache_clear()
    try:
        actual, labels, used_train, used_test, _ = load_dataset("nhanesi", 2)
        expected = np.nanmedian(x[train, 0])
        assert actual[train[0], 0] == actual[test[0], 0] == expected
        np.testing.assert_array_equal(labels, y)
        np.testing.assert_array_equal(used_train, train)
        np.testing.assert_array_equal(used_test, test)
        assert np.isnan(x[train[0], 0])  # no mutation of the loader's data
    finally:
        load_dataset.cache_clear()


def test_catalog_dimension_drift_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upstream changes require an explicit new dataset identity and matrix inventory."""
    monkeypatch.setattr(
        shipped, "load_mushroom", lambda: (pd.DataFrame(np.ones((40, 21))), pd.Series([0, 1] * 20))
    )
    load_dataset.cache_clear()
    with pytest.raises(ValueError, match="unexpected dimensions"):
        load_dataset("mushroom")
    load_dataset.cache_clear()


@pytest.mark.parametrize(
    "dataset", ["adult_census", "mushroom", "ionosphere", "nhanesi", "communities_and_crime"]
)
def test_shipped_local_games_have_finite_singletons(dataset: str) -> None:
    """Each newly bundled dataset constructs a genuine eleven-player game."""
    game, metadata = make_family("local_baseline", dataset=dataset, n_players=11)
    coalitions = np.vstack(
        [np.zeros(11, dtype=bool), np.eye(11, dtype=bool), np.ones(11, dtype=bool)]
    )
    assert np.isfinite(game(coalitions)).all()
    assert metadata["dataset_source"] == DATASETS[dataset]["source"]
    assert len(metadata["feature_indices"]) == 11
    assert set(metadata["train_indices"]).isdisjoint(metadata["test_indices"])


@pytest.mark.parametrize("recipe", ["knn", "tnn", "weighted_knn", "binary_weighted_knn"])
def test_neighbor_games_use_class_indices_without_relabeling_targets(
    recipe: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equivalent class names must yield identical utilities, including binary comparisons."""
    from shapiq_benchmark import families

    original, _ = make_family(recipe, dataset="wine", n_players=11, instance_seed=1)
    x, y, train, test, names = load_dataset("wine", 1)
    labels = (y + 1) * 10
    monkeypatch.setattr(families, "_dataset", lambda *args: (x, labels, train, test, names))
    renamed, metadata = make_family(recipe, dataset="wine", n_players=11, instance_seed=1)
    coalitions = np.vstack(
        [np.zeros(11, dtype=bool), np.eye(11, dtype=bool), np.ones(11, dtype=bool)]
    )
    np.testing.assert_allclose(renamed(coalitions), original(coalitions), rtol=0, atol=0)
    assert metadata["class_labels"][metadata["class_index"]] == metadata["point_label"]
    assert metadata["point_label"] in (10, 20, 30)


def test_first_download_and_cache_reload_produce_identical_game_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenML's initial arrays cannot define a different game from its rounded CSV cache."""
    name = "tabarena_airfoil_self_noise"
    cached = np.arange(80, dtype=float).reshape(40, 2)
    calls = []

    def loader() -> tuple:
        calls.append(1)
        return pd.DataFrame(cached + (1e-12 if len(calls) == 1 else 0)), pd.Series(np.arange(40))

    monkeypatch.setitem(DATASETS, name, {**DATASETS[name], "n_features": 2, "n_samples": 40})
    monkeypatch.setattr(shipped, "load_tabarena_airfoil_self_noise", loader)
    load_dataset.cache_clear()
    try:
        first = load_dataset(name)[0]
        load_dataset.cache_clear()
        repeated = load_dataset(name)[0]
        np.testing.assert_array_equal(first, cached)
        np.testing.assert_array_equal(first, repeated)
    finally:
        load_dataset.cache_clear()


def test_optional_raw_cache_never_falls_back_to_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A warmed NPZ preserves bytes/names and missing entries fail before loader calls."""
    from shapiq_benchmark import datasets

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
