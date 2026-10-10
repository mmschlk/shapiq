"""Explicit neighbor recipes preserve nested players, columns and fitting-only inputs."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq_benchmark import datasets
from shapiq_benchmark.games import load_game, prepare_structured


@pytest.fixture
def neighbor_data(monkeypatch):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(1800, 8))
    y = np.arange(len(x)) % 3
    monkeypatch.setitem(datasets.DATASETS, "fixture", {"task": "classification", "n_features": 8})
    monkeypatch.setattr(
        datasets, "load_raw_dataset", lambda name: (x, y, [f"f{i}" for i in range(8)])
    )
    datasets.load_dataset.cache_clear()
    yield x, y
    datasets.load_dataset.cache_clear()


def test_explicit_pool_keeps_legacy_default_and_no_heldout_players(neighbor_data):
    _, labels, legacy, test, _ = datasets.load_dataset("fixture", 2)
    _, _, explicit_legacy, explicit_test, _ = datasets.load_dataset("fixture", 2, train_limit=512)
    np.testing.assert_array_equal(legacy, explicit_legacy)
    np.testing.assert_array_equal(test, explicit_test)
    _, _, pool, heldout, _ = datasets.load_dataset("fixture", 2, train_limit=1280)
    assert len(pool) == 1280 and not set(pool).intersection(heldout)
    ordered = datasets.nested_stratified_rows(pool, labels, 2)
    assert set(ordered) == set(pool) and len(set(ordered)) == len(pool)
    assert set(labels[ordered[:32]]) == set(labels)
    np.testing.assert_array_equal(ordered, datasets.nested_stratified_rows(pool, labels, 2))
    for count in (32, 128, 1024):
        assert (
            max(np.bincount(labels[ordered[:count]])) - min(np.bincount(labels[ordered[:count]]))
            <= 1
        )


@pytest.mark.parametrize("kind", ["knn", "tnn"])
def test_large_neighbors_keep_nested_rows_and_features(neighbor_data, tmp_path, kind):
    games = []
    for count, width in [(32, 4), (1024, 8)]:
        game = prepare_structured(
            [
                {
                    "id": f"{kind}-{count}",
                    "oracle": kind,
                    "dataset": "fixture",
                    "index": "SV",
                    "order": 1,
                    "instance_seed": 2,
                    "n_players": count,
                    "training_rows": 1280,
                    "row_selection": "nested_stratified",
                    "input_features": width,
                    "feature_rule": "nested",
                }
            ],
            tmp_path,
        )[0]
        with np.load(tmp_path / game["artifact"], allow_pickle=False) as arrays:
            games.append((game, arrays["selected_rows"].copy(), arrays["test_indices"].copy()))
            assert arrays["x_train"].shape == (count, width)
        assert game["metadata"]["fitting_rows"] == 1280
        oracle = load_game(game, tmp_path)
        assert np.isfinite(oracle(np.array([[False] * count, [True] * count]))).all()
        assert len(game["truth"]["values"]) == count
    np.testing.assert_array_equal(games[0][1], games[1][1][:32])
    np.testing.assert_array_equal(games[0][2], games[1][2])
    assert set(games[0][0]["metadata"]["feature_indices"]) < set(
        games[1][0]["metadata"]["feature_indices"]
    )


def test_unavailable_neighbor_count_is_not_silently_shrunk(neighbor_data, tmp_path):
    with pytest.raises(ValueError, match="training split size"):
        prepare_structured(
            [
                {
                    "id": "too-many",
                    "oracle": "knn",
                    "dataset": "fixture",
                    "index": "SV",
                    "order": 1,
                    "n_players": 1024,
                    "training_rows": 512,
                    "row_selection": "nested_stratified",
                }
            ],
            tmp_path,
        )
