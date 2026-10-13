"""Compact constructors preserve explicit players, holdout rows and baseline semantics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from shapiq_benchmark.campaign_models import coalition_model, prepare_model
from shapiq_benchmark.families import (
    CONSTRUCTION_PROTOCOL,
    FAMILY_CATALOG,
    dataset_compatibility,
    make_family,
    profile_compatibility,
)


@pytest.mark.parametrize("name", ["local_baseline", "feature_selection", "dataset_valuation"])
def test_compact_family_roundtrip(name, tmp_path):
    options = {
        "dataset": "breast_cancer",
        "n_players": 4,
        "model_profile": "linear",
        "model_cache": str(tmp_path),
        "instance_seed": 2,
    }
    game, metadata = make_family(name, **options)
    repeated, again = make_family(name, **options)
    coalitions = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(bool)
    values = game(coalitions)
    np.testing.assert_array_equal(values, repeated(coalitions))
    np.testing.assert_allclose(values, game(coalitions[::-1])[::-1], atol=1e-14)
    assert np.isfinite(values).all()
    assert metadata == again
    assert metadata["construction_protocol"] == CONSTRUCTION_PROTOCOL
    assert set(metadata["train_indices"]).isdisjoint(metadata["test_indices"])
    assert game.n_players == 4


def test_feature_selection_preserves_campaign_empty_zero(tmp_path):
    game, metadata = make_family(
        "feature_selection",
        dataset="breast_cancer",
        n_players=4,
        model_profile="linear",
        model_cache=str(tmp_path),
    )
    prepared = prepare_model("breast_cancer", 4, 0, "linear", cache_dir=str(tmp_path))
    assert game.normalization_value == 0
    assert game(np.zeros((1, 4), bool))[0] == 0
    fitted = clone(coalition_model(prepared)).fit(prepared.x_train, prepared.y_train)
    expected = accuracy_score(prepared.y_test, fitted.predict(prepared.x_test))
    assert game(np.ones((1, 4), bool))[0] == pytest.approx(expected)
    assert metadata["refit_protocol"]["empty"] == "zero"


def test_group_players_cover_training_only(tmp_path):
    game, metadata = make_family(
        "dataset_valuation",
        dataset="breast_cancer",
        n_players=4,
        model_profile="linear",
        input_features=8,
        model_cache=str(tmp_path),
    )
    groups = metadata["group_indices"]
    rows = [row for group in groups for row in group]
    assert rows == metadata["train_indices"]
    assert len(rows) == len(set(rows))
    assert set(rows).isdisjoint(metadata["test_indices"])
    assert metadata["input_features"] == 8
    assert game(np.zeros((1, 4), bool))[0] == 0


def test_compact_catalog_excludes_superseded_recipes():
    assert set(FAMILY_CATALOG) == {
        "local_baseline",
        "feature_selection",
        "dataset_valuation",
        "pathdependent_tree",
        "interventional_tree",
        "knn",
        "tnn",
    }
    assert dataset_compatibility("knn", "diabetes") == "Requires class labels."
    assert profile_compatibility("feature_selection", "breast_cancer", "linear") is None
    assert profile_compatibility("feature_selection", "breast_cancer", "tabpfn_prediction")
    with pytest.raises(ValueError, match="compact"):
        make_family("cluster", dataset="breast_cancer", n_players=4)
    with pytest.raises(ValueError, match="twelve"):
        make_family("local_baseline", dataset="breast_cancer", n_players=13)
