"""Representative shipped-game recipes preserve boundedness and payoff semantics."""

from __future__ import annotations

import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from shapiq_benchmark.families import FAMILY_CATALOG, make_family


@pytest.mark.parametrize("name", FAMILY_CATALOG)
def test_recipe_is_bounded_reproducible_and_explicit(name: str) -> None:
    """Fresh recipes reproduce sampled realizations; deterministic ones ignore batching."""
    for dependency in FAMILY_CATALOG[name]["dependencies"]:
        pytest.importorskip(dependency)
    with threadpool_limits(limits=1):
        game, metadata = make_family(name)
        assert 1 <= game.n_players <= 8
        coalitions = (
            np.random.default_rng(123).integers(0, 2, size=(5, game.n_players)).astype(bool)
        )
        coalitions[0], coalitions[-1] = False, True
        values = game(coalitions)
        assert values.shape == (5,)
        assert np.all(np.isfinite(values))
        json.dumps(metadata, allow_nan=False)
        recreated, repeated_metadata = make_family(name)
        assert metadata == repeated_metadata
        np.testing.assert_allclose(values, recreated(coalitions), rtol=1e-12, atol=1e-12)
        if not metadata["stochastic_frozen"]:
            np.testing.assert_allclose(values, game(coalitions), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(values[::-1], game(coalitions[::-1]), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(
                values,
                np.concatenate([game(row[None]) for row in coalitions]),
                rtol=1e-12,
                atol=1e-12,
            )
        if not metadata["synthetic"]:
            assert metadata["dataset"] in ("california_housing", "iris")
            assert len(metadata["data_sha256"]) == 64
            assert set(metadata["train_indices"]).isdisjoint(metadata["test_indices"])
            if metadata["player_unit"] == "feature":
                assert game.n_players == (4 if metadata["dataset"] == "iris" else 8)


def test_dataset_players_are_full_training_groups() -> None:
    """Grouped valuation retains all real training rows with disjoint players."""
    game, metadata = make_family("dataset_valuation")
    groups = metadata["group_indices"]
    assert game.n_players == len(groups) == 8
    assert sorted(row for group in groups for row in group) == sorted(metadata["train_indices"])
    assert all(len(group) == 64 for group in groups)


def test_baseline_companion_preserves_original_case() -> None:
    """The fixed forest companion retains the original zero game and point selection."""
    original, original_metadata = make_family("local_baseline")
    companion, metadata = make_family("local_baseline_forest")
    coalitions = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(bool)
    np.testing.assert_array_equal(original(coalitions), np.zeros(256))
    values = companion(coalitions)
    assert np.all(np.isfinite(values))
    assert np.ptp(values) > 0
    for key in ("point_row", "background_indices", "train_indices", "test_indices"):
        assert metadata[key] == original_metadata[key]
    assert metadata["model"] == "RandomForestRegressor"
    assert metadata["model_parameters"]["n_estimators"] == 8
    assert metadata["model_parameters"]["max_depth"] == 4
    assert metadata["model_parameters"]["random_state"] == 0
    assert metadata["model_parameters"]["n_jobs"] == 1
    assert metadata["application_family"] == "local_explanation"


def test_knn_uses_held_out_true_label_and_fixed_denominator() -> None:
    """The row utility targets the selected real point's class, with exactly k denominator."""
    game, metadata = make_family("knn")
    assert metadata["class_index"] == metadata["point_label"]
    assert metadata["point_row"] in metadata["test_indices"]
    for row in range(game.n_players):
        coalition = np.zeros((1, game.n_players), dtype=bool)
        coalition[0, row] = True
        assert game(coalition)[0] == (
            1 / 3 if game.y_train_indices[row] == metadata["point_label"] else 0
        )


def test_synthetics_are_separate_diagnostics() -> None:
    """Synthetic payoff fixtures cannot silently enter the real-data headline."""
    assert {name for name, info in FAMILY_CATALOG.items() if info["synthetic"]} == {
        "unanimity",
        "soum",
        "dummy",
        "random",
    }
    assert FAMILY_CATALOG["local_marginal"]["application_family"] == "local_explanation"
    assert FAMILY_CATALOG["interventional_tree"]["application_family"] == "local_explanation"
    assert FAMILY_CATALOG["dataset_valuation"]["application_family"] == "data_valuation"
