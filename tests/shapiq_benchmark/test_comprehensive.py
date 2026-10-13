"""Bounded checks of the compact manifest; no dataset downloads or model fits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from shapiq_benchmark import (
    campaign_runner as runner,
    comprehensive as c,
    focused,
)
from shapiq_benchmark.campaign_datasets import DATASETS

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmark/suites/comprehensive.csv"


@pytest.fixture
def approved_constructors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manifest-only stand-in for the approved frozen estimator signatures."""

    def constructor(*, ridge: float, low_budget_equal_allocation: bool) -> None:
        pass

    for name in focused.METHOD_PARAMETERS:
        monkeypatch.setitem(runner.METHODS, name, constructor)


def test_fixed_manifest_and_balanced_dataset_inventory() -> None:
    assert MANIFEST.read_text() == c.manifest_text() == c.manifest_text()
    assert len(c.CLASSIFICATION) == len(c.REGRESSION) == 12
    assert len(set(c.CLASSIFICATION + c.REGRESSION)) == 24
    assert all(DATASETS[d]["task"] == "classification" for d in c.CLASSIFICATION)
    assert all(DATASETS[d]["task"] == "regression" for d in c.REGRESSION)
    rows = c.recipes()
    assert len(rows) == len({r["id"] for r in rows}) == 512
    assert Counter(r["reference"] for r in rows) == {
        "enumeration": 389,
        "tree_solver": 88,
        "neighbor_solver": 35,
    }


def test_dimensions_are_independent_and_nested() -> None:
    rows = c.recipes()
    excluded = {r["id"] for r in json.loads(c.PREPARED_SELECTION.read_text())["excluded_core"]}
    assert all(r["feature_rule"] == "nested" for r in rows)
    for dataset in c.CLASSIFICATION + c.REGRESSION:
        width = DATASETS[dataset]["n_features"]
        core = [r for r in rows if r["dataset"] == dataset and r["model"] == "linear"]
        for family in ("local_baseline", "feature_selection"):
            actual = {int(r["n_players"]) for r in core if r["construction"] == family}
            intended = {n for n in c.GENERIC_COUNTS if n <= width} | (
                {width} if width < 12 else set()
            )
            assert actual == {
                n
                for n in intended
                if f"comprehensive-{family}-{dataset}-linear-p{n}" not in excluded
            }
        assert {
            int(r["n_players"])
            for r in core
            if r["construction"] == "dataset_valuation"
            and int(r["input_features"]) == min(12, width)
        } == {
            n
            for n in c.GENERIC_COUNTS
            if f"comprehensive-dataset_valuation-{dataset}-linear-p{n}-f{min(12, width)}"
            not in excluded
        }
    assert all(
        int(r["input_features"]) == min(12, DATASETS[r["dataset"]]["n_features"])
        for r in rows
        if r["construction"] == "dataset_valuation"
    )
    assert all(r["model"] in c.CORE_MODELS for r in rows if r["reference"] == "enumeration")
    assert all(
        r["model"] == "random_forest" and int(r["n_players"]) in {8, 16, 32}
        for r in rows
        if r["reference"] == "tree_solver"
    )


def test_prepared_neighbor_replacements_preserve_original_settings() -> None:
    selection = json.loads(c.PREPARED_SELECTION.read_text())
    neighbors = [r for r in c.recipes() if r["reference"] == "neighbor_solver"]
    assert len(selection["excluded_core"]) == len(neighbors) == 35
    assert neighbors == selection["neighbor_recipes"]
    assert all(
        r["dataset"] in c.CLASSIFICATION
        and int(r["n_players"]) in {32, 64}
        and r["training_rows"] == "5000"
        and r["row_selection"] == "nested_stratified"
        and int(r["input_features"]) == min(12, DATASETS[r["dataset"]]["n_features"])
        for r in neighbors
    )
    assert {r["application"] for r in c.recipes()} == {"local", "data", "features"}


def test_suite_preserves_settings_and_reference_targets(approved_constructors: None) -> None:
    suite = c.build_suite(MANIFEST)
    assert suite["seeds"] == [0, 1, 2] and suite["game_seeds"] == [0, 1, 2, 3]
    assert len(suite["methods"]) == 22
    assert suite["method_parameters"] == focused.METHOD_PARAMETERS
    assert suite["relative_budgets"] == [0.5, 1, 2, 4, 8, 16, 32]
    assert suite["min_players"] == 4
    assert len(suite["families"]) == 389
    assert len(suite["games"]) == 431
    for game in suite["games"]:
        if game["oracle"] in {"knn", "tnn"}:
            assert (game["index"], game["order"]) == ("SV", 1)
            assert game["training_rows"] == 5000
        elif game["oracle"] == "pathdependent_tree":
            assert game["index"] in {"SV", "k-SII", "SII"}
    assert c.summary(suite)["intended_targets"] == 11060


def test_retained_recipe_settings_equal_historical_subset() -> None:
    # Exact historical dictionaries, in the frozen prepared-selection order.
    assert (
        hashlib.sha256(
            json.dumps(c.recipes(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == "5e8b5539ca82aaaee02a5cef8afb1bea8fad8f936a367f199b051d4b268e897f"
    )


def test_snapshot_selection_uses_full_recipe_and_budget_identity() -> None:
    selection = c.selection_manifest()
    valuation = next(r for r in c.recipes() if r["construction"] == "dataset_valuation")
    assert c.retains_cell(selection, valuation, 32 * int(valuation["n_players"]))
    assert not c.retains_cell(selection, valuation, 64 * int(valuation["n_players"]))
    assert not c.retains_cell(selection, {**valuation, "input_features": "1"}, 32)
    assert not c.retains_cell(selection, {**valuation, "feature_rule": "all"}, 32)
    assert not c.retains_cell(selection, {**valuation, "new_scientific_option": "x"}, 32)
    assert not c.retains_cell(selection, {**valuation, "id": "unknown"}, 32)
    assert not c.retains_cell(selection, valuation, 33)
    natural = next(r for r in c.recipes() if r["n_players"] == "5")
    sparse = {k: v for k, v in natural.items() if v}
    assert c.retains_cell(selection, sparse, 3)  # ceil(0.5 * 5), not exact ratio equality.
    assert not c.retains_cell(selection, sparse, 2)


def test_changed_fixed_manifest_rejected(tmp_path: Path, approved_constructors: None) -> None:
    path = tmp_path / "changed.csv"
    path.write_text(c.manifest_text().replace("-p4,", "-p3,", 1))
    with pytest.raises(ValueError, match="deterministic"):
        c.build_suite(path)
