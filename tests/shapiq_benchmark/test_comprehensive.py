"""Bounded checks of the expanded manifest; no dataset downloads or model fits."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from shapiq_benchmark import (
    comprehensive as c,
    focused,
    runner,
)
from shapiq_benchmark.datasets import DATASETS

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
    assert len(rows) == len({r["id"] for r in rows}) == 1270
    assert Counter(r["reference"] for r in rows) == {
        "enumeration": 782,
        "tree_solver": 360,
        "neighbor_solver": 128,
    }
    assert not {r["id"] for r in rows} & {
        r["id"] for r in csv.DictReader((ROOT / "benchmark/suites/focused.csv").open())
    }


def test_dimensions_are_independent_and_nested() -> None:
    rows = c.recipes()
    assert all(r["feature_rule"] == "nested" for r in rows)
    for dataset in c.CLASSIFICATION + c.REGRESSION:
        width = DATASETS[dataset]["n_features"]
        core = [r for r in rows if r["dataset"] == dataset and r["model"] == "linear"]
        for family in ("local_baseline", "feature_selection"):
            actual = {int(r["n_players"]) for r in core if r["construction"] == family}
            assert actual == {n for n in c.GENERIC_COUNTS if n <= width} | (
                {width} if width < 16 else set()
            )
        assert {
            int(r["n_players"])
            for r in core
            if r["construction"] == "dataset_valuation"
            and int(r["input_features"]) == min(12, width)
        } == set(c.GENERIC_COUNTS)
    for dataset in c.CONTRAST_DATASETS:
        width = DATASETS[dataset]["n_features"]
        widths = {
            int(r["input_features"])
            for r in rows
            if r["dataset"] == dataset
            and r["model"] == "linear"
            and r["construction"] == "dataset_valuation"
            and r["n_players"] == "12"
        }
        assert widths == {n for n in c.INPUT_COUNTS if n <= width} | {width, min(12, width)}
    assert not any(
        r["model"] == "rbf_svm" and r["construction"] == "dataset_valuation" for r in rows
    )


def test_neighbor_rows_not_feature_counts_bound_players() -> None:
    rows = [r for r in c.recipes() if r["reference"] == "neighbor_solver"]
    expected_max = {"wine": 128, "ionosphere": 256, "breast_cancer": 256, "digits": 1024}
    for dataset, maximum in expected_max.items():
        assert max(int(r["n_players"]) for r in rows if r["dataset"] == dataset) == maximum
    assert all(
        r["training_rows"] == "5000" and r["row_selection"] == "nested_stratified" for r in rows
    )
    assert all(r["dataset"] in c.CLASSIFICATION for r in rows)


def test_suite_preserves_settings_and_reference_targets(approved_constructors: None) -> None:
    suite = c.build_suite(MANIFEST)
    assert suite["seeds"] == [0, 1, 2] and suite["game_seeds"] == [0, 1, 2, 3]
    assert len(suite["methods"]) == 22
    assert suite["method_parameters"] == focused.METHOD_PARAMETERS
    assert suite["relative_budgets"] == [0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    assert suite["min_players"] == 4
    assert len(suite["families"]) == 782
    assert len(suite["games"]) == 1748
    for game in suite["games"]:
        if game["oracle"] in {"knn", "tnn"}:
            assert (game["index"], game["order"]) == ("SV", 1)
            assert game["training_rows"] == 5000
        elif game["oracle"] == "pathdependent_tree":
            assert game["index"] in {"SV", "k-SII", "SII"}
    assert c.summary(suite)["intended_targets"] == 25760


def test_changed_fixed_manifest_rejected(tmp_path: Path, approved_constructors: None) -> None:
    path = tmp_path / "changed.csv"
    path.write_text(c.manifest_text().replace("-p4,", "-p3,", 1))
    with pytest.raises(ValueError, match="deterministic"):
        c.build_suite(path)


def test_legacy_focused_defaults_unchanged(approved_constructors: None) -> None:
    suite = focused.build_suite(ROOT / "benchmark/suites/focused.csv")
    assert suite["min_players"] == 12 and suite["seeds"] == [0]
    assert len(suite["focused_design"]["recipes"]) == 112
    recipes = suite["focused_design"]["recipes"][:96]
    # Original rows remain unchanged even though new optional columns are supported.
    assert (
        hashlib.sha256(
            json.dumps(recipes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == "5ed3a49ab127677e3c914c06c3878de6c6a8b61c80379d71b2bbdefd8d902820"
    )
