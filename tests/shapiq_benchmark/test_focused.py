"""Validate compact manifests without downloading data or fitting models."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from shapiq_benchmark import (
    campaign_runner as runner,
    focused,
)

MANIFEST = Path(__file__).resolve().parents[2] / "benchmark/suites/comprehensive.csv"


@pytest.fixture
def approved_constructors(monkeypatch):
    def constructor(*, ridge: float, low_budget_equal_allocation: bool) -> None:
        pass

    for name in focused.METHOD_PARAMETERS:
        monkeypatch.setitem(runner.METHODS, name, constructor)


@pytest.fixture
def rows():
    with MANIFEST.open() as stream:
        return list(csv.DictReader(stream))


def write_manifest(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_compact_design_and_reference_capabilities(approved_constructors):
    suite = focused.build_suite(MANIFEST)
    recipes = suite["focused_design"]["recipes"]
    assert len(recipes) == 512
    assert len(suite["families"]) == 389
    assert len(suite["game_seeds"]) * len(recipes) == 2048
    assert suite["relative_budgets"] == [0.5, 1, 2, 4, 8, 16, 32]
    assert suite["seeds"] == [0, 1, 2]
    assert "cell_timeout_policy" not in suite  # Execution limits come from settings.json.
    for recipe in recipes:
        if recipe["reference"] == "enumeration":
            continue
        indices = {spec["index"] for spec in suite["games"] if spec["basecase_id"] == recipe["id"]}
        expected = (
            {"SV"}
            if recipe["reference"] == "neighbor_solver"
            else {"SV", "SII", "k-SII"}
            if recipe["construction"] == "pathdependent_tree"
            else {target["index"] for target in focused.TARGETS}
        )
        assert indices == expected
    assert suite["method_parameters"] == focused.METHOD_PARAMETERS


def test_builder_accepts_a_smaller_manifest(tmp_path, rows, approved_constructors):
    suite = focused.build_suite(write_manifest(tmp_path / "small.csv", rows[:1]))
    assert len(suite["families"]) == 1 and not suite["games"]
    assert suite["name"] == "focused-4-v1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("application", "features"),
        ("subtype", "neighbor_examples"),
        ("reference", "neighbor_solver"),
        ("input_features", "12"),
        ("feature_rule", "continuous"),
        ("id", "../bad"),
        ("n_players", "16"),
        ("model", "xgboost"),
        ("construction", "local_marginal"),
    ],
)
def test_invalid_recipe_rejected(tmp_path, rows, approved_constructors, field, value):
    row = {**rows[0], field: value}
    with pytest.raises(ValueError):
        focused.build_suite(write_manifest(tmp_path / "invalid.csv", [row]))


@pytest.mark.parametrize("mutation", ["duplicate_id", "equivalent_recipe", "unknown", "missing"])
def test_invalid_manifest_rejected(tmp_path, rows, approved_constructors, mutation):
    subset = [rows[0].copy(), rows[1].copy()]
    if mutation == "duplicate_id":
        subset[1]["id"] = subset[0]["id"]
    elif mutation == "equivalent_recipe":
        subset[1] = {**subset[0], "id": "other-id"}
    elif mutation == "unknown":
        for row in subset:
            row["typo"] = ""
    else:
        for row in subset:
            del row["construction"]
    with pytest.raises(ValueError):
        focused.build_suite(write_manifest(tmp_path / "invalid.csv", subset))


def test_valuation_features_are_independent_of_players(tmp_path, rows, approved_constructors):
    row = next(
        row
        for row in rows
        if row["construction"] == "dataset_valuation" and row["dataset"] == "breast_cancer"
    )
    narrow = {**row, "id": "narrow", "input_features": "4"}
    suite = focused.build_suite(write_manifest(tmp_path / "inputs.csv", [row, narrow]))
    assert [spec["input_features"] for spec in suite["families"]] == [12, 4]
    assert len({spec["n_players"] for spec in suite["families"]}) == 1


def test_neighbor_pool_rules(tmp_path, rows, approved_constructors):
    row = next(row for row in rows if row["construction"] == "knn")
    invalid = {**row, "training_rows": str(int(row["n_players"]) - 1)}
    with pytest.raises(ValueError, match="neighbor fitting pool"):
        focused.build_suite(write_manifest(tmp_path / "neighbor.csv", [invalid]))


def test_manifest_generation_is_portable_but_execution_remains_strict(tmp_path, rows, monkeypatch):
    def legacy_constructor() -> None:
        pass

    monkeypatch.setitem(runner.METHODS, "LeverageSHAP", legacy_constructor)
    suite = focused.build_suite(write_manifest(tmp_path / "legacy.csv", rows[:1]))
    with pytest.raises(ValueError, match="constructor parameters for LeverageSHAP"):
        runner.validate_suite(suite)
