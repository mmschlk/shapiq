"""The focused manifest preserves application counts and exact-reference boundaries."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from shapiq_benchmark import focused, runner
from shapiq_benchmark.families import _CoalitionRefit, _retraining_game, profile_compatibility
from shapiq_benchmark.models import PreparedModel
from shapiq_games.benchmark.feature_selection.base import FeatureSelection

MANIFEST = Path(__file__).resolve().parents[2] / "benchmark/suites/focused.csv"


@pytest.fixture
def approved_constructors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate manifest tests from whether the estimator PR has been integrated."""

    def constructor(*, ridge: float, low_budget_equal_allocation: bool) -> None:
        pass

    for name in focused.METHOD_PARAMETERS:
        monkeypatch.setitem(runner.METHODS, name, constructor)


def test_focused_design_and_reference_capabilities(approved_constructors: None) -> None:
    """448 intended instances retain equal application weights and qualified adapters."""
    suite = focused.build_suite(MANIFEST)
    recipes = suite["focused_design"]["recipes"]
    assert Counter(row["application"] for row in recipes) == {
        "local": 32,
        "data": 40,
        "features": 40,
    }
    assert len(recipes) * len(suite["game_seeds"]) == 448
    assert suite["relative_budgets"] == [0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    assert runner.cell_timeout_policy(suite) == {
        "ordinary_seconds": 30,
        "extended_seconds": 120,
        "min_players": 128,
        "min_relative_budget": 32,
    }
    assert all(spec["device"] == "cpu" for spec in suite["families"] if "device" in spec)
    assert len(suite["families"]) == 79
    for recipe in recipes:
        if recipe["reference"] == "enumeration":
            continue
        targets = {
            (spec["index"], spec["order"])
            for spec in suite["games"]
            if spec["basecase_id"] == recipe["id"]
        }
        if recipe["reference"] == "neighbor_solver":
            assert targets == {("SV", 1)}
        elif recipe["construction"] == "pathdependent_tree":
            assert targets == {("SV", 1), ("k-SII", 2), ("SII", 2)}
        else:
            assert targets == {(target["index"], target["order"]) for target in focused.TARGETS}
    assert suite["method_parameters"]["LeverageSHAP"] == {
        "ridge": 0.001,
        "low_budget_equal_allocation": True,
    }
    assert {
        spec["basecase_id"] for spec in suite["games"] if spec.get("feature_rule") == "nested"
    } == {"local-25", "local-26", "local-27", "local-29", "local-30", "local-31"}


def _write_manifest(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_original_manifest_and_execution_specs_are_unchanged(
    tmp_path: Path, approved_constructors: None
) -> None:
    """Golden hashes recorded before expansion protect all original recipes and suite settings."""
    suite = focused.build_suite(MANIFEST)
    legacy_recipes = suite["focused_design"]["recipes"][:96]
    assert _canonical_hash(legacy_recipes) == (
        "5ed3a49ab127677e3c914c06c3878de6c6a8b61c80379d71b2bbdefd8d902820"
    )
    assert _canonical_hash({"families": suite["families"][:63], "games": suite["games"]}) == (
        "204061aeb49571cc60fe3e39015dba700c4cf35807d8fd672a9167b6005aa4a7"
    )
    legacy_path = tmp_path / "legacy.csv"
    _write_manifest(legacy_path, legacy_recipes)
    legacy = focused.build_suite(legacy_path)
    assert legacy["name"] == "focused-384-v1"
    assert _canonical_hash(legacy) == (
        "816b96d9e805690a8075b722690883c2a057f27b5cf7d3246dfa691e25856dbb"
    )


def test_added_retraining_pairs_keep_input_and_player_dimensions_separate(
    approved_constructors: None,
) -> None:
    """The four added dataset/model pairs vary twelve/fourteen players with explicit inputs."""
    suite = focused.build_suite(MANIFEST)
    expected = [
        ("breast_cancer", "random_forest", 24),
        ("digits", "linear", 64),
        ("tabarena_miami_housing", "lightgbm", 15),
        ("tabarena_superconductivity", "linear", 64),
    ]
    specs = {spec["id"]: spec for spec in suite["families"]}
    for offset, (dataset, model, inputs) in enumerate(expected):
        for shift, players in enumerate((12, 14)):
            number = 33 + 2 * offset + shift
            for application in ("data", "features"):
                spec = specs[f"{application}-{number}"]
                assert spec["dataset"] == dataset and spec["model_profile"] == model
                assert spec["n_players"] == players and spec["feature_rule"] == "nested"
                if application == "data":
                    assert spec["family"] == "dataset_valuation"
                    assert spec["input_features"] == inputs
                else:
                    assert spec["family"] == "feature_selection"
                    assert "input_features" not in spec
    assert suite["name"] == "focused-448-v1"
    assert suite["focused_design"]["application_weights"] == dict.fromkeys(
        ("local", "data", "features"), 1 / 3
    )


def test_builder_accepts_a_smaller_reviewed_manifest(
    tmp_path: Path, approved_constructors: None
) -> None:
    """Counts come from the manifest rather than a campaign-specific ninety-six-row gate."""
    with MANIFEST.open() as stream:
        rows = list(csv.DictReader(stream))
    path = tmp_path / "small.csv"
    _write_manifest(path, rows[:1])
    suite = focused.build_suite(path)
    assert len(suite["families"]) == 1 and not suite["games"]
    assert suite["name"] == "focused-4-v1"


@pytest.mark.parametrize(
    "mutation",
    [
        "application",
        "subtype",
        "reference_family",
        "input_family",
        "input_count",
        "rule_family",
        "rule",
        "id",
        "duplicate_id",
        "equivalent_defaults",
        "unknown_column",
        "missing_column",
    ],
)
def test_invalid_manifest_semantics_rejected(
    tmp_path: Path, approved_constructors: None, mutation: str
) -> None:
    """Reject mislabeled applications, ignored options and duplicate scientific identities."""
    with MANIFEST.open() as stream:
        rows = list(csv.DictReader(stream))
    if mutation == "application":
        rows[0]["application"] = "features"
    elif mutation == "subtype":
        rows[32]["subtype"] = "neighbor_examples"
    elif mutation == "reference_family":
        rows[12]["reference"] = "neighbor_solver"
    elif mutation == "input_family":
        rows[0]["input_features"] = "12"
    elif mutation == "input_count":
        rows[96]["input_features"] = "31"
    elif mutation == "rule_family":
        rows[0]["feature_rule"] = "nested"
    elif mutation == "rule":
        rows[96]["feature_rule"] = "continuous"
    elif mutation == "id":
        rows[0]["id"] = "../bad"
    elif mutation == "duplicate_id":
        rows[1]["id"] = rows[0]["id"]
    elif mutation == "equivalent_defaults":
        rows.append(
            {**rows[32], "id": "duplicate-defaults", "input_features": "12", "feature_rule": "all"}
        )
    elif mutation == "unknown_column":
        for row in rows:
            row["typo"] = ""
    else:
        for row in rows:
            del row["construction"]
    path = tmp_path / "invalid.csv"
    _write_manifest(path, rows)
    with pytest.raises(ValueError):
        focused.build_suite(path)


def test_distinct_input_dimensions_are_distinct_recipes(
    tmp_path: Path, approved_constructors: None
) -> None:
    """Equal player counts can legitimately identify different valuation input spaces."""
    with MANIFEST.open() as stream:
        rows = list(csv.DictReader(stream))
    original = rows[96]
    narrow = {**original, "id": "narrow", "input_features": "12"}
    path = tmp_path / "dimensions.csv"
    _write_manifest(path, [original, narrow])
    suite = focused.build_suite(path)
    assert [spec["input_features"] for spec in suite["families"]] == [24, 12]


def test_old_estimator_source_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The new suite must not silently execute historical low-budget behavior."""

    def legacy_constructor() -> None:
        pass

    monkeypatch.setitem(runner.METHODS, "LeverageSHAP", legacy_constructor)
    with pytest.raises(ValueError, match="constructor parameters for LeverageSHAP"):
        focused.build_suite(MANIFEST)


@pytest.mark.parametrize("mutation", ["duplicate", "enumeration", "feature_count"])
def test_invalid_recipe_rejected(
    tmp_path: Path, approved_constructors: None, mutation: str
) -> None:
    """Reject duplicated games and impossible reference requests before model preparation."""
    with MANIFEST.open() as stream:
        rows = list(csv.DictReader(stream))
    if mutation == "duplicate":
        rows[1] = {**rows[0], "id": rows[1]["id"]}
    elif mutation == "enumeration":
        rows[0]["n_players"] = "32"
    else:
        rows[0]["n_players"] = "16"
    path = tmp_path / "invalid.csv"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        focused.build_suite(path)


@pytest.mark.parametrize("classification", [True, False])
def test_svm_feature_coalitions_match_direct_refits(*, classification: bool) -> None:
    """New SVC/SVR eligibility preserves shipped feature-selection utility and scaling."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 3))
    y = (x[:, 0] > 0).astype(int) if classification else x[:, 0] * x[:, 1]
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", SVC(C=2.0) if classification else SVR(C=2.0)),
        ]
    )
    prepared = PreparedModel(
        model.fit(x[:24], y[:24]),
        x[:24],
        y[:24],
        x[24:32],
        y[24:32],
        x[32:],
        y[32:],
        {
            "task": "classification" if classification else "regression",
            "model_profile": "rbf_svm",
            "model_parameters": {"C": 2.0},
        },
    )
    game, _ = _retraining_game(FeatureSelection, "feature_selection", prepared, 3, 0)
    masks = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1]], dtype=bool)
    expected = [0.0]
    for mask in masks[1:]:
        fitted = clone(model).fit(prepared.x_train[:, mask], prepared.y_train)
        predicted = fitted.predict(prepared.x_test[:, mask])
        expected.append(
            accuracy_score(prepared.y_test, predicted)
            if classification
            else -mean_squared_error(prepared.y_test, predicted)
        )
    np.testing.assert_allclose(game(masks), expected)
    np.testing.assert_allclose(game(masks[::-1]), expected[::-1])
    assert profile_compatibility("feature_selection", "ionosphere", "rbf_svm") is None
    assert profile_compatibility("dataset_valuation", "ionosphere", "rbf_svm") is not None
    if classification:
        refit = _CoalitionRefit(prepared)
        refit.fit(prepared.x_train[:2], np.array([5, 5]))
        np.testing.assert_array_equal(refit.predict(prepared.x_test), np.full(8, 5))
