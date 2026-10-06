"""The focused manifest preserves application counts and exact-reference boundaries."""

from __future__ import annotations

import csv
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
    """384 instances are recipe/seed pairs, not the number of target definitions."""
    suite = focused.build_suite(MANIFEST)
    recipes = suite["focused_design"]["recipes"]
    assert Counter(row["application"] for row in recipes) == {
        "local": 32,
        "data": 32,
        "features": 32,
    }
    assert len(recipes) * len(suite["game_seeds"]) == 384
    assert suite["relative_budgets"] == [0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    assert runner.cell_timeout_policy(suite) == {
        "ordinary_seconds": 30,
        "extended_seconds": 120,
        "min_players": 128,
        "min_relative_budget": 32,
    }
    assert all(spec["device"] == "cpu" for spec in suite["families"] if "device" in spec)
    assert len(suite["families"]) == 63
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
