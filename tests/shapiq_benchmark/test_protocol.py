"""Compact recipes retain explicit model and exact-reference boundaries."""

from __future__ import annotations

import pytest

from shapiq_benchmark.focused import TARGETS
from shapiq_benchmark.protocol import CONSTRUCTIONS, adapter_reason, recipe_spec


@pytest.mark.parametrize(
    ("family", "model", "indices"),
    [
        ("pathdependent_tree", "random_forest", {"SV", "SII", "k-SII"}),
        ("interventional_tree", "random_forest", {"SV", "SII", "k-SII", "STII", "FSII", "FBII"}),
        ("knn", "knn", {"SV"}),
        ("tnn", "tnn", {"SV"}),
    ],
)
def test_structured_target_boundaries(family, model, indices):
    row = {
        "id": "test",
        "family": family,
        "model_profile": model,
        "dataset": "breast_cancer",
        "n_players": 32,
        "reference": "structured",
    }
    assert adapter_reason(family, row["dataset"], model, structured=True) is None
    specs = recipe_spec(row, TARGETS)
    assert {spec["index"] for spec in specs} == indices
    assert len({spec["id"] for spec in specs}) == len(specs)
    assert all(spec["n_players"] == 32 for spec in specs)
    if family in {"knn", "tnn"}:
        assert specs[0]["row_selection"] == "stratified"
        assert "model_profile" not in specs[0]


@pytest.mark.parametrize(
    ("family", "dataset", "model", "structured"),
    [
        ("knn", "diabetes", "knn", True),
        ("local_baseline", "breast_cancer", "linear", True),
        ("pathdependent_tree", "breast_cancer", "linear", True),
        ("local_baseline", "breast_cancer", "xgboost", False),
        ("image", "breast_cancer", "linear", False),
    ],
)
def test_unsupported_pairs_rejected(family, dataset, model, structured):
    assert adapter_reason(family, dataset, model, structured=structured)


def test_enumerated_recipes_preserve_model_identity():
    for family in ("local_baseline", "feature_selection", "dataset_valuation"):
        assert CONSTRUCTIONS[family]["models"] == ["linear", "random_forest"]
        row = {
            "id": "test",
            "family": family,
            "model_profile": "linear",
            "dataset": "diabetes",
            "n_players": 4,
            "reference": "enumeration",
        }
        assert recipe_spec(row, TARGETS) == [
            {
                "id": "test",
                "family": family,
                "model_profile": "linear",
                "dataset": "diabetes",
                "n_players": 4,
            }
        ]
