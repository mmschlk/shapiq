"""Campaign limits apply across batches before any scheduler submission."""

from __future__ import annotations

from shapiq_benchmark.planning import select_core


def test_caps_and_construction_balance():
    suite = {
        "targets": [{"index": "SV", "order": 1}],
        "game_seeds": [0, 1, 2, 3],
        "relative_budgets": [0.5, 1, 2],
        "seeds": [0],
        "methods": ["KernelSHAP"],
        "core_limits": {"recipes": 2},
    }
    entries = [
        ("families", {"id": str(i), "family": name, "n_players": n})
        for i, (name, n) in enumerate([("a", 11), ("a", 12), ("b", 11), ("c", 20)])
    ]
    selected, decision = select_core(entries, suite, {"KernelSHAP": {"indices": ["SV"]}})
    assert [spec["family"] for _, spec in selected] == ["a", "b"]
    assert decision["upper_bounds"]["supported_cells"] == 24
    assert len(decision["deferred"]) == 2
    assert all(decision["upper_bounds"][key] <= value for key, value in decision["limits"].items())


def test_diverse_high_dimension_core_is_deterministic():
    """A tight core cannot silently become only the first dataset, model and d."""
    suite = {
        "targets": [{"index": "SV", "order": 1}],
        "game_seeds": [0, 1, 2, 3],
        "relative_budgets": [0.5, 1, 2],
        "seeds": [0],
        "methods": ["KernelSHAP"],
        "core_limits": {"recipes": 4},
    }
    entries = [
        (
            "families",
            {
                "id": f"{family}-{dataset}-{model}-{n}",
                "family": family,
                "dataset": dataset,
                "model_profile": model,
                "n_players": n,
            },
        )
        for family in ("local", "global", "valuation", "ensemble")
        for dataset in ("adult", "mushroom", "wine")
        for model in ("linear", "random_forest", "xgboost")
        for n in (16, 20)
    ]
    catalog = {"KernelSHAP": {"indices": ["SV"]}}
    selected, decision = select_core(entries, suite, catalog)
    assert select_core(list(reversed(entries)), suite, catalog) == (selected, decision)
    assert len({spec["family"] for _, spec in selected}) == 4
    assert len({spec["dataset"] for _, spec in selected}) == 3
    assert len({spec["model_profile"] for _, spec in selected}) == 3
    assert {spec["n_players"] for _, spec in selected} == {16, 20}
    assert selected[0][1]["model_profile"] == "random_forest"
    assert all(decision["upper_bounds"][key] <= value for key, value in decision["limits"].items())


def test_oversized_recipe_does_not_hide_a_smaller_alternative():
    """Balance preference never spends a cap or strands a construction unnecessarily."""
    suite = {
        "targets": [{"index": "SV", "order": 1}],
        "game_seeds": [0, 1, 2, 3],
        "relative_budgets": [1],
        "seeds": [0],
        "methods": ["KernelSHAP"],
        "core_limits": {"coalition_values": 4 * 2**16},
    }
    entries = [
        (
            "families",
            {
                "id": "preferred-too-large",
                "family": "local",
                "n_players": 20,
                "model_profile": "random_forest",
            },
        ),
        ("families", {"id": "fits", "family": "local", "n_players": 16, "model_profile": "linear"}),
    ]
    selected, decision = select_core(entries, suite, {"KernelSHAP": {"indices": ["SV"]}})
    assert selected == [entries[1]]
    assert decision["deferred"][0]["id"] == "preferred-too-large"


def test_structured_core_starts_with_sv_and_rotates_interaction_targets():
    """Dataset/model diversity must not leave either tree construction without SV."""
    suite = {
        "targets": [],
        "game_seeds": [0, 1, 2, 3],
        "relative_budgets": [1],
        "seeds": [0],
        "methods": ["test"],
        "core_limits": {"recipes": 6},
    }
    entries = [
        (
            "games",
            {
                "id": f"{family}-{dataset}-{model}-{index}",
                "oracle": family,
                "dataset": dataset,
                "model_profile": model,
                "n_players": 24,
                "index": index,
                "order": 1 if index == "SV" else 2,
            },
        )
        for family in ("tree", "pathdependent_tree")
        for dataset in ("adult", "mushroom")
        for model in ("random_forest", "xgboost")
        for index in ("SV", "SII", "k-SII", "FBII")
        if family == "tree" or index != "FBII"
    ]
    catalog = {"test": {"indices": ["SV", "SII", "k-SII", "FBII"]}}
    selected, decision = select_core(entries, suite, catalog)
    assert select_core(list(reversed(entries)), suite, catalog) == (selected, decision)
    assert [spec["index"] for _, spec in selected[:2]] == ["SV", "SV"]
    for family in ("tree", "pathdependent_tree"):
        targets = [spec["index"] for _, spec in selected if spec["oracle"] == family]
        assert targets[0] == "SV"
        assert len(set(targets)) == 3
    assert len({spec["dataset"] for _, spec in selected}) == 2
    assert len({spec["model_profile"] for _, spec in selected}) == 2
