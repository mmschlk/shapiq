"""Fixed-panel aggregation must resist imbalance, missingness, and result ordering."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from shapiq_benchmark.summary import comparisons, summarize, weighted_median, weights_for


def fixture_data(games: list[dict], methods: dict[str, list[float]]) -> dict:
    """Build one complete outcome per method, game, and seed."""
    return {
        "games": [
            {"id": str(i), "index": "SV", "order": 1, **game} for i, game in enumerate(games)
        ],
        "methods": {name: {} for name in methods},
        "suite": {"budgets": [16], "seeds": [0, 1]},
        "records": [
            {
                "game_id": str(i),
                "method": method,
                "budget": 16,
                "seed": seed,
                "status": "ok",
                "nmse": value,
            }
            for method, values in methods.items()
            for i, value in enumerate(values)
            for seed in (0, 1)
        ],
    }


def overall(data: dict, *, draws: int = 20) -> dict:
    """Select the full-family preset."""
    return next(
        panel for panel in summarize(data, bootstrap_draws=draws) if panel["family"] is None
    )


def test_equal_family_weight_and_midpoint_weighted_median() -> None:
    """Adding more points to one family must not let it dominate the overall score."""
    data = fixture_data(
        [{"family": "a", "stratum": "a"}, *[{"family": "b", "stratum": "b"}] * 3],
        {"KernelSHAP": [0, 10, 10, 10]},
    )
    row = overall(data)["rows"][0]
    assert row["mean"] == pytest.approx(5)
    assert row["median"] == 5
    assert weighted_median(np.array([1, 9]), np.array([0.5, 0.5])) == 5


def test_equal_stratum_weight() -> None:
    """Many instances within a stratum do not increase that stratum's influence."""
    data = fixture_data(
        [{"family": "a", "stratum": "one"}, *[{"family": "a", "stratum": "two"}] * 3],
        {"KernelSHAP": [2, 6, 6, 6]},
    )
    assert overall(data)["rows"][0]["mean"] == pytest.approx(4)


def test_partial_method_enters_rankings_without_changing_history() -> None:
    """Successful cells get scores and matched Elo; history keeps the complete panel."""
    data = fixture_data([{"family": "a", "stratum": "one"}], {"KernelSHAP": [0.2], "SVARM": [0.1]})
    before = overall(data)
    data["methods"]["SHAPIQ"] = {}
    data["records"].append(
        {"game_id": "0", "method": "SHAPIQ", "budget": 16, "seed": 0, "status": "ok", "nmse": 0}
    )
    after = overall(data)
    assert before["history"] == after["history"]
    rows = {row["method"]: row for row in after["rows"]}
    partial = rows["SHAPIQ"]
    assert partial["eligible"] and not partial["complete"]
    assert partial["valid"] == 1 and partial["planned"] == 2
    assert partial["mean"] == partial["median"] == 0
    assert partial["coverage_weight"] == 0.5
    assert partial["elo"] > rows["SVARM"]["elo"] > rows["KernelSHAP"]["elo"]
    assert partial["ci"] is None


def test_ties_and_input_order_do_not_change_elo() -> None:
    """Batch ratings remain symmetric and independent of insertion or match order."""
    data = fixture_data([{"family": "a", "stratum": "one"}], {"KernelSHAP": [1], "SVARM": [1.005]})
    result = overall(data)
    assert [row["elo"] for row in result["rows"]] == pytest.approx([1000, 1000])
    matches, _ = comparisons(
        np.array([[1, 1], [1.005, 1.005]]), np.array([0.5, 0.5]), ["KernelSHAP", "SVARM"]
    )
    assert matches[0]["ties"] == 2
    assert "matches" not in result
    data["records"].reverse()
    data["methods"] = dict(reversed(data["methods"].items()))
    assert overall(data) == result


def test_strict_five_method_order_converges() -> None:
    """A real-panel ordering must not fail when line search reaches roundoff."""
    values = np.array([[2, 2], [1, 1], [3, 3], [5, 5], [4, 4]])
    weights, methods = np.array([0.5, 0.5]), list("abcde")
    matches, ratings = comparisons(values, weights, methods)
    ratings = np.array(ratings)
    assert np.all(np.isfinite(ratings))
    assert ratings.mean() == pytest.approx(1000)
    assert np.argsort(-ratings).tolist() == [1, 0, 2, 4, 3]

    # Check stationarity independently from the optimizer's success flag.
    skills = (ratings - 1000) * np.log(10) / 400
    gradient = 0.001 * skills
    for match in matches:
        a, b = methods.index(match["a"]), methods.index(match["b"])
        residual = 1 / (1 + np.exp(skills[b] - skills[a])) - match["score_a"]
        gradient[a] += residual
        gradient[b] -= residual
    assert np.max(np.abs(gradient)) < 1e-7

    permutation = np.array([4, 2, 0, 3, 1])
    _, reordered = comparisons(values[permutation], weights, [methods[i] for i in permutation])
    np.testing.assert_allclose(
        np.array(reordered)[np.argsort(permutation)], ratings, atol=2e-5, rtol=0
    )


def test_unknown_dates_do_not_set_frontier() -> None:
    """An excellent but undated estimator stays in accuracy rankings, outside history."""
    data = fixture_data(
        [{"family": "a", "stratum": "one"}], {"KernelSHAP": [1], "SVARM": [0.5], "Undated": [0.01]}
    )
    history = overall(data)["history"]
    assert history["unknown_dates"] == ["Undated"]
    assert [point["value"] for point in history["mean"]] == [1, 0.5]
    assert history["mean"][0]["date"] == "2017-05-22"
    assert history["methods"][0]["url"] == "https://arxiv.org/abs/1705.07874"


def test_zero_truth_is_excluded_for_every_method() -> None:
    """Degenerate games do not become selective omissions or automatic wins."""
    data = fixture_data(
        [{"family": "a", "stratum": "one"}] * 2, {"KernelSHAP": [0, 2], "SVARM": [0, 1]}
    )
    for row in data["records"]:
        if row["game_id"] == "0":
            row.update(nmse=None, zero_truth_energy=True)
    result = overall(data)
    assert result["excluded_zero_energy_games"] == ["0"]
    assert all(row["eligible"] and row["planned"] == 2 for row in result["rows"])


def test_cluster_intervals_require_replication() -> None:
    """Explanation points from one fitted model do not manufacture independent units."""
    games = [{"family": "a", "stratum": "one"}] * 3
    data = fixture_data(games, {"KernelSHAP": [1, 2, 4], "SVARM": [4, 2, 1]})
    assert overall(data)["uncertainty"]["available"] is False
    replicated = copy.deepcopy(data)
    for i, game in enumerate(replicated["games"]):
        game["metadata"] = {"cluster_id": str(i)}
    result = overall(replicated)
    assert result["uncertainty"]["available"] is True
    assert all(row["ci"]["mean"][0] < row["ci"]["mean"][1] for row in result["rows"])
    assert all(row["ci"]["elo"][0] < row["ci"]["elo"][1] for row in result["rows"])


def test_bootstrap_keeps_budget_grid_fixed_and_pairs_methods() -> None:
    """Adding a fixed offset at a second budget shifts every paired mean draw by half."""
    games = [
        {"family": "a", "stratum": "one", "metadata": {"cluster_id": str(i)}} for i in range(3)
    ]
    data = fixture_data(games, {"KernelSHAP": [1, 2, 4], "SVARM": [1, 2, 4]})
    original = overall(data)
    assert original["rows"][0]["ci"] == original["rows"][1]["ci"]
    data["suite"]["budgets"] = [16, 32]
    data["records"] += [{**row, "budget": 32, "nmse": row["nmse"] + 100} for row in data["records"]]
    panel = next(
        item
        for item in summarize(data, bootstrap_draws=20)
        if item["family"] is None and item["budgets"] == [16, 32]
    )
    assert panel["rows"][0]["mean"] == pytest.approx(original["rows"][0]["mean"] + 50)
    assert panel["rows"][0]["ci"]["mean"] == pytest.approx(
        np.array(original["rows"][0]["ci"]["mean"]) + 50
    )


def test_preset_identity_includes_snapshot_and_method_source() -> None:
    """Identical labels cannot give changed benchmark or estimator versions the same ID."""
    data = fixture_data([{"family": "a", "stratum": "one"}], {"KernelSHAP": [1]})
    old = overall(data)["id"]
    data["snapshot_id"] = "new-snapshot"
    newer = overall(data)["id"]
    assert newer != old
    data["methods"]["KernelSHAP"] = {"source_sha256": "changed"}
    assert overall(data)["id"] != newer


def test_unequal_game_budget_counts_preserve_game_weights_and_bootstrap() -> None:
    """Extra planned budgets within one game cannot increase its aggregate weight."""
    games = [
        {"family": "a", "stratum": "one", "metadata": {"cluster_id": str(i)}} for i in range(2)
    ]
    data = fixture_data(games, {"KernelSHAP": [0, 10], "SVARM": [1, 11]})
    original = overall(data)
    data["suite"].update(budgets=[16, 32], budgets_by_game={"0": [16], "1": [16, 32]})
    data["records"] += [{**row, "budget": 32} for row in data["records"] if row["game_id"] == "1"]
    result = overall(data)
    cells, weights = weights_for(data["games"], data["suite"]["budgets_by_game"], [0, 1])
    assert len(cells) == 6
    assert weights.sum() == pytest.approx(1)
    assert weights == pytest.approx([0.25, 0.25, 0.125, 0.125, 0.125, 0.125])
    assert result["rows"][0]["mean"] == pytest.approx(5)
    assert result["rows"][0]["median"] == 5
    assert result["rows"][0]["ci"]["mean"] == pytest.approx(original["rows"][0]["ci"]["mean"])
    assert result["game_budgets"] == {"0": [16], "1": [16, 32]}


def test_relative_presets_keep_missing_cells_and_exact_game_grids() -> None:
    """A ratio is a fixed per-game budget, including cells with no successful run."""
    data = fixture_data(
        [
            {"family": "a", "stratum": "one", "n_players": 8},
            {"family": "b", "stratum": "one", "n_players": 16},
        ],
        {"KernelSHAP": [1, 2]},
    )
    data["suite"].update(
        budgets=[16, 32, 64, 128],
        budgets_by_game={"0": [16, 64], "1": [32, 128]},
        relative_budgets=[2, 8],
    )
    # Game 1 has an available result at 16, but none at its planned B/d=2 budget 32.
    panels = summarize(data, bootstrap_draws=0)
    relative = next(p for p in panels if p["family"] is None and p["relative_budget"] == 2)
    assert relative["game_budgets"] == {"0": [16], "1": [32]}
    assert relative["budgets"] == [16, 32]
    row = relative["rows"][0]
    assert row["planned"] == 4
    assert row["valid"] == 2
    assert row["missing"] == 2
    assert row["eligible"] is True
    assert row["complete"] is False
    assert row["mean"] == 1
    assert all(p["panel"] == "real" for p in panels)
    signatures = [
        tuple((key, tuple(value)) for key, value in p["game_budgets"].items()) for p in panels
    ]
    assert len(signatures) == len(set(signatures))
    assert {tuple(p["game_ids"]) for p in panels} == {("0", "1"), ("0",), ("1",)}


def test_synthetic_games_never_enter_real_panels() -> None:
    """A diagnostic game sharing the target and family remains a separate population."""
    data = fixture_data(
        [
            {"family": "a", "stratum": "one"},
            {"family": "a", "stratum": "one", "metadata": {"synthetic": True}},
        ],
        {"KernelSHAP": [1, 1000]},
    )
    panels = summarize(data, bootstrap_draws=0)
    assert len(panels) == 2
    real = next(panel for panel in panels if panel["panel"] == "real")
    diagnostic = next(panel for panel in panels if panel["panel"] == "diagnostic")
    assert real["game_ids"] == ["0"]
    assert real["rows"][0]["mean"] == 1
    assert diagnostic["game_ids"] == ["1"]
    assert diagnostic["rows"][0]["mean"] == 1000
    assert real["id"] != diagnostic["id"]


def test_partial_cells_renormalize_original_weights() -> None:
    """Observed-cell averaging preserves hierarchical mass, not a raw row average."""
    data = fixture_data(
        [
            {"family": "a", "stratum": "a"},
            {"family": "b", "stratum": "b"},
            {"family": "b", "stratum": "b"},
        ],
        {"Partial": [4, 8, 20], "None": [0, 0, 0]},
    )
    # Weights .25,.25,.125,.125,.125,.125; successes at positions 0,2,3.
    for row in data["records"]:
        if row["method"] == "None" or (row["game_id"], row["seed"]) not in {
            ("0", 0),
            ("1", 0),
            ("1", 1),
        }:
            row["status"] = "failed"
        elif row["game_id"] == "1" and row["seed"] == 1:
            row["nmse"] = 12
    result = overall(data)
    rows = {r["method"]: r for r in result["rows"]}
    assert rows["Partial"]["mean"] == 7
    assert rows["Partial"]["median"] == 6
    assert rows["Partial"]["coverage_weight"] == 0.5
    assert rows["Partial"]["valid"] == 3 and rows["Partial"]["planned"] == 6
    assert rows["None"]["mean"] is None and rows["None"]["median"] is None
    assert rows["None"]["eligible"] is False
    assert all(r["elo"] is None for r in result["rows"])
    assert result["summary_protocol"] == "available-cells-v2"


def test_partial_elo_uses_overlap_mass_and_ignores_nonfinite_cells() -> None:
    """One observed win has less evidence than four, and missing cells are never losses."""
    weights = np.full(4, 0.25)
    complete = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    matches, full = comparisons(complete, weights, ["a", "b"])
    partial = np.array([[0.0, np.nan, np.nan, np.nan], [1.0, 1.0, np.inf, 1.0]])
    matches, ratings = comparisons(partial, weights, ["a", "b"])
    assert matches[0]["n_pairs"] == matches[0]["wins"] == 1
    assert matches[0]["losses"] == 0
    assert matches[0]["observed_weight"] == matches[0]["score_a"] == 0.25
    assert 1000 < ratings[0] < full[0]
    assert ratings[1] < 1000
    skills = (np.array(ratings) - 1000) * np.log(10) / 400
    residual = 0.25 / (1 + np.exp(skills[1] - skills[0])) - 0.25
    np.testing.assert_allclose(0.001 * skills + [residual, -residual], 0, atol=1e-7)
    _, reverse = comparisons(partial[::-1], weights, ["b", "a"])
    np.testing.assert_allclose(reverse[::-1], ratings)


def test_disconnected_elo_is_withheld_and_overlap_chain_is_identified() -> None:
    """L2 alone cannot identify global ranks across disconnected comparison graphs."""
    weights = np.array([0.5, 0.5])
    disconnected = np.array([[1.0, np.nan], [2.0, np.nan], [np.nan, 3.0]])
    matches, ratings = comparisons(disconnected, weights, list("abc"))
    assert len(matches) == 1 and ratings is None
    connected = np.array([[1.0, np.nan], [2.0, 2.0], [np.nan, 3.0]])
    matches, ratings = comparisons(connected, weights, list("abc"))
    assert len(matches) == 2 and ratings[0] > ratings[1] > ratings[2]
    assert np.mean(ratings) == pytest.approx(1000)


def test_partial_pool_keeps_complete_accuracy_intervals_but_withholds_elo_intervals() -> None:
    """A complete-subset Elo interval must not accompany a larger-pool point estimate."""
    data = fixture_data(
        [{"family": "a", "stratum": "one", "metadata": {"cluster_id": str(i)}} for i in range(3)],
        {"KernelSHAP": [1, 2, 4], "SVARM": [4, 2, 1]},
    )
    before = overall(data)
    data["methods"]["SHAPIQ"] = {}
    data["records"].append(
        {"game_id": "0", "method": "SHAPIQ", "budget": 16, "seed": 0, "status": "ok", "nmse": 0.5}
    )
    after = overall(data)
    assert before["history"] == after["history"]
    assert "Elo intervals withheld" in after["uncertainty"]["reason"]
    for row in after["rows"]:
        if row["complete"]:
            previous = next(r for r in before["rows"] if r["method"] == row["method"])
            assert row["ci"]["mean"] == pytest.approx(previous["ci"]["mean"])
            assert row["ci"]["median"] == pytest.approx(previous["ci"]["median"])
            assert row["ci"]["elo"] is None
        else:
            assert row["ci"] is None


@pytest.mark.parametrize(
    "values", [[2.0, 100.0], [0.0] * 99 + [1.0] * 99, [1.0, 4.0, 9.0], [1e308, 1e308]]
)
def test_equal_weight_median_matches_numpy(values: list[float]) -> None:
    """An even number of equally weighted runs cannot select the better half's endpoint."""
    sample = np.asarray(values)
    expected = sample[0] if sample[0] == 1e308 else float(np.median(sample))
    assert weighted_median(sample, np.ones(len(sample))) == expected


def test_weighted_median_skips_zero_mass_and_only_interpolates_half_boundary() -> None:
    """Zero-weight values do not become neighbors; unequal mass crosses without interpolation."""
    assert weighted_median(np.array([2.0, 999.0, 100.0]), np.array([0.5, 0.0, 0.5])) == 51
    assert weighted_median(np.array([2.0, 100.0]), np.array([0.6, 0.4])) == 2
    assert weighted_median(np.array([2.0, 100.0]), np.array([0.4, 0.6])) == 100
