"""Family coverage and relative-grid snapshot semantics."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from shapiq_benchmark.campaign_runner import identity, load_snapshot
from shapiq_benchmark.materialize import prepare_families
from shapiq_benchmark.prepare import prepare, write_snapshot

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def compact_linear_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise exact table serialization cheaply with the compact linear profile."""
    from shapiq_benchmark import materialize

    original = materialize.make_family

    def make(name: str, **kwargs: object) -> tuple:
        return original(name, **{"model_profile": "linear", **kwargs})

    monkeypatch.setattr(materialize, "make_family", make)


def test_relative_family_snapshot_keeps_real_metadata(tmp_path: Path) -> None:
    """Frozen values, exact truths, and per-game grids are authenticated together."""
    suite = {
        "name": "test",
        "families": ["feature_selection", "local_baseline"],
        "targets": [{"index": "SV", "order": 1}],
        "methods": ["KernelSHAP"],
        "relative_budgets": [2, 8],
        "seeds": [0],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    snapshot = prepare(path, tmp_path / "snapshot")
    assert len(snapshot["games"]) == 2
    assert snapshot["suite"]["budgets"] == [16, 64]
    assert all(grid == [16, 64] for grid in snapshot["suite"]["budgets_by_game"].values())
    assert {g["metadata"]["synthetic"] for g in snapshot["games"]} == {False}
    assert load_snapshot(tmp_path / "snapshot")[0] == snapshot


def test_preparation_failure_preserves_other_families(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing optional models are visible without replacing their payoff."""
    from shapiq_benchmark import materialize

    original = materialize.make_family

    def factory(name: str) -> tuple:
        if name == "feature_selection":
            message = "/private/model is unavailable"
            raise ImportError(message)
        return original(name)

    monkeypatch.setattr(materialize, "make_family", factory)
    games, coverage = prepare_families(
        ["feature_selection", "local_baseline"], [{"index": "SV", "order": 1}], tmp_path
    )
    assert len(games) == 1
    assert [row["status"] for row in coverage] == ["unavailable", "measured"]
    assert "/private" not in json.dumps(coverage)


def test_bad_targets_rejected_before_constructing_games(tmp_path: Path) -> None:
    """A target typo cannot silently turn the whole library into unavailable games."""
    with pytest.raises(ValueError, match="Targets"):
        prepare_families(["local_baseline"], [{"index": "SV", "order": 2}], tmp_path)
    assert not list(tmp_path.iterdir())


def test_four_constructions_have_shared_strata_and_one_estimator_seed(tmp_path: Path) -> None:
    """Instances share panel strata but have distinct artifacts and independent cluster IDs."""
    suite = {
        "families": ["feature_selection", "local_baseline"],
        "targets": [{"index": "SV", "order": 1}, {"index": "k-SII", "order": 2}],
        "methods": ["KernelSHAP"],
        "relative_budgets": [1, 2],
        "game_seeds": [0, 1, 2, 3],
        "seeds": [0],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    snapshot = prepare(path, tmp_path / "snapshot")
    assert len(snapshot["games"]) == 16
    assert len(snapshot["artifacts"]) == 8
    assert snapshot["suite"]["seeds"] == [0]
    for name in suite["families"]:
        games = [g for g in snapshot["games"] if g["metadata"]["case_id"] == name]
        assert len({g["stratum"] for g in games}) == 1
        assert len({g["metadata"]["cluster_id"] for g in games}) == 4
        assert {g["metadata"]["instance_seed"] for g in games} == set(range(4))
        assert len({g["id"] for g in games}) == 8
        assert all("-i" in g["artifact"] for g in games)
    assert load_snapshot(tmp_path / "snapshot")[0] == snapshot


def test_explicit_variants_keep_kind_but_have_independent_panel_settings(tmp_path: Path) -> None:
    """Native and subset dimensions have separate identities without losing family grouping."""
    specs = [
        {"id": "local_baseline", "family": "local_baseline", "n_players": 4},
        {"id": "baseline-wine-6", "family": "local_baseline", "dataset": "wine", "n_players": 6},
    ]
    targets = [{"index": "SV", "order": 1}, {"index": "SII", "order": 2}]
    games, coverage = prepare_families(specs, targets, tmp_path, instance_seed=1)
    assert len(games) == 4 and all(entry["status"] == "measured" for entry in coverage)
    assert {g["metadata"]["game_kind"] for g in games} == {"local_baseline"}
    assert {g["metadata"]["case_id"] for g in games} == {"local_baseline", "baseline-wine-6"}
    assert {g["n_players"] for g in games} == {4, 6}
    assert len({g["stratum"] for g in games}) == len({g["artifact"] for g in games}) == 2
    for game in games:
        values = np.load(tmp_path / game["artifact"])["values"]
        assert len(values) == 2 ** game["n_players"]
        assert game["truth"]["baseline"] == pytest.approx(values[0])


@pytest.mark.parametrize(
    "specs",
    [
        ["local_baseline", {"id": "local_baseline", "family": "local_baseline"}],
        [{"id": "../escape", "family": "local_baseline"}],
        [{"id": "typo", "family": "local_baseline", "players": 4}],
        [{"id": "too-large", "family": "local_baseline", "n_players": 21}],
    ],
)
def test_invalid_family_specs_fail_before_creating_artifacts(tmp_path: Path, specs: list) -> None:
    """Configuration mistakes must not turn into ambiguous missing benchmark cases."""
    with pytest.raises(ValueError):
        prepare_families(specs, [{"index": "SV", "order": 1}], tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("seeds", [[], [0, 0], [-1], [True], [1.5], "0123"])
def test_invalid_game_seeds_rejected(tmp_path: Path, seeds: object) -> None:
    """Invalid construction grids must fail before any family preparation."""
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            {
                "families": ["local_baseline"],
                "targets": [{"index": "SV", "order": 1}],
                "methods": ["KernelSHAP"],
                "relative_budgets": [1],
                "seeds": [0],
                "game_seeds": seeds,
            }
        )
    )
    with pytest.raises(ValueError, match="game_seeds"):
        prepare(path, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


@pytest.mark.parametrize("minimum", [0, -1, True, 11.0, "11"])
def test_invalid_minimum_players_fails_before_preparation(tmp_path: Path, minimum: object) -> None:
    """A requested dimensionality floor must be an unambiguous positive integer."""
    suite = {
        "families": ["local_baseline"],
        "targets": [{"index": "SV", "order": 1}],
        "methods": ["KernelSHAP"],
        "relative_budgets": [1],
        "seeds": [0],
        "min_players": minimum,
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match="min_players"):
        prepare(path, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


def test_minimum_players_rejects_explicit_and_actual_small_games(tmp_path: Path) -> None:
    """Neither a bad config nor a smaller-than-requested factory output may become a snapshot."""
    suite = {
        "families": [{"id": "small", "family": "local_baseline", "n_players": 8}],
        "targets": [{"index": "SV", "order": 1}],
        "methods": ["KernelSHAP"],
        "relative_budgets": [1],
        "seeds": [0],
        "min_players": 11,
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match="below min_players"):
        prepare(path, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()
    with pytest.raises(ValueError, match="Constructed games violate"):
        write_snapshot(suite, [{"n_players": 10}], tmp_path)
    assert not (tmp_path / "snapshot.json").exists()


def test_eleven_player_snapshot_retains_exact_truth_and_relative_grid(tmp_path: Path) -> None:
    """The new dimensionality floor accepts genuine higher-dimensional game construction."""
    suite = {
        "families": [{"id": "dummy-11", "family": "local_baseline", "n_players": 11}],
        "targets": [{"index": "SV", "order": 1}],
        "methods": ["KernelSHAP"],
        "relative_budgets": [0.5, 128],
        "seeds": [0],
        "min_players": 11,
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    snapshot = prepare(path, tmp_path / "snapshot")
    (game,) = snapshot["games"]
    assert game["n_players"] == 11
    assert snapshot["suite"]["budgets_by_game"][game["id"]] == [6, 1408]
    assert len(np.load(tmp_path / "snapshot" / game["artifact"])["values"]) == 2048
    assert load_snapshot(tmp_path / "snapshot")[0] == snapshot
    snapshot["suite"]["min_players"] = 12
    snapshot["snapshot_id"] = identity(
        {key: value for key, value in snapshot.items() if key != "snapshot_id"}
    )
    (tmp_path / "snapshot" / "snapshot.json").write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="min_players"):
        load_snapshot(tmp_path / "snapshot")
