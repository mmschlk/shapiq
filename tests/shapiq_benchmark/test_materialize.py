"""Family coverage and relative-grid snapshot semantics."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from shapiq_benchmark.materialize import prepare_families
from shapiq_benchmark.prepare import prepare
from shapiq_benchmark.runner import load_snapshot

if TYPE_CHECKING:
    from pathlib import Path


def test_relative_family_snapshot_keeps_real_and_diagnostic_metadata(tmp_path: Path) -> None:
    """Frozen values, exact truths, and per-game grids are authenticated together."""
    suite = {
        "name": "test",
        "families": ["local_marginal", "dummy"],
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
    assert {g["metadata"]["synthetic"] for g in snapshot["games"]} == {False, True}
    assert load_snapshot(tmp_path / "snapshot")[0] == snapshot


def test_preparation_failure_preserves_other_families(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing optional models are visible without replacing their payoff."""
    from shapiq_benchmark import materialize

    original = materialize.make_family

    def factory(name: str) -> tuple:
        if name == "local_marginal":
            message = "/private/model is unavailable"
            raise ImportError(message)
        return original(name)

    monkeypatch.setattr(materialize, "make_family", factory)
    games, coverage = prepare_families(
        ["local_marginal", "dummy"], [{"index": "SV", "order": 1}], tmp_path
    )
    assert len(games) == 1
    assert [row["status"] for row in coverage] == ["unavailable", "measured"]
    assert "/private" not in json.dumps(coverage)


def test_bad_targets_rejected_before_constructing_games(tmp_path: Path) -> None:
    """A target typo cannot silently turn the whole library into unavailable games."""
    with pytest.raises(ValueError, match="Targets"):
        prepare_families(["dummy"], [{"index": "SV", "order": 2}], tmp_path)
    assert not list(tmp_path.iterdir())


def test_image_adapter_preserves_the_active_game() -> None:
    """Removing legacy reported dummy players leaves all genuine coalition utilities intact."""
    from shapiq_benchmark.media import ActiveImage

    class Original:
        n_players = 3

        def __call__(self, rows: np.ndarray) -> np.ndarray:
            return rows[:, 0] + 2 * rows[:, 2]

    active = ActiveImage(Original(), [0, 2])
    np.testing.assert_equal(active(np.array([[0, 0], [1, 0], [0, 1], [1, 1]])), [0, 1, 2, 3])


def test_four_constructions_have_shared_strata_and_one_estimator_seed(tmp_path: Path) -> None:
    """Instances share panel strata but have distinct artifacts and independent cluster IDs."""
    suite = {
        "families": ["local_baseline_forest", "dummy"],
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
        "uncertainty",
        {"id": "uncertainty-wine-6", "family": "uncertainty", "dataset": "wine", "n_players": 6},
    ]
    targets = [{"index": "SV", "order": 1}, {"index": "SII", "order": 2}]
    games, coverage = prepare_families(specs, targets, tmp_path, instance_seed=1)
    assert len(games) == 4 and all(entry["status"] == "measured" for entry in coverage)
    assert {g["metadata"]["game_kind"] for g in games} == {"uncertainty"}
    assert {g["metadata"]["case_id"] for g in games} == {"uncertainty", "uncertainty-wine-6"}
    assert {g["n_players"] for g in games} == {4, 6}
    assert len({g["stratum"] for g in games}) == len({g["artifact"] for g in games}) == 2
    for game in games:
        values = np.load(tmp_path / game["artifact"])["values"]
        assert len(values) == 2 ** game["n_players"]
        assert game["truth"]["baseline"] == pytest.approx(values[0])


@pytest.mark.parametrize(
    "specs",
    [
        ["dummy", {"id": "dummy", "family": "dummy"}],
        [{"id": "../escape", "family": "dummy"}],
        [{"id": "typo", "family": "dummy", "players": 4}],
        [{"id": "too-large", "family": "dummy", "n_players": 13}],
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
                "families": ["dummy"],
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
