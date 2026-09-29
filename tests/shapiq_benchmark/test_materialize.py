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
