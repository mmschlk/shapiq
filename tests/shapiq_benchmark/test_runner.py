"""Regression tests for benchmark accounting, scoring, and frozen artifacts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from shapiq import InteractionValues
from shapiq_benchmark.prepare import prepare
from shapiq_benchmark.runner import (
    BudgetExceededError,
    CountedGame,
    candidate_factory,
    digest,
    identity,
    load_snapshot,
    run,
    run_one,
    score,
    table_game,
)

if TYPE_CHECKING:
    from pathlib import Path


def game_spec() -> dict:
    """A sparse two-player truth fixture."""
    return {
        "n_players": 2,
        "index": "SV",
        "order": 1,
        "truth": {"coordinates": [[0]], "values": [2.0]},
    }


def estimate(values: dict, **kwargs: object) -> InteractionValues:
    """Construct output in an intentionally arbitrary coordinate ordering."""
    return InteractionValues(values, index="SV", n_players=2, min_order=0, max_order=1, **kwargs)


def test_counted_oracle() -> None:
    """Duplicates and one-dimensional calls count, denied rows are not evaluated."""
    oracle = CountedGame(lambda matrix: matrix.sum(axis=1), 2, 3)
    np.testing.assert_equal(oracle(np.array([1, 0])), [1])
    np.testing.assert_equal(oracle(np.array([[1, 0], [1, 0]])), [1, 1])
    with pytest.raises(BudgetExceededError):
        oracle(np.array([0, 0]))
    assert oracle.queries == 3
    assert oracle.requested == 4
    assert oracle.exceeded
    with pytest.raises(BudgetExceededError):
        oracle(np.empty((0, 2)))


def test_coordinate_alignment_and_zero_energy() -> None:
    """Score absent sparse coefficients and extraneous predictions, excluding baseline."""
    result = score(estimate({(1,): 3.0, (): 100.0, (0,): 2.0}), game_spec())
    assert result["mse"] == 4.5
    assert result["nmse"] == 2.25
    assert score(estimate({(0,): 2.0}), game_spec())["nmse"] == 0
    zero = game_spec()
    zero["truth"]["values"] = [0.0]
    assert score(estimate({(0,): 1.0}), zero)["nmse"] is None
    assert score(estimate({(0,): 1.0}), zero)["mse"] == 0.5


def test_reject_wrong_target_and_coordinates() -> None:
    """Do not report plausible metrics for invalid estimator outputs."""
    output = estimate({(0,): 2.0})
    output.index = "BV"
    with pytest.raises(ValueError, match="wrong player"):
        score(output, game_spec())
    output.index = "SV"
    output.interactions[(2,)] = 1.0
    with pytest.raises(ValueError, match="coordinate"):
        score(output, game_spec())
    output.interactions = {(0,): float("nan")}
    with pytest.raises(ValueError, match="Nonfinite"):
        score(output, game_spec())


def test_artifact_verification_and_failure_persistence(tmp_path: Path) -> None:
    """Tampering fails before execution; caught budget violations remain failures."""
    values = np.array([0.0, 2.0, 0.0, 2.0])
    np.savez(tmp_path / "game.npz", values=values)
    game = {**game_spec(), "id": "test", "artifact": "game.npz"}
    snapshot = {
        "schema_version": 1,
        "provenance": {},
        "games": [game],
        "artifacts": {"game.npz": digest(tmp_path / "game.npz")},
        "suite": {"methods": ["KernelSHAP"], "budgets": [4], "seeds": [0]},
    }
    snapshot["snapshot_id"] = identity(snapshot)
    (tmp_path / "snapshot.json").write_text(json.dumps(snapshot))
    load_snapshot(tmp_path)
    baseline = run(tmp_path, tmp_path / "baseline")
    assert baseline["records"][0]["nmse"] == pytest.approx(0)
    candidate = tmp_path / "candidate.py"
    candidate.write_text("""import numpy as np
from shapiq import InteractionValues
class Candidate:
    def approximate(self, budget, game):
        try:
            game(np.zeros((budget + 1, 2)))
        except RuntimeError:
            pass
        return InteractionValues({(0,): 2.0}, index="SV", n_players=2, min_order=0, max_order=1)
def factory(n, index, order, seed):
    return Candidate()
""")
    result = run(tmp_path, tmp_path / "candidate_results", f"{candidate}:factory")
    assert result["records"][0]["status"] == "failed"
    assert result["records"][0]["queries"] == 0
    assert "BudgetExceeded" in result["records"][0]["error"]
    assert (tmp_path / "candidate_results" / "results.csv").exists()
    np.savez(tmp_path / "game.npz", values=values + 1)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_snapshot(tmp_path)


def test_frozen_preparation_is_repeatable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The recipe is reproducible and oracle values survive batching and ordering."""
    rng = np.random.default_rng(42)
    features = pd.DataFrame(rng.normal(size=(200, 8)), columns=list("abcdefgh"))
    target = features["a"] + 2 * features["b"]
    monkeypatch.setattr(
        "shapiq_benchmark.prepare.load_california_housing", lambda: (features, target)
    )
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {"game": "california_tree", "methods": ["KernelSHAP"], "budgets": [32], "seeds": [0]}
        )
    )
    first = prepare(suite, tmp_path / "first")
    second = prepare(suite, tmp_path / "second")
    assert first["snapshot_id"] == second["snapshot_id"]
    with np.load(tmp_path / "first" / first["games"][0]["artifact"]) as artifact:
        oracle = table_game(artifact["values"], 8)
        coalitions = rng.integers(0, 2, size=(20, 8))
        np.testing.assert_equal(oracle(coalitions)[::-1], oracle(coalitions[::-1]))
        np.testing.assert_equal(
            oracle(coalitions), np.concatenate([oracle(row[None]) for row in coalitions])
        )


def test_error_overflow_rejected() -> None:
    """Large finite coefficients must not leak infinity into result JSON."""
    with pytest.raises((ValueError, OverflowError)):
        score(estimate({(0,): np.float64(1e308)}), game_spec())


def test_candidate_with_dataclass(tmp_path: Path) -> None:
    """Candidate modules support normal Python class decorators."""
    path = tmp_path / "dataclass_candidate.py"
    path.write_text(
        "from __future__ import annotations\nfrom dataclasses import dataclass\n"
        "@dataclass\nclass Candidate:\n    n: int\n"
        "def factory(n, index, order, seed):\n    return Candidate(n)\n"
    )
    _, factory, _ = candidate_factory(f"{path}:factory")
    assert factory(2, "SV", 1, 0).n == 2


def test_numpy_coordinates_are_json_serializable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sparse estimators may return numpy integer keys; result persistence must accept them."""

    class Sparse:
        def approximate(self, budget: int, game: object) -> InteractionValues:
            return estimate({(np.int64(0),): 2.0})

    np.savez(tmp_path / "game.npz", values=np.array([0.0, 2.0, 0.0, 2.0]))
    monkeypatch.setattr("shapiq_benchmark.runner.builtin_factory", lambda *args: Sparse())
    result = run_one({**game_spec(), "artifact": "game.npz"}, tmp_path, "ProxySPEX", 4, 0)
    assert result["status"] == "ok"
    coordinates = json.loads(json.dumps(result))["estimate"]["coordinates"]
    assert [0] in coordinates
    assert all(type(index) is int for coordinate in coordinates for index in coordinate)


def test_global_rng_backends_repeat_across_cells(tmp_path: Path) -> None:
    """Dependencies drawing during import, construction, and execution use the cell seed."""
    np.savez(tmp_path / "game.npz", values=np.array([0.0, 2.0, 0.0, 2.0]))
    path = tmp_path / "global_rng_candidate.py"
    path.write_text("""import random
import numpy as np
from shapiq import InteractionValues
python_import = random.random()
numpy_import = np.random.random()
class Candidate:
    def __init__(self):
        self.python_draw = python_import + random.random()
        self.numpy_draw = numpy_import + np.random.random()
    def approximate(self, budget, game):
        return InteractionValues(
            {(0,): self.python_draw + random.random(),
             (1,): self.numpy_draw + np.random.random()},
            index="SV", n_players=2, min_order=0, max_order=1)
def factory(n, index, order, seed):
    return Candidate()
""")
    game = {**game_spec(), "artifact": "game.npz"}
    spec = f"{path}:factory"
    first = run_one(game, tmp_path, "candidate", 4, 17, spec)
    other = run_one(game, tmp_path, "candidate", 4, 18, spec)
    repeated = run_one(game, tmp_path, "candidate", 4, 17, spec)
    assert first["status"] == other["status"] == repeated["status"] == "ok"
    assert first["estimate"] == repeated["estimate"]
    assert first["estimate"] != other["estimate"]
