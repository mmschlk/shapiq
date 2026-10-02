"""Cost qualification keeps four instances matched and never selects on attribution scores."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from shapiq_benchmark import qualification

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def controller(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid real datasets, CPU pinning and source scans when testing inventory decisions."""
    monkeypatch.setattr(qualification, "provenance", lambda: {"source_sha256": "frozen-source"})
    monkeypatch.setattr(
        qualification, "hardware", lambda: {"cpu_model": "fixture", "affinity": [0, 1]}
    )
    monkeypatch.setattr(
        qualification,
        "_bounded_dataset_identity",
        lambda name, *_: {"dataset": name, "data_sha256": "data"},
    )


def suite() -> dict:
    return {
        "game_seeds": [0, 1, 2, 3],
        "families": [{"id": name, "family": "dummy", "n_players": 11} for name in ("fast", "slow")],
        "games": [{"id": "structured", "oracle": "tree", "n_players": 30}],
        "seeds": [0],
        "relative_budgets": [0.5, 1, 2],
    }


def test_one_slow_seed_excludes_the_whole_recipe(
    controller: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """All four seeds are probed; one over-limit instance cannot skew the retained sample."""
    called = []

    def pilot(identity: dict, *_: object) -> dict:
        called.append((identity["spec"]["id"], identity["seed"]))
        return {"status": "measured", "projected_seconds": 500 if called[-1] == ("slow", 2) else 10}

    monkeypatch.setattr(qualification, "_run_pilot", pilot)
    original = suite()
    qualified = qualification.qualify_suite(original, tmp_path, "cache", max_seconds=100, workers=2)
    assert len(called) == 8 and set(called) == {
        (name, seed) for name in ("fast", "slow") for seed in range(4)
    }
    assert [spec["id"] for spec in qualified["families"]] == ["fast"]
    assert qualified["games"] == original["games"]
    assert original == suite()
    exclusion = qualified["preparation_exclusions"][0]
    assert exclusion["reason"] == "projected_cost_limit"
    assert exclusion["spec"]["id"] == "slow" and len(exclusion["instances"]) == 4
    assert json.loads((tmp_path / "qualified-suite.json").read_text()) == qualified
    assert (
        qualified["preparation_preflight"]["requested_suite_sha256"]
        == hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    )


def test_failures_are_explicit_and_private(
    controller: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unavailable datasets do not leak raw local exception strings into the public suite."""

    def unavailable(*_: object) -> dict:
        message = "private /secret/home/cache path"
        raise ValueError(message)

    monkeypatch.setattr(qualification, "_bounded_dataset_identity", unavailable)
    original = suite()
    for spec in original["families"]:
        spec["dataset"] = "broken"
    qualified = qualification.qualify_suite(original, tmp_path, "cache")
    assert qualified["families"] == []
    assert len(qualified["preparation_exclusions"]) == 2
    assert all(item["reason"] == "preflight_failed" for item in qualified["preparation_exclusions"])
    assert "/secret" not in json.dumps(qualified)
    assert "/secret" in next((tmp_path / "private").glob("*-dataset.log")).read_text()


def test_zero_payoffs_are_valid_and_cost_projection_counts_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cost gating must not quietly become a near-zero-truth or difficult-game filter."""
    seen = []

    class Game:
        n_players = 13

        def __call__(self, coalitions: np.ndarray) -> np.ndarray:
            seen.append(coalitions.copy())
            return np.zeros(len(coalitions))

    monkeypatch.setattr(qualification, "make_family", lambda *args, **kwargs: (Game(), {}))
    times = iter([10, 12, 20, 23])
    monkeypatch.setattr(qualification.time, "perf_counter", lambda: next(times))
    result = qualification._pilot({"family": "dummy", "n_players": 13}, 2, "cache")
    assert result["status"] == "measured"
    assert len(seen[0]) == 15 and len(seen[1]) == 32
    assert not seen[0][0].any() and seen[0][1].all()
    assert result["projected_constructions"] == 3
    assert result["projected_seconds"] == 2 * (2 * 3 + 3 / 32 * 8192)


def test_source_change_rejects_decision(
    controller: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pilot cannot certify measurements collected across a source change."""
    sources = iter([{"source_sha256": "old"}, {"source_sha256": "new"}])
    monkeypatch.setattr(qualification, "provenance", lambda: next(sources))
    monkeypatch.setattr(
        qualification, "_run_pilot", lambda *_: {"status": "measured", "projected_seconds": 1}
    )
    with pytest.raises(ValueError, match="Source changed"):
        qualification.qualify_suite(suite(), tmp_path, "cache")
    assert not (tmp_path / "qualified-suite.json").exists()


def test_pilot_process_timeout_kills_worker_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stuck fit cannot hold the controller past its independent pilot deadline."""

    class Process:
        pid = 321

        def wait(self, timeout: float | None = None) -> int:
            if timeout is not None:
                command = "pilot"
                raise qualification.subprocess.TimeoutExpired(command, timeout)
            return -9

    killed = []
    monkeypatch.setattr(qualification.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(qualification.os, "killpg", lambda *args: killed.append(args))
    identity = {"spec": {"family": "dummy", "n_players": 11}, "seed": 0}
    result = qualification._run_pilot(identity, tmp_path, "cache", 0.1, None)
    assert killed == [(321, qualification.signal.SIGKILL)]
    assert result == {"status": "failed", "reason": "pilot_timeout", "limit_seconds": 0.1}
    # Identical source/game/device identity reuses the preserved failed pilot.
    assert qualification._run_pilot(identity, tmp_path, "cache", 0.1, None) == result
    assert len(killed) == 1
