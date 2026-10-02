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
    assert len(called) == 12 and set(called) == {
        (name, seed) for name in ("fast", "slow", "structured") for seed in range(4)
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
    original["games"] = []
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


def test_structured_seed_failure_excludes_requested_recipe(
    controller: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A large solver must qualify at actual width for every seed, not only a small proxy."""
    original = suite()
    original["families"] = []
    original["games"][0].update(n_players=1776, index="SII", order=2)
    calls = []

    def pilot(identity: dict, directory: Path, cache: str, timeout: float, cpu: int) -> dict:
        calls.append((identity, timeout))
        return (
            {"status": "failed", "reason": "pilot_timeout"}
            if identity["seed"] == 2
            else {"status": "measured", "projected_seconds": 20}
        )

    monkeypatch.setattr(qualification, "_run_pilot", pilot)
    qualified = qualification.qualify_suite(
        original, tmp_path, "cache", structured_timeout=90, structured_memory_gb=8
    )
    assert qualified["games"] == []
    assert len(calls) == 4
    assert all(identity["spec"]["n_players"] == 1776 for identity, _ in calls)
    assert all(identity["structured_memory_gb"] == 8 and limit == 90 for identity, limit in calls)
    exclusion = qualified["preparation_exclusions"][0]
    assert exclusion["kind"] == "games" and exclusion["spec"] == original["games"][0]
    assert len(exclusion["instances"]) == 4


def test_structured_pilot_measures_full_requested_solver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The measured work includes artifact serialization and retains actual solver dimensions."""
    seen = []

    def prepare(specs: list[dict], output: Path) -> list[dict]:
        seen.extend(specs)
        (output / "game.npz").write_bytes(b"fixture artifact")
        return [{"artifact": "game.npz", "n_players": 1776, "index": "SII", "order": 2}]

    monkeypatch.setattr(qualification, "prepare_structured", prepare)
    times = iter([1, 8])
    monkeypatch.setattr(qualification.time, "perf_counter", lambda: next(times))
    spec = {"id": "native", "oracle": "tree", "n_players": 1776, "index": "SII", "order": 2}
    result = qualification._structured_pilot(spec, 3, str(tmp_path / ".models"))
    assert seen == [{**spec, "id": "native-i3", "instance_seed": 3}]
    assert result["actual_preparation_seconds"] == 7
    assert result["n_players"] == 1776
    assert result["artifact_bytes"] == len(b"fixture artifact")
    assert result["peak_rss_bytes"] > 0
    assert not list(tmp_path.glob("structured-pilot-*"))


@pytest.mark.parametrize("status", ["noise_dominated", "indeterminate"])
def test_noisy_imputation_is_excluded_without_changing_the_game(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """The gate keeps evidence and never silently substitutes the higher-sample oracle."""
    game = object()
    options = []

    def factory(*args: object, **kwargs: object) -> tuple:
        options.append(kwargs)
        return game, {}

    def stability(actual: object, *, seed: int) -> dict:
        assert actual is game and seed == 3
        return {"status": status, "levels": [{"noise_ratio": 0.9}]}

    monkeypatch.setattr(qualification, "make_family", factory)
    monkeypatch.setattr(qualification, "imputation_stability", stability)
    with pytest.raises(qualification.QualityExclusion) as caught:
        qualification._pilot(
            {"family": "local_gaussian", "n_players": 11, "quality_protocol": "quality-v2"},
            3,
            "cache",
        )
    assert caught.value.reason == "unstable_imputation"
    assert caught.value.details["status"] == status
    assert options == [{"instance_seed": 3, "n_players": 11, "quality_protocol": "quality-v2"}]
