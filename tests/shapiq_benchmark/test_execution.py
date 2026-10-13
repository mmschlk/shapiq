"""Resource verification must inspect the actual array task, not its siblings."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from shapiq_benchmark import (
    campaign_runner as runner,
    execution,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def profiled_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supply a valid pinned worker without contacting Slurm."""
    monkeypatch.setattr(
        execution, "hardware", lambda: {"cpu_model": "AMD EPYC 9754", "affinity": [6]}
    )
    for name in execution.THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("SLURM_JOB_ID", "361642")
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)


@pytest.mark.parametrize("task", ["0", "6", None])
def test_profile_selects_exact_array_task(
    profiled_environment: None, monkeypatch: pytest.MonkeyPatch, task: str | None
) -> None:
    """An array-root response contains sibling records that must not overwrite this allocation."""
    if task is not None:
        monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "361642")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", task)
    expected = "361642" if task is None else f"361642_{task}"
    commands = []

    def description(command: list[str], **kwargs: object) -> str:
        commands.append(command)
        if command[2] == "node":
            return "NodeName=himem02 CPUTot=128 ThreadsPerCore=1"
        allocated = "OverSubscribe=NO NumNodes=1 NodeList=himem02 NumCPUs=128"
        if command[3] == expected:
            return allocated
        return allocated + "\nOverSubscribe=NO NumNodes=1-1 NodeList= NumCPUs=128\n"

    monkeypatch.setattr(execution.subprocess, "check_output", description)
    execution.verify_profile(execution.PROFILE)
    assert commands == [
        ["scontrol", "show", "job", expected, "-o"],
        ["scontrol", "show", "node", "himem02", "-o"],
    ]


def test_profile_rejects_ambiguous_response(
    profiled_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even compatible-looking sibling records cannot establish one allocation."""
    monkeypatch.setattr(
        execution.subprocess,
        "check_output",
        lambda *args, **kwargs: (
            "OverSubscribe=NO NumNodes=1 NodeList=himem01 NumCPUs=128\n"
            "OverSubscribe=NO NumNodes=1 NodeList=himem02 NumCPUs=128\n"
        ),
    )
    with pytest.raises(ValueError, match="ambiguous job allocation"):
        execution.verify_profile(execution.PROFILE)


@pytest.mark.parametrize("tamper", [False, True])
def test_worker_authenticates_only_parent_selected_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, tamper: bool
) -> None:
    """A cell must not scan the panel, but must still reject changes to its own oracle."""
    artifact = tmp_path / "game.npz"
    artifact.write_bytes(b"authenticated oracle")
    request = {
        "memory_gb": None,
        "artifact_root": str(tmp_path),
        "authenticated_game": {"id": "selected", "artifact": "game.npz"},
        "game_id": "selected",
        "artifact_sha256": runner.digest(artifact),
        "expected_source_hash": "source",
        "method": "KernelSHAP",
        "budget": 4,
        "seed": 0,
        "timing_profile": "diagnostic",
    }
    if tamper:
        artifact.write_bytes(b"changed oracle")
    checked = []
    digest = runner.digest
    monkeypatch.setattr(runner, "digest", lambda path: checked.append(path) or digest(path))
    monkeypatch.setattr(runner, "provenance", lambda: {"source_sha256": "source"})

    def forbidden(*args: object) -> None:
        pytest.fail("worker reloaded the full snapshot")

    monkeypatch.setattr(runner, "load_snapshot", forbidden)
    monkeypatch.setattr(runner, "run_one", lambda *args, **kwargs: {"status": "ok", "nmse": 0})
    source, destination = tmp_path / "request.json", tmp_path / "response.json"
    source.write_text(json.dumps(request))
    execution.worker(source, destination)
    response = json.loads(destination.read_text())
    assert checked == [artifact]
    assert response["status"] == ("failed" if tamper else "ok")
    if tamper:
        assert "artifact changed" in response["error"]
    else:
        assert response["worker"]["peak_rss_bytes"] > 0


@pytest.mark.parametrize(
    ("startup", "estimator", "scoring", "startup_limit", "output_limit", "expected"),
    [
        (0.25, 0.01, 0.25, 2, 2, "ok"),
        (0, 0.4, 0, 2, 2, "TimeoutError:"),
        (0.4, 0.01, 0, 0.1, 2, "StartupTimeoutError:"),
        (0, 0.01, 0.4, 2, 0.1, "OutputTimeoutError:"),
    ],
)
def test_estimator_timeout_excludes_startup_and_scoring(
    monkeypatch: pytest.MonkeyPatch,
    startup: float,
    estimator: float,
    scoring: float,
    startup_limit: float,
    output_limit: float,
    expected: str,
) -> None:
    """Real child processes exercise independent startup/execution/output deadlines."""
    child = """
import json, pathlib, sys, time
request = json.loads(pathlib.Path(sys.argv[1]).read_text())
phase = pathlib.Path(request['execution_phase_path'])
def mark(name, seconds=None):
    temporary = phase.with_suffix('.tmp')
    temporary.write_text(json.dumps({'phase': name, 'time': time.perf_counter(), 'seconds': seconds}))
    temporary.replace(phase)
time.sleep(request['startup'])
mark('started')
start = time.perf_counter()
time.sleep(request['estimator'])
elapsed = time.perf_counter() - start
mark('finished', elapsed)
time.sleep(request['scoring'])
pathlib.Path(sys.argv[2]).write_text(json.dumps({'status': 'ok', 'seconds': elapsed, 'execution_seconds': elapsed}))
"""
    popen = subprocess.Popen

    def launch(command: list[str], **kwargs: object) -> subprocess.Popen:
        return popen([sys.executable, "-S", "-c", child, *command[-2:]], **kwargs)

    monkeypatch.setattr(execution.subprocess, "Popen", launch)
    result = execution.isolated(
        {"startup": startup, "estimator": estimator, "scoring": scoring},
        0.1,
        None,
        startup_timeout=startup_limit,
        output_timeout=output_limit,
    )
    if expected == "ok":
        assert result["status"] == "ok"
        assert result["seconds"] < 0.1
        assert result["wall_seconds"] > 0.5
    else:
        assert result["status"] == "failed"
        assert result["error"].startswith(expected)
        assert result["seconds"] is None


def test_readiness_follows_constructor_and_precedes_approximate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callbacks bracket approximate, never oracle creation, constructor, or scoring."""
    events = []
    counted = SimpleNamespace(queries=0, requested=0, exceeded=False, evaluation_seconds=None)
    monkeypatch.setattr(runner, "load_game", lambda *args: events.append("oracle"))
    monkeypatch.setattr(runner, "CountedGame", lambda *args: counted)

    def construct(*args: object) -> SimpleNamespace:
        events.append("constructor")

        def approximate(**kwargs: object) -> SimpleNamespace:
            events.append("approximate")
            return SimpleNamespace(dict_values={(0,): 1.0})

        return SimpleNamespace(approximate=approximate)

    monkeypatch.setattr(runner, "builtin_factory", construct)
    monkeypatch.setattr(runner, "score", lambda *args: events.append("score") or {})
    result = runner.run_one(
        {"n_players": 1, "index": "SV", "order": 1},
        None,
        "test",
        2,
        0,
        execution_started=lambda: events.append("ready"),
        execution_finished=lambda seconds: events.append("finished"),
    )
    assert result["status"] == "ok"
    assert events == ["oracle", "constructor", "ready", "approximate", "finished", "score"]
