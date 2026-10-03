"""Resource verification must inspect the actual array task, not its siblings."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from shapiq_benchmark import execution, runner

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
