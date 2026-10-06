"""Reservations remain conservative until exact terminal task accounting arrives."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "focused_campaign", Path(__file__).resolve().parents[2] / "benchmark/focused_campaign.py"
)
assert SPEC and SPEC.loader
focused = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(focused)


def test_pending_and_missing_accounting_keep_full_reservation() -> None:
    reservations = [{"id": f"42_{i}", "cpus": 16, "seconds": 7200} for i in range(4)]
    first = focused.budget_status(reservations, {})
    assert first["committed_cpu_hours"] == 128
    # One terminal task consumed 4 hours; three still reserve 32 each.
    second = focused.budget_status(reservations, {"42_0": 14400})
    assert second["settled_cpu_hours"] == 4
    assert second["reserved_cpu_hours"] == 96
    assert second["committed_cpu_hours"] == 100


def test_limit_includes_all_commitments_and_actual_overrun() -> None:
    reservation = [{"id": "42", "cpus": 128, "seconds": 16 * 3600}]
    assert focused.budget_status(reservation, {})["remaining_cpu_hours"] == 0
    assert not focused.budget_status(reservation, {"42": 2048 * 3600 + 1})["within_cap"]
    assert not focused.budget_status([*reservation, {"id": "43", "cpus": 1, "seconds": 1}], {})[
        "within_cap"
    ]


@pytest.mark.parametrize(
    ("reservations", "settled"),
    [
        ([{"id": "42_0", "cpus": 1, "seconds": 10}] * 2, {}),
        ([{"id": task, "cpus": 1, "seconds": 10} for task in ["42", "42_0"]], {}),
        ([{"id": "42_[0-3]", "cpus": 1, "seconds": 10}], {}),
        ([{"id": "42.batch", "cpus": 1, "seconds": 10}], {}),
        ([{"id": "42", "cpus": 0, "seconds": 10}], {}),
        ([{"id": "42", "cpus": 1, "seconds": float("inf")}], {}),
        ([{"id": "42", "cpus": 1, "seconds": 10}], {"43": 0}),
        ([{"id": "42", "cpus": 1, "seconds": 10}], {"42": -1}),
        ([{"id": "42", "cpus": 1, "seconds": 10}], {"42": float("nan")}),
    ],
)
def test_ambiguous_or_invalid_accounting_fails_closed(reservations: list, settled: dict) -> None:
    with pytest.raises(ValueError):
        focused.budget_status(reservations, settled)


def test_changed_request_does_not_reuse_directory(tmp_path: Path) -> None:
    path = tmp_path / "request.json"
    focused.immutable_json(path, {"suite": "original"})
    focused.immutable_json(path, {"suite": "original"})
    with pytest.raises(ValueError, match="Changed frozen request"):
        focused.immutable_json(path, {"suite": "changed"})
    assert json.loads(path.read_text()) == {"suite": "original"}


def test_preflight_rejects_login_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        focused.preflight(argparse.Namespace())


def test_preflight_delegates_all_seeds_and_explicit_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shapiq_benchmark import execution, qualification, runner

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        focused.subprocess,
        "check_output",
        lambda *_, **__: "JobId=123 JobState=RUNNING NumNodes=1 NumCPUs=2 AllocTRES=cpu=2,mem=24G",
    )
    for name in execution.THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(runner, "provenance", lambda: {"source": "frozen"})
    monkeypatch.setattr(
        execution, "hardware", lambda: {"cpu_model": "AMD EPYC 9754", "affinity": [0, 1]}
    )
    suite = {"game_seeds": [0, 1, 2, 3], "families": [{"id": "one"}], "games": []}
    source_path, suite_path = tmp_path / "source.json", tmp_path / "suite.json"
    source_path.write_text(json.dumps({"source": "frozen"}))
    suite_path.write_text(json.dumps(suite))
    captured = {}

    def qualify(actual: dict, path: Path, **limits: object) -> dict:
        captured.update(suite=actual, path=path, limits=limits)
        return actual

    monkeypatch.setattr(qualification, "qualify_suite", qualify)
    args = argparse.Namespace(
        suite=suite_path,
        output=tmp_path / "pilot",
        expected_source=source_path,
        workers=2,
        max_preparation_seconds=3600,
        pilot_timeout=120,
        structured_timeout=180,
        memory_gb=12,
    )
    focused.preflight(args)
    assert captured["suite"] == suite
    assert captured["limits"] == {
        "model_cache": args.output / "models",
        "max_seconds": 3600,
        "pilot_timeout": 120,
        "structured_timeout": 180,
        "structured_memory_gb": 12,
        "workers": 2,
    }
    source_path.write_text(json.dumps({"source": "wrong"}))
    with pytest.raises(ValueError, match="source/environment"):
        focused.preflight(args)


def test_exact_array_task_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "150")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "100")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    commands = []

    def control(command: list, **_: object) -> str:
        commands.append(command)
        return (
            "JobId=150 ArrayJobId=100 ArrayTaskId=3 JobState=RUNNING "
            "NumNodes=1 NumCPUs=16 AllocTRES=cpu=16,mem=192G Gres=(null)"
        )

    monkeypatch.setattr(focused.subprocess, "check_output", control)
    task, fields = focused.slurm_allocation(16)
    assert task == "100_3" and fields["NumCPUs"] == "16"
    assert commands == [["scontrol", "show", "job", "100_3", "-o"]]


@pytest.mark.parametrize(
    "output",
    [
        "JobId=123 JobState=RUNNING NumNodes=1 NumCPUs=16 AllocTRES=cpu=16,gres/gpu=1",
        "JobId=123 JobState=RUNNING NumNodes=1 NumCPUs=8 AllocTRES=cpu=8",
        "JobId=123 JobState=RUNNING NumNodes=1 NumCPUs=129 AllocTRES=cpu=129",
        "JobId=123 JobState=RUNNING NumNodes=2 NumCPUs=16 AllocTRES=cpu=16",
        "JobId=999 JobState=RUNNING NumNodes=1 NumCPUs=16 AllocTRES=cpu=16",
        "JobId=123 JobState=PENDING NumNodes=1 NumCPUs=16 AllocTRES=cpu=16",
        "JobId=123\nJobId=124",
    ],
)
def test_invalid_slurm_allocation_rejected(output: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.setattr(focused.subprocess, "check_output", lambda *_, **__: output)
    with pytest.raises(ValueError):
        focused.slurm_allocation(16)
