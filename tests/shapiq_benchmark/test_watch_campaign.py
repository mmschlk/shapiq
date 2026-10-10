"""Continuation survives idle implementation stages without duplicate wake-ups."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "watch_campaign", Path(__file__).resolve().parents[2] / "benchmark/watch_campaign.py"
)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def test_optional_heartbeat_continues_work_while_jobs_run() -> None:
    config = {"status": "active", "created_at": 0, "heartbeat_seconds": 900}
    jobs = {"123": "RUNNING"}
    state = {"last_progress_at": 10}
    assert watch.wake_reason(config, state, jobs, 909) is None
    assert "Scheduled progress check" in watch.wake_reason(config, state, jobs, 910)
    assert watch.wake_reason(config, {**state, "pending_delivery": True}, jobs, 910) is None
    assert watch.wake_reason({**config, "status": "cancelled"}, state, jobs, 910) is None


def test_idle_implementation_and_terminal_jobs() -> None:
    config = {"status": "active", "created_at": 0, "idle_seconds": 100}
    assert watch.wake_reason(config, {}, {}, 99) is None
    assert watch.wake_reason(config, {}, {}, 100)
    assert watch.wake_reason(config, {}, {"1": "RUNNING"}, 200) is None
    assert watch.wake_reason(config, {}, {"1": "FAILED", "2": "PENDING"}, 1)
    assert watch.wake_reason(config, {"pending_delivery": True}, {"1": "FAILED"}, 200) is None
    acknowledged = {"notified_jobs": {"1": "COMPLETED"}, "last_progress_at": 190}
    assert watch.wake_reason(config, acknowledged, {"1": "COMPLETED"}, 200) is None
    assert watch.wake_reason(config, acknowledged, {"1": "COMPLETED"}, 300)
    for status in ("paused", "cancelled", "complete"):
        assert watch.wake_reason({**config, "status": status}, {}, {}, 300) is None


def test_missing_accounting_is_not_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(watch.subprocess, "check_output", lambda *a, **kw: "1|COMPLETED|\n")
    with pytest.raises(RuntimeError, match="Accounting is incomplete"):
        watch.job_states({"jobs": ["1", "2"], "started_on": "2026-10-01"})


def test_delivery_once_and_acknowledgement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "status": "active",
        "created_at": 0,
        "codex": "codex",
        "thread_id": "thread",
        "cwd": str(tmp_path),
    }
    (tmp_path / "watch.json").write_text(json.dumps(config))
    calls = []

    def queue(command: list, **kwargs: object) -> object:
        calls.append(command)
        return type("Result", (), {"stdout": "queued", "returncode": 0})()

    monkeypatch.setattr(watch.subprocess, "run", queue)
    watch.check(tmp_path)
    watch.check(tmp_path)
    assert len(calls) == 1
    assert "--thread" in calls[0]
    assert "Continue remaining authorized work" in calls[0][-1]
    assert "active scientific plan named in the campaign handoff" in calls[0][-1]
    assert "ROADMAP" not in calls[0][-1]
    watch.check(tmp_path, acknowledge=True)
    watch.check(tmp_path)
    assert len(calls) == 1
    assert not json.loads((tmp_path / "watch-state.json").read_text())["pending_delivery"]


def test_failed_queue_retries_without_false_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed delivery must not claim success or permanently suppress the next tick."""
    config = {
        "status": "active",
        "created_at": 0,
        "codex": "codex",
        "thread_id": "thread",
        "cwd": str(tmp_path),
    }
    (tmp_path / "watch.json").write_text(json.dumps(config))

    def fail(*args: object, **kwargs: object) -> object:
        return type("Result", (), {"returncode": 1, "stderr": "queue storage failed"})()

    monkeypatch.setattr(watch.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="queue storage failed"):
        watch.check(tmp_path)
    assert not (tmp_path / "watch-state.json").exists()
    calls = []
    monkeypatch.setattr(
        watch.subprocess,
        "run",
        lambda *a, **k: (
            calls.append(a) or type("Result", (), {"stdout": "receipt", "returncode": 0})()
        ),
    )
    watch.check(tmp_path)
    assert len(calls) == 1
    assert json.loads((tmp_path / "watch-state.json").read_text())["pending_delivery"]


def test_paused_and_incomplete_accounting_never_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation and incomplete accounting prevent delivery, not just completion claims."""
    config = {"status": "paused", "created_at": 0, "jobs": ["1", "2"], "started_on": "2026-10-01"}
    (tmp_path / "watch.json").write_text(json.dumps(config))
    monkeypatch.setattr(watch.subprocess, "run", lambda *a, **k: pytest.fail("unexpected delivery"))
    monkeypatch.setattr(watch.subprocess, "check_output", lambda *a, **k: "1|COMPLETED|\n")
    watch.check(tmp_path)
    config["status"] = "active"
    (tmp_path / "watch.json").write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="Accounting is incomplete"):
        watch.check(tmp_path)
    assert not (tmp_path / "watch-state.json").exists()


def test_notified_history_survives_new_job_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing the active job list cannot erase a delivered terminal-job receipt."""
    config = {
        "status": "active",
        "created_at": 0,
        "codex": "codex",
        "thread_id": "thread",
        "cwd": str(tmp_path),
    }
    (tmp_path / "watch.json").write_text(json.dumps(config))
    (tmp_path / "watch-state.json").write_text(json.dumps({"notified_jobs": {"1": "COMPLETED"}}))
    monkeypatch.setattr(watch, "job_states", lambda _: {"2": "COMPLETED"})
    monkeypatch.setattr(
        watch.subprocess,
        "run",
        lambda *a, **k: type("Result", (), {"stdout": "receipt", "returncode": 0})(),
    )
    watch.check(tmp_path)
    assert json.loads((tmp_path / "watch-state.json").read_text())["notified_jobs"] == {
        "1": "COMPLETED",
        "2": "COMPLETED",
    }


def test_array_tasks_are_expanded_and_all_accounted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Array root IDs cannot hide unfinished or missing element jobs."""
    commands = []

    def accounting(command: list[str], **kwargs: object) -> str:
        commands.append(command)
        return "123_0|COMPLETED|\n123_1|RUNNING|\n"

    monkeypatch.setattr(watch.subprocess, "check_output", accounting)
    config = {"jobs": ["123_0", "123_1"], "started_on": "2026-10-01"}
    assert watch.job_states(config) == {"123_0": "COMPLETED", "123_1": "RUNNING"}
    assert "--array" in commands[0]
    assert commands[0][commands[0].index("-j") + 1] == "123"
    with pytest.raises(RuntimeError, match="Accounting is incomplete"):
        watch.job_states({**config, "jobs": ["123_0", "123_1", "123_2"]})


def test_pending_accounting_ranges_use_expanded_live_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unmaterialized array elements appear individually in squeue, not necessarily sacct."""
    commands = []
    jobs = [f"{root}_{task}" for root in (361650, 361651) for task in range(20)]

    def output(command: list[str], **kwargs: object) -> str:
        commands.append(command)
        if command[0].endswith("sacct"):
            return (
                "361650_0|COMPLETED|\n361650_1|RUNNING|\n"
                "361650_[2-19%1]|PENDING|\n361651_[0-19%1]|PENDING|\n"
            )
        return (
            "\n".join(
                f"{job}|{'RUNNING' if job == '361650_1' else 'PENDING'}"
                for job in jobs
                if job != "361650_0"
            )
            + "\n999999_0|RUNNING\n"
        )

    monkeypatch.setattr(watch.subprocess, "check_output", output)
    states = watch.job_states({"jobs": jobs, "started_on": "2026-10-01"})
    assert len(states) == 40
    assert states["361650_0"] == "COMPLETED"
    assert states["361650_1"] == "RUNNING"
    assert list(states.values()).count("PENDING") == 38
    assert "--array" in commands[1] and "--format=%i|%T" in commands[1]
    assert any(argument.startswith("--user=") for argument in commands[1])


@pytest.mark.parametrize("live_state", ["PENDING", "RUNNING", "COMPLETING"])
def test_live_requeued_task_overrides_stale_terminal_accounting(
    monkeypatch: pytest.MonkeyPatch, live_state: str
) -> None:
    """Accounting from an earlier attempt cannot notify completion of a live requeue."""
    monkeypatch.setattr(
        watch.subprocess,
        "check_output",
        lambda command, **kwargs: (
            "123_0|COMPLETED|\n" if command[0].endswith("sacct") else f"123_0|{live_state}\n"
        ),
    )
    assert watch.job_states({"jobs": ["123_0"], "started_on": "2026-10-01"}) == {
        "123_0": live_state
    }


def test_queue_terminal_state_does_not_replace_missing_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only an explicit terminal accounting record establishes completed work."""
    monkeypatch.setattr(
        watch.subprocess,
        "check_output",
        lambda command, **kwargs: "" if command[0].endswith("sacct") else "123_0|COMPLETED\n",
    )
    with pytest.raises(RuntimeError, match="Accounting is incomplete"):
        watch.job_states({"jobs": ["123_0"], "started_on": "2026-10-01"})


@pytest.mark.parametrize("failure", ["FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"])
def test_failed_dependency_gets_an_idle_retry_after_acknowledgement(failure: str) -> None:
    """An acknowledged failure cannot leave its dependent pending job silently stalled."""
    config = {"status": "active", "created_at": 0, "idle_seconds": 100}
    jobs = {"123_0": failure, "124_0": "PENDING"}
    state = {"notified_jobs": {"123_0": failure}, "last_progress_at": 20}
    assert watch.wake_reason(config, state, jobs, 119) is None
    assert "pending dependencies" in watch.wake_reason(config, state, jobs, 120)
    assert watch.wake_reason(config, {**state, "pending_delivery": True}, jobs, 120) is None
    assert watch.wake_reason(config, {**state, "last_progress_at": 120}, jobs, 121) is None
    assert watch.wake_reason(config, state, {**jobs, "125_0": "RUNNING"}, 120) is None


def test_ordinary_resource_queue_does_not_trigger_blocked_dependency_retry() -> None:
    """Pending resources without a failed predecessor must not generate idle notifications."""
    config = {"status": "active", "created_at": 0, "idle_seconds": 100}
    assert watch.wake_reason(config, {}, {"124_0": "PENDING"}, 200) is None
    state = {"notified_jobs": {"123_0": "COMPLETED"}, "last_progress_at": 20}
    assert watch.wake_reason(config, state, {"123_0": "COMPLETED", "124_0": "PENDING"}, 200) is None
