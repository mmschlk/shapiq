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
        return type("Result", (), {"stdout": "queued"})()

    monkeypatch.setattr(watch.subprocess, "run", queue)
    watch.check(tmp_path)
    watch.check(tmp_path)
    assert len(calls) == 1
    assert "--thread" in calls[0]
    assert "start the next phase" in calls[0][-1]
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

    def fail(*args: object, **kwargs: object) -> None:
        raise watch.subprocess.CalledProcessError(1, "codex")

    monkeypatch.setattr(watch.subprocess, "run", fail)
    with pytest.raises(watch.subprocess.CalledProcessError):
        watch.check(tmp_path)
    assert not (tmp_path / "watch-state.json").exists()
    calls = []
    monkeypatch.setattr(
        watch.subprocess,
        "run",
        lambda *a, **k: calls.append(a) or type("Result", (), {"stdout": "receipt"})(),
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
        watch.subprocess, "run", lambda *a, **k: type("Result", (), {"stdout": "receipt"})()
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
