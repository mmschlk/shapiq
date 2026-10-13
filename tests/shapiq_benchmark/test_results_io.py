"""Compact checkpoints preserve legacy results and recover only complete records."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from shapiq_benchmark.results_io import Checkpoint, read_results, result_inputs


def panel(tmp_path: Path) -> tuple[Path, dict]:
    """One large shared metadata envelope makes accidental copying visible."""
    snapshot = {
        "snapshot_id": "fixture",
        "provenance": {},
        "suite": {"methods": ["test"]},
        "games": [{"id": "test", "metadata": {"large": "x" * 100000}}],
        "coverage": [],
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot))
    return path, {
        **{key: snapshot[key] for key in ("snapshot_id", "suite", "games", "coverage")},
        "snapshot_provenance": {},
        "schema_version": 1,
        "records": [],
        "campaign": {"planned": 2, "completed": 0, "complete": False},
    }


def test_small_append_round_trip_and_legacy(tmp_path: Path) -> None:
    """Appending a cell never rewrites a metadata envelope or prior record."""
    snapshot, data = panel(tmp_path)
    checkpoint = Checkpoint(tmp_path, snapshot, data)
    manifest = tmp_path / "results.json"
    initial = manifest.read_bytes()
    first = {"method": "test", "status": "ok", "nmse": 0.1}
    checkpoint.append(first)
    data["records"].append(first)
    assert manifest.read_bytes() == initial
    assert len(initial) < 1000
    assert (tmp_path / "records.jsonl").stat().st_size < 100
    checkpoint.finish(data)
    assert read_results(manifest) == data
    assert result_inputs(manifest) == [manifest, snapshot.resolve(), tmp_path / "records.jsonl"]
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(data))
    assert read_results(legacy) == data
    assert result_inputs(legacy) == [legacy]


def test_recovery_keeps_complete_appends_and_discards_only_partial_tail(tmp_path: Path) -> None:
    """A killed runner cannot silently lose a completed cell or export a torn checkpoint."""
    snapshot, data = panel(tmp_path)
    checkpoint = Checkpoint(tmp_path, snapshot, data)
    row = {"status": "ok", "method": "test"}
    checkpoint.append(row)
    journal = tmp_path / "records.jsonl"
    with journal.open("ab") as stream:
        stream.write(b'{"status":')
    manifest = tmp_path / "results.json"
    with pytest.raises(ValueError, match="incomplete or changed"):
        read_results(manifest)
    recovered = read_results(manifest, recover=True)
    assert recovered["records"] == [row]
    assert recovered["campaign"]["completed"] == 1
    Checkpoint(tmp_path, snapshot, recovered)
    assert read_results(manifest) == recovered
    assert journal.read_bytes().endswith(b"\n")


@pytest.mark.parametrize("part", ["snapshot", "committed_record", "complete_tail"])
def test_tampering_and_invalid_complete_lines_fail_closed(tmp_path: Path, part: str) -> None:
    """Recovery cannot bypass the snapshot or previously committed journal digest."""
    snapshot, data = panel(tmp_path)
    data["records"] = [{"status": "ok"}]
    Checkpoint(tmp_path, snapshot, data)
    journal = tmp_path / "records.jsonl"
    if part == "snapshot":
        snapshot.write_text(snapshot.read_text() + " ")
    elif part == "committed_record":
        journal.write_text('{"status":"failed"}\n')
    else:
        with journal.open("ab") as stream:
            stream.write(b"invalid JSON\n")
    with pytest.raises(ValueError):
        read_results(tmp_path / "results.json", recover=True)


def test_checkpoint_rejects_snapshot_changed_after_parent_validation(tmp_path: Path) -> None:
    """A new file digest must not silently bind old records to changed game metadata."""
    snapshot, result = panel(tmp_path)
    changed = json.loads(snapshot.read_text())
    changed["suite"]["methods"] = ["different"]
    snapshot.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="after parent authentication"):
        Checkpoint(tmp_path, snapshot, result)
    assert not (tmp_path / "results.json").exists()
