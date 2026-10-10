"""Compact private checkpoints: one shared snapshot and an append-only cell journal.

Readers expand this storage format to the original results schema. Final exports
authenticate the entire journal; only a locked runner may recover an interrupted
append and accept complete records written after the last manifest checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

FORMAT = "snapshot-journal-v1"
SNAPSHOT_FIELDS = ("suite", "games", "coverage")


def _bytes(value: dict) -> bytes:
    return (json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n").encode()


def result_inputs(path: Path) -> list[Path]:
    """List every file needed to authenticate a result, including legacy files."""
    data = json.loads(path.read_text())
    if data.get("storage_format") != FORMAT:
        return [path]
    return [path, (path.parent / data["snapshot_file"]).resolve(), path.parent / "records.jsonl"]


def read_results(path: Path, *, recover: bool = False) -> dict:
    """Read either format; reject incomplete journals except during locked resume."""
    data = json.loads(path.read_text())
    if data.get("storage_format") != FORMAT:
        return data
    snapshot_path = (path.parent / data.pop("snapshot_file")).resolve()
    raw_snapshot = snapshot_path.read_bytes()
    if hashlib.sha256(raw_snapshot).hexdigest() != data.pop("snapshot_file_sha256"):
        message = "Result snapshot bytes changed."
        raise ValueError(message)
    snapshot = json.loads(raw_snapshot)
    if snapshot["snapshot_id"] != data["snapshot_id"]:
        message = "Result snapshot identity changed."
        raise ValueError(message)
    journal = (path.parent / "records.jsonl").read_bytes()
    expected = data.pop("journal")
    if not recover and hashlib.sha256(journal).hexdigest() != expected["sha256"]:
        message = "Result journal is incomplete or changed."
        raise ValueError(message)
    # A newline is the commit marker. Invalid complete lines always fail closed.
    lines = journal.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        if not recover:
            message = "Result journal has an interrupted final record."
            raise ValueError(message)
        lines.pop()
    if recover and (
        len(lines) < expected["records"]
        or hashlib.sha256(b"".join(lines[: expected["records"]])).hexdigest() != expected["sha256"]
    ):
        message = "Previously committed result journal changed."
        raise ValueError(message)
    records = [json.loads(line) for line in lines]
    if not recover and len(records) != expected["records"]:
        message = "Result journal record count changed."
        raise ValueError(message)
    del data["storage_format"]
    data.update({key: snapshot.get(key, []) for key in SNAPSHOT_FIELDS})
    data["snapshot_provenance"] = snapshot["provenance"]
    data["records"] = records
    if recover and "campaign" in data:
        data["campaign"].update(
            completed=len(records), complete=len(records) == data["campaign"]["planned"]
        )
    return data


class Checkpoint:
    """Append one cell at a time; write the small manifest at run boundaries."""

    def __init__(self, output: Path, snapshot: Path, result: dict) -> None:
        """Initialize a journal or migrate a verified legacy checkpoint once."""
        self.output = output
        self.snapshot = snapshot / "snapshot.json" if snapshot.is_dir() else snapshot
        raw = self.snapshot.read_bytes()
        declared = json.loads(raw)
        if (
            declared["snapshot_id"] != result["snapshot_id"]
            or declared["provenance"] != result["snapshot_provenance"]
            or any(declared.get(key, []) != result[key] for key in SNAPSHOT_FIELDS)
        ):
            message = "Snapshot metadata changed after parent authentication."
            raise ValueError(message)
        self.snapshot_hash = hashlib.sha256(raw).hexdigest()
        self.journal_hash = hashlib.sha256()
        # Rebuild once on resume (also migrates legacy checkpoints). This removes
        # only a recoverable incomplete final append, never a complete record.
        temporary = output / "records.jsonl.tmp"
        with temporary.open("wb") as stream:
            for record in result["records"]:
                raw = _bytes(record)
                stream.write(raw)
                self.journal_hash.update(raw)
        temporary.replace(output / "records.jsonl")
        self.finish(result)

    def append(self, record: dict) -> None:
        """Commit one complete line without copying earlier rows or game metadata."""
        raw = _bytes(record)
        with (self.output / "records.jsonl").open("ab") as stream:
            stream.write(raw)
        self.journal_hash.update(raw)

    def finish(self, result: dict) -> None:
        """Atomically publish the journal digest and ordinary campaign progress."""
        manifest = {
            key: value
            for key, value in result.items()
            if key not in (*SNAPSHOT_FIELDS, "snapshot_provenance", "records")
        }
        manifest.update(
            storage_format=FORMAT,
            snapshot_file=os.path.relpath(self.snapshot, self.output),
            snapshot_file_sha256=self.snapshot_hash,
            journal={"sha256": self.journal_hash.hexdigest(), "records": len(result["records"])},
        )
        temporary = self.output / "results.json.tmp"
        temporary.write_bytes(_bytes(manifest))
        temporary.replace(self.output / "results.json")
