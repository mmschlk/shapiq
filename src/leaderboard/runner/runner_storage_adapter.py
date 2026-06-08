"""Runner storage adapter for the leaderboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from leaderboard.storage.connection import DatabaseClient


def save_raw_results(db: DatabaseClient, raw_results: list[dict[str, Any]]) -> None:
    """Store raw benchmark run records in the database.

    Args:
        db: The database client used to store the records.
        raw_results: The raw run records to insert.
    """
    for run_record in raw_results:
        db.insert_one(run_record)


def json_default(value: object) -> object:
    """Convert non-standard numeric values to JSON-compatible values."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def save_raw_results_jsonl(
    raw_results: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Append raw benchmark results to a JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as file:
        for run_record in raw_results:
            file.write(json.dumps(run_record, default=json_default))
            file.write("\n")
