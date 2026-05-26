"""Runner storage adapter for the leaderboard."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
