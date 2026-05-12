from typing import Any
from leaderboard.storage.database import MongoDBClient

def save_raw_results(db: MongoDBClient, raw_results: list[dict[str, Any]]) -> None:
    """Store raw benchmark run records in the database.

    Args:
        db: The database client used to store the records.
        raw_results: The raw run records to insert.
    """
    for run_record in raw_results:
        db.insert_run(run_record)