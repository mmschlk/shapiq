from __future__ import annotations

from leaderboard.runner.runner_storage_adapter import save_raw_results


class FakeDB:
    """Small fake database object for testing."""

    def __init__(self):
        self.inserted_runs = []

    def insert_one(self, run_record):
        self.inserted_runs.append(run_record)


def test_save_raw_results():
    """Test if all results are saved in the fake database."""
    fake_db = FakeDB()

    raw_results = [
        {
            "run_id": "run-1",
            "metrics": {"mse": 0.1},
        },
        {
            "run_id": "run-2",
            "metrics": {"mse": 0.2},
        },
    ]

    save_raw_results(
        db=fake_db,
        raw_results=raw_results,
    )

    assert len(fake_db.inserted_runs) == 2
    assert fake_db.inserted_runs[0] == raw_results[0]
    assert fake_db.inserted_runs[1] == raw_results[1]
