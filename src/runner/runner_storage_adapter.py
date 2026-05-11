from leaderboard.storage.database import MongoDBClient

def save_raw_results(db: MongoDBClient, raw_results: list[dict]) -> None:
    for run_record in raw_results:
        db.insert_run(run_record)