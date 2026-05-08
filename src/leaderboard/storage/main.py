# This file demonstrates how to use the storage module to manage the result store.

import os
import json
from dotenv import load_dotenv

from database import MongoDBClient
from config import RunConfig
from metrics import MetricsLoader


def load_env():
    """Load environment variables from .env file."""
    load_dotenv()

    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "shapiq-leaderboard")

    if not uri:
        raise ValueError("MONGODB_URI not set in .env")

    return uri, db_name


def upload_jsonl(db: MongoDBClient, filepath: str):
    """Upload runs from a JSONL file into MongoDB."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found")

    inserted = 0

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                run_data = json.loads(line)
                db.insert_run(run_data)
                inserted += 1
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")

    print(f"Inserted {inserted} runs.")


def print_configurations(db: MongoDBClient):
    """Fetch and print all unique configurations."""
    configs = db.get_all_configurations()

    print(f"\nFound {len(configs)} unique configurations:\n")

    for i, config in enumerate(configs, start=1):
        print(f"--- Config {i} ---")
        print(config)
        print()

    return configs


def main():
    uri, db_name = load_env()

    # connect to DB
    db = MongoDBClient(uri=uri, db_name=db_name)

    # path to your JSONL file
    jsonl_path = "data/results_raw.jsonl" 

    # upload data
    upload_jsonl(db, jsonl_path)

    # print configurations
    configs = print_configurations(db)

    # print aggregated metrics for the first config
    if configs:
        print(f"Aggregated metrics for Config 1:")
        metrics_loader = MetricsLoader(db)
        aggregated_metrics = metrics_loader.aggregate_metrics(configs[0])
        print(json.dumps(aggregated_metrics, indent=2))


if __name__ == "__main__":
    main()