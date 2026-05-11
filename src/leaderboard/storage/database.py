from typing import List, Dict, Any
from pymongo import MongoClient
from config import RunConfig


class MongoDBClient:
    def __init__(self, uri: str, db_name: str = "shapiq"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db["runs"]

    def insert_run(self, run_data: Dict[str, Any]) -> None:
        """Insert a single run document."""
        self.collection.insert_one(run_data)

    def get_all_runs(self) -> List[Dict[str, Any]]:
        """Fetch all runs (without Mongo _id)."""
        return list(self.collection.find({}, {"_id": 0}))

    def get_runs_by_config(self, config: RunConfig) -> List[Dict[str, Any]]:
        """Fetch runs matching a configuration (ignores seed differences)."""
        query = config.to_dict()
        return list(self.collection.find(query, {"_id": 0}))

    def get_all_configurations(self) -> List[RunConfig]:
        """Return unique configurations across all runs."""
        pipeline = [
            {
                "$group": {
                    "_id": {
                        "game_name": "$game_name",
                        "n_players": "$n_players",
                        "approximator_name": "$approximator_name",
                        "index": "$index",
                        "max_order": "$max_order",
                        "budget": "$budget",
                        "ground_truth_method": "$ground_truth_method",
                        "game_params": "$game_params",
                        "approximator_params": "$approximator_params",
                    }
                }
            }
        ]

        configs = []
        for entry in self.collection.aggregate(pipeline):
            configs.append(RunConfig(**entry["_id"]))

        return configs