from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from leaderboard.storage.data_classes import RunConfig


def load_env() -> tuple[str, str]:
    """Load MongoDB connection parameters from environment variables."""
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "shapiq-leaderboard")
    if not uri:
        raise ValueError("MONGODB_URI is not set in the environment.")
    return uri, db_name


class MongoDBClient:
    """
    MongoDB client for shapiq experiment results.

    Handles the connection and exposes all read/write operations needed
    by the rest of the codebase.  

    Parameters
    ----------
    uri:
        MongoDB connection string (e.g. ``"mongodb://localhost:27017"``).
    db_name:
        Database name (default: ``"shapiq"``).
    collection_name:
        Collection name (default: ``"runs"``).
    """

    def __init__(self,
                 uri: str,
                 db_name: str = "shapiq",
                 collection_name: str = "runs") -> None:
        
        self._client: MongoClient = MongoClient(uri)
        self._db: Database = self._client[db_name]
        self.collection: Collection = self._db[collection_name]


    # Connection Handling

    def close(self) -> None:
        """Close the underlying MongoDB connection."""
        self._client.close()

    def __enter__(self) -> "MongoDBClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def check_connection(self) -> bool:
        """Check if the connection to MongoDB is alive."""
        try:
            self._client.admin.command("ping")
            return True
        except Exception:
            return False

    
    # Write 

    def insert_one(self, document: Dict[str, Any]) -> None:
        """Insert a single run document."""
        self.collection.insert_one(document)

    def insert_many(self, documents: List[Dict[str, Any]]) -> None:
        """Bulk-insert run documents (no-op for an empty list)."""
        if documents:
            self.collection.insert_many(documents)

    # Delete
    
    def delete_all(self) -> int:
        """Delete every document in the collection. Returns deleted count."""
        return self.collection.delete_many({}).deleted_count

    def delete_by_config(self, config: RunConfig) -> int:
        """Delete all documents matching *config*. Returns deleted count."""
        return self.collection.delete_many(config.to_dict()).deleted_count


    # Read — generic

    def get_all(self) -> List[Dict[str, Any]]:
        """Return every document (without ``_id``)."""
        return list(self.collection.find({}, {"_id": 0}))

    def get_by_config(self, config: RunConfig) -> List[Dict[str, Any]]:
        """Return all documents whose fields match *config*."""
        return list(self.collection.find(config.to_dict(), {"_id": 0}))


    # Read — domain helpers    

    def get_unique_configs(self) -> List[RunConfig]:
        """Return one ``RunConfig`` per unique configuration in the collection."""
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
        return [RunConfig.from_dict(e["_id"]) for e in self.collection.aggregate(pipeline)]

    def get_games(self) -> List[str]:
        """Return sorted distinct game names."""
        return sorted(self.collection.distinct("game_name"))

    def get_by_game(self, game_name: str) -> List[Dict[str, Any]]:
        """Return all runs for a given game name."""
        return list(self.collection.find({"game_name": game_name}, {"_id": 0}))

    def get_approximators(self) -> List[str]:
        """Return sorted distinct approximator names."""
        return sorted(self.collection.distinct("approximator_name"))

    def get_by_approximator(self, approximator_name: str) -> List[Dict[str, Any]]:
        """Return all runs that used a given approximator."""
        return list(self.collection.find({"approximator_name": approximator_name}, {"_id": 0}))

    def count_by_config(self, config: RunConfig) -> int:
        """Return the number of runs stored for *config*."""
        return self.collection.count_documents(config.to_dict())