"""This module defines the MongoDBClient class.

Provides functionality for:
- Connecting to a MongoDB database using parameters from environment variables.
- Inserting, querying, and deleting run records in the database.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Self

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import OperationFailure

from leaderboard.storage.data_classes import RunConfig

from .connection_exceptions import MissingMongoURIError
from .client import DatabaseClient

if TYPE_CHECKING:
    from pymongo.collection import Collection
    from pymongo.database import Database


def load_env() -> tuple[str, str]:
    """Load MongoDB connection parameters from environment variables."""
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "shapiq-leaderboard")
    if not uri:
        raise MissingMongoURIError from None
    return uri, db_name


class MongoDBClient(DatabaseClient):
    """MongoDB client for shapiq experiment results.

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

    def __init__(self, uri: str, db_name: str = "shapiq", collection_name: str = "runs") -> None:
        """Initialize the MongoDB client and connect to the specified database and collection."""
        self._client: MongoClient = MongoClient(uri)
        self._db: Database = self._client[db_name]
        self.collection: Collection = self._db[collection_name]

    @classmethod
    def from_env(cls) -> MongoDBClient:
        """Create a MongoDBClient instance using connection parameters from environment variables.

        Returns: MongoDBClient.
        """
        uri, db_name = load_env()
        return cls(uri=uri, db_name=db_name)

    def test_connection(self) -> bool:
        """Test the connection to MongoDB by sending a ping command."""
        try:
            self._client.admin.command("ping")
        except (ConnectionError, OperationFailure):
            return False
        else:
            return True

    # Connection Handling

    def close(self) -> None:
        """Close the underlying MongoDB connection."""
        self._client.close()

    def __enter__(self) -> Self:
        """Enable use as a context manager (with statement)."""
        return self

    def __exit__(self, *_: object) -> None:
        """Ensure the connection is closed when exiting a context."""
        self.close()

    # Write

    def insert_one(self, document: dict[str, Any]) -> None:
        """Insert a single run document."""
        self.collection.insert_one(document)

    def insert_many(self, documents: list[dict[str, Any]]) -> None:
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

    # Read - generic

    def get_all(self) -> list[dict[str, Any]]:
        """Return every document (without ``_id``)."""
        return list(self.collection.find({}, {"_id": 0}))

    def get_by_config(self, config: RunConfig) -> list[dict[str, Any]]:
        """Return all documents whose fields match *config*."""
        return list(self.collection.find(config.to_dict(), {"_id": 0}))

    # Read - domain helpers

    def get_unique_configs(self) -> list[RunConfig]:
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

    def get_games(self) -> list[str]:
        """Return sorted distinct game names."""
        return sorted(self.collection.distinct("game_name"))

    def get_by_game(self, game_name: str) -> list[dict[str, Any]]:
        """Return all runs for a given game name."""
        return list(self.collection.find({"game_name": game_name}, {"_id": 0}))

    def get_approximators(self) -> list[str]:
        """Return sorted distinct approximator names."""
        return sorted(self.collection.distinct("approximator_name"))

    def get_by_approximator(self, approximator_name: str) -> list[dict[str, Any]]:
        """Return all runs that used a given approximator."""
        return list(self.collection.find({"approximator_name": approximator_name}, {"_id": 0}))

    def count_by_config(self, config: RunConfig) -> int:
        """Return the number of runs stored for *config*."""
        return self.collection.count_documents(config.to_dict())
