"""This module defines an abstract database client class.

Provides functionality for:
- Connecting to a database using parameters from environment variables.
- Inserting, querying, and deleting run records in the database.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self

from leaderboard.storage.connection.utilities import process_raw_runs

if TYPE_CHECKING:
    import pandas as pd

    from leaderboard.storage.data_classes import RunConfig


class DatabaseClient(ABC):
    """Abstract base class for database clients.

    Defines the interface for all database backends used to store and
    retrieve shapiq experiment results.
    """

    @classmethod
    @abstractmethod
    def from_env(cls, args: dict) -> Self:
        """Create a DatabaseClient instance using connection parameters from environment variables.

        Args:
            args: Arguments for the database client constructor.
                Order of priority is:
                    1) values in ``args``,
                    2) environment variables,
                    3) default values (if applicable).

        Returns:
            DatabaseClient: An instance of the database client.
        """

    def load_dataframe(self) -> pd.DataFrame:
        """Load all runs, filter failures, flatten metrics, and aggregate.

        Returns a DataFrame ready for the leaderboard UI with columns:
            game_name, approximator_name, budget,
            mse_mean, mse_std, mae_mean, mae_std,
            ground_truth_method,
            runtime_mean, runtime_min, runtime_max,
            n_seeds
        """
        raw = self.get_all()
        return process_raw_runs(raw)

    @abstractmethod
    def test_connection(self) -> bool:
        """Test the connection to the database.

        Returns:
        -------
        bool
            ``True`` if the connection is healthy, ``False`` otherwise.
        """

    # Connection Handling

    @abstractmethod
    def close(self) -> None:
        """Close the underlying database connection."""

    def __enter__(self) -> Self:
        """Enable use as a context manager (with statement)."""
        return self

    def __exit__(self, *_: object) -> None:
        """Ensure the connection is closed when exiting a context."""
        self.close()

    # Write

    @abstractmethod
    def insert_one(self, document: dict[str, Any]) -> None:
        """Insert a single run document."""

    @abstractmethod
    def safe_insert_one(self, document: dict[str, Any], mode: str = "merge") -> bool:
        """Append a single run document only if no existing document matches its config and seed.

        Args:
            document: the run document to insert
            mode: merge / replace / skip (default: merge)
                - merge: if a matching document exists, override metrics that are different and keep the rest; update timestamp to newest document
                - replace: if a matching document exists, replace it entirely with the new document
                - skip: if a matching document exists, do not modify it

        Returns True if the document was inserted, or False if a matching document already exists.
        """

    @abstractmethod
    def insert_many(self, documents: list[dict[str, Any]]) -> None:
        """Bulk-insert run documents (no-op for an empty list)."""

    def safe_insert_many(self, documents: list[dict[str, Any]], mode: str = "merge") -> int:
        """Bulk-insert run documents, ensuring no duplicates for the same config.

        Args:
            documents: the run documents to insert
            mode: merge / replace / skip (default: merge)
                - merge: if a matching document exists, override metrics that are different and keep the rest; update timestamp to newest document
                - replace: if a matching document exists, replace it entirely with the new document
                - skip: if a matching document exists, do not modify it
        """
        if not documents:
            return 0

        inserted_count = 0

        for doc in documents:
            if self.safe_insert_one(doc, mode=mode):
                inserted_count += 1

        return inserted_count

    # Delete

    @abstractmethod
    def delete_by_id(self, doc_id: str) -> int:
        """Delete a document by its unique identifier. Returns 1 if deleted, 0 if not found."""

    @abstractmethod
    def delete_all(self) -> int:
        """Delete every document in the collection.

        Returns:
        -------
        int
            Number of deleted documents.
        """

    @abstractmethod
    def delete_by_config(self, config: RunConfig) -> int:
        """Delete all documents matching *config*.

        Returns:
        -------
        int
            Number of deleted documents.
        """

    # Read - generic

    @abstractmethod
    def get_all(self) -> list[dict[str, Any]]:
        """Return every document (without internal database identifiers)."""

    @abstractmethod
    def get_by_config(self, config: RunConfig) -> list[dict[str, Any]]:
        """Return all documents whose fields match *config*."""

    # Read - domain helpers

    @abstractmethod
    def get_unique_configs(self) -> list[RunConfig]:
        """Return one ``RunConfig`` per unique configuration in the collection."""

    @abstractmethod
    def get_games(self) -> list[str]:
        """Return sorted distinct game names."""

    @abstractmethod
    def get_by_game(self, game_name: str) -> list[dict[str, Any]]:
        """Return all runs for a given game name."""

    @abstractmethod
    def get_approximators(self) -> list[str]:
        """Return sorted distinct approximator names."""

    @abstractmethod
    def get_by_approximator(self, approximator_name: str) -> list[dict[str, Any]]:
        """Return all runs that used a given approximator."""

    @abstractmethod
    def count_by_config(self, config: RunConfig) -> int:
        """Return the number of runs stored for *config*."""
