"""This module defines an abstract database client class.

Provides functionality for:
- Connecting to a database using parameters from environment variables.
- Inserting, querying, and deleting run records in the database.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self

from leaderboard.storage.connection.utilities import _process

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
        return _process(raw)

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
    def insert_many(self, documents: list[dict[str, Any]]) -> None:
        """Bulk-insert run documents (no-op for an empty list)."""

    # Delete

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
