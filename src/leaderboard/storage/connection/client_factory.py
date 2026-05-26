"""This module defines a factory function for creating database clients based on environment variables.

This is the main entry point for obtaining a database client instance, the specific type determined via the DatabaseBackend enum and dict mapping. The factory abstracts away the details of which precise client implementation is used, allowing the rest of the codebase to interact only with the abstract DatabaseClient interface.

Provides functionality for:
- Mapping database backend types to their corresponding client classes.
- Creating a database client instance.
"""

from __future__ import annotations

from enum import StrEnum

from .client import DatabaseClient
from .mongo_client import MongoDBClient
from .connection_exceptions import UnsupportedDatabaseBackendError


class DatabaseBackend(StrEnum):
    """Enum for supported database backends."""

    MONGODB = "mongodb"


class DatabaseClientFactory:
    """Factory for creating database clients.

    Examples
    --------
    >>> client = DatabaseClientFactory.from_env(DatabaseBackend.MONGODB)

    Entry point for obtaining a database client instance.
    """

    # Using the registry we keep track of the supported backends and their corresponding client classes. 
    _registry: dict[DatabaseBackend, type[DatabaseClient]] = {
        DatabaseBackend.MONGODB: MongoDBClient,
    }

    @classmethod
    def create_client(cls, backend: DatabaseBackend | str) -> DatabaseClient:
        """Create a database client using environment variables. Uses the registry to determine which client class to instantiate based on the provided backend.

        Parameters
        ----------
        backend:
            The database backend to use.

        Raises
        ------
        UnsupportedDatabaseBackendError
            If *backend* is not a registered backend.
        """

        # Attempt to cast the backend to the enum type if it's provided as a string
        if isinstance(backend, str):
            try:
                backend = DatabaseBackend(backend)
            except ValueError:
                raise UnsupportedDatabaseBackendError(backend, list(cls._registry)) from None

        client_class = cls._registry.get(backend)
        if client_class is None:
            raise UnsupportedDatabaseBackendError(str(backend), list(cls._registry)) from None
        return client_class.from_env()