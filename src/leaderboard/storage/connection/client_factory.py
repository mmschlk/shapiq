"""This module defines a factory function for creating database clients based on environment variables.

This is the main entry point for obtaining a database client instance, the specific type determined via the DatabaseBackend enum and dict mapping. The factory abstracts away the details of which precise client implementation is used, allowing the rest of the codebase to interact only with the abstract DatabaseClient interface.

Provides functionality for:
- Mapping database backend types to their corresponding client classes.
- Creating a database client instance.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from .connection_exceptions import UnsupportedDatabaseBackendError
from .local_client import LocalClient
from .mongo_client import MongoDBClient

if TYPE_CHECKING:
    from .client import DatabaseClient


class DatabaseBackend(StrEnum):
    """Enum for supported database backends."""

    MONGODB = "mongodb"
    LOCAL = "local"


class DatabaseClientFactory:
    """Factory for creating database clients.

    Examples:
    --------
    >>> client = DatabaseClientFactory.create_client(DatabaseBackend.MONGODB)
    >>> client = DatabaseClientFactory.create_client(DatabaseBackend.LOCAL)
    >>> client = DatabaseClientFactory.create_client("local")  # string form also accepted

    Entry point for obtaining a database client instance.
    """

    # Using the registry we keep track of the supported backends and their corresponding client classes.
    _registry: ClassVar[dict[DatabaseBackend, type[DatabaseClient]]] = {
        DatabaseBackend.MONGODB: MongoDBClient,
        DatabaseBackend.LOCAL: LocalClient,
    }

    @classmethod
    def create_client(cls, backend: DatabaseBackend | str, db_args: dict) -> DatabaseClient:
        """Create a database client using environment variables. Uses the registry to determine which client class to instantiate based on the provided backend.

        Parameters
        ----------
        backend:
            The database backend to use. Accepted values are the members of
            ``DatabaseBackend`` or their string equivalents (``"mongodb"``,
            ``"local"``).

        db_args:
            Arguments to pass to the client constructor.
            Uses the arguments from args by default, falls back to environment variables if not provided.
            If environment variables are also not provided, fallbacks to default values

        Raises:
        ------
        UnsupportedDatabaseBackendError
            If *backend* is not a registered backend.
        """
        if isinstance(backend, str):
            try:
                backend = DatabaseBackend(backend)
            except ValueError:
                raise UnsupportedDatabaseBackendError(backend, list(cls._registry)) from None

        client_class = cls._registry.get(backend)
        if client_class is None:
            raise UnsupportedDatabaseBackendError(str(backend), list(cls._registry)) from None
        return client_class.from_env(db_args)
