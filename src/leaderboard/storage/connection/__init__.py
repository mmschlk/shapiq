"""This module provides the MongoDBClient class for connecting to a MongoDB database."""

from .client import DatabaseClient
from .client_factory import DatabaseBackend, DatabaseClientFactory
from .connection_exceptions import (
    DBClientError,
    DBConnectionError,
    MissingMongoURIError,
    UnsupportedDatabaseBackendError,
)

__all__ = [
    "DatabaseBackend",
    "DatabaseClientFactory",
    "DatabaseClient",
    "DBClientError",
    "MissingMongoURIError",
    "DBConnectionError",
    "UnsupportedDatabaseBackendError",
]
