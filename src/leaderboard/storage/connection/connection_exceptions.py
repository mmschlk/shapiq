"""Exceptions related to database connection errors."""

from __future__ import annotations


class DBClientError(Exception):
    """Base exception for database client errors."""

    def __init__(self) -> None:
        """Initialize with a custom error message."""
        super().__init__("Database client error occurred.")


class MissingMongoURIError(DBClientError):
    """Raised when MONGODB_URI is not set."""

    def __init__(self) -> None:
        """Initialize with a message indicating that MONGODB_URI is missing."""
        super().__init__("MONGODB_URI is not set in the environment.")


class UnsupportedDatabaseBackendError(DBClientError):
    """Raised when an unsupported database backend is specified."""

    def __init__(self, unsupported_backend: str, supported_backends: list[str]) -> None:
        """Initialize with a message indicating an unsupported backend."""
        super().__init__(
            f"Unsupported database backend specified: {unsupported_backend}. Supported backends: {', '.join(supported_backends)}."
        )


class DBConnectionError(DBClientError):
    """Raised when the database client fails to connect to the database."""

    def __init__(self, client: str = "Database") -> None:
        """Initialize with a message indicating a connection failure."""
        super().__init__(f"Failed to connect to {client}")