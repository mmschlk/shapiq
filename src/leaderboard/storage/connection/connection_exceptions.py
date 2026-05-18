"""Exceptions related to database connection errors."""

from __future__ import annotations


class MongoDBClientError(Exception):
    """Base exception for MongoDB client errors."""

    def __init__(self) -> None:
        """Initialize with a custom error message."""
        super().__init__("MongoDB client error occurred.")


class MissingMongoURIError(MongoDBClientError):
    """Raised when MONGODB_URI is not set."""

    def __init__(self) -> None:
        """Initialize with a message indicating that MONGODB_URI is missing."""
        super().__init__("MONGODB_URI is not set in the environment.")
