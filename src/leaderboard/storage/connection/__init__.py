"""This module provides the MongoDBClient class for connecting to a MongoDB database."""

from .client import MongoDBClient
from .connection_exceptions import MissingMongoURIError, MongoDBClientError, MongoDBConnectionError

__all__ = ["MongoDBClient", "MissingMongoURIError", "MongoDBClientError", "MongoDBConnectionError"]
