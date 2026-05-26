# The database client

This module defines the connecting to a database logic. It is built around three components:
- a **factory** (`DatabaseClientFactory`) that creates clients based on environment variables,
- an **abstract client** (`DatabaseClient`) that defines the public API for all backends,
- and one or more **concrete clients** (e.g. `MongoDBClient`) that implement the abstract interface for specific databases.

## Architecture

```
DatabaseClientFactory
        │
        │  .create_client(backend)
        ▼
DatabaseClient  (ABC)
        │
        ├── MongoDBClient
        └── (future backends …)
```

User code depends only on `DatabaseClientFactory` and `DatabaseClient`. Concrete implementations are an internal detail.

---

## Interaction Flow

1. Creation of a client

```python
from leaderboard.storage.connection import DatabaseClientFactory, DatabaseBackend

client = DatabaseClientFactory.create_client(DatabaseBackend.MONGODB)
```

`create_client` reads all required credentials from environment variables (e.g. `MONGODB_URI`) and returns a fully initialised `DatabaseClient`. 

It is also possible to use a string instead of the specially defined `DatabaseBackend` enum, but using the enum is recommended for better type safety and discoverability. Using a string will attempt to cast it to a `DatabaseBackend` value, which will raise an `UnsupportedBackendError` if the string is not valid.


2. Using the client

Regardless of the concrete backend type, the returned object is always a `DatabaseClient`, so all user code can be written against this single, stable interface.

## `DatabaseClient` Interface

The abstract base class defines the full public API that every backend must implement.

### Connection

| Method | Description |
|---|---|
| `from_env() → Self` | Classmethod. Construct the client from environment variables. |
| `test_connection() → bool` | Returns `True` if the backend is reachable. |
| `close() → None` | Close the underlying connection (called automatically by `__exit__`). |

### Write

| Method | Description |
|---|---|
| `insert_one(document)` | Insert a single run document. |
| `insert_many(documents)` | Bulk-insert documents; no-op for an empty list. |

### Delete

| Method | Description |
|---|---|
| `delete_all() → int` | Delete every document; returns deleted count. |
| `delete_by_config(config) → int` | Delete documents matching a `RunConfig`; returns deleted count. |

### Read

| Method | Description |
|---|---|
| `get_all()` | Return every document. |
| `get_by_config(config)` | Return documents matching a `RunConfig`. |
| `get_unique_configs()` | Return one `RunConfig` per unique configuration. |
| `get_games()` | Return sorted list of distinct game names. |
| `get_by_game(game_name)` | Return all runs for a given game. |
| `get_approximators()` | Return sorted list of distinct approximator names. |
| `get_by_approximator(name)` | Return all runs for a given approximator. |
| `count_by_config(config) → int` | Count stored runs matching a `RunConfig`. |

---

## `DatabaseClientFactory`

### Built-in backends

| `DatabaseBackend` value | Concrete class | Required env vars |
|---|---|---|
| `"mongodb"` | `MongoDBClient` | `MONGODB_URI`, `MONGODB_DB` (optional, default `"shapiq-leaderboard"`) |


### Registering a new backend

The factory uses an internal `_registry` dict, which must be populated with the mapping from `DatabaseBackend` values to concrete client classes. 

## Exceptions

All exceptions inherit from `DBClientError` so callers can catch the entire hierarchy with a single `except` clause if needed.

```
DBClientError
├── MissingMongoURIError
├── UnsupportedDatabaseBackendError
└── DBConnectionError
```

| Exception | Raised when |
|---|---|
| `DBClientError` | Base class; catch-all for any connection-layer error. |
| `MissingMongoURIError` | `MONGODB_URI` is absent from the environment when `MongoDBClient.from_env()` is called. |
| `UnsupportedDatabaseBackendError` | The backend string passed to the factory is not registered. Includes the list of supported backends in the message. |
| `DBConnectionError` | The client is initialised but cannot reach the database (e.g. `test_connection()` fails). |
