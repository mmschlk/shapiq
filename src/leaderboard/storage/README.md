# The Storage Module

This module handles connectivity and storage for the Shapiq Living Benchmark Leaderboard. It defines the [database client](#database-client), that is then used to interact with the chosen database in order to store and retrieve [data](#database-objects) about the approximator runs in (or set to populate) the leaderboard. Additionally, this module provides a [command line interface](#command-line-interface-for-storage-interactions) to interact with (and synchronize) the defined storage systems directly from the command line.

We support the following storage systems:
- MongoDB (via the `pymongo` library)
- Local file storage (via the `json` library, through the use of locally stored `JSONL` files)
- HuggingFace Datasets (via the `huggingface_hub` library)

The implementation of the client allows for easy extension to other storage systems, as is explained in the [Database Client](#database-client) section.


## Module Overview

The module is structured as follows:

```
.
├── README.md
├── __init__.py                        # exposes the DatabaseClient and defined connectivity errors
├── cli
│   ├── README.md                      # usage guide and documentation for the CLI
│   ├── __init__.py           
│   ├── cli.py                         # entry point for the CLI
│   ├── formatting.py                  # formatting utilities for the CLI
│   ├── query_context.py               # context manager for the `seq` command of the CLI
│   ├── registry.py                    # registry for the available storage backends
│   └── repl.py                        # defines CLI grammar (commands, arguments, and options)
├── connection
│   ├── README.md                      # documentation for the database client
│   ├── __init__.py                    # exposes the database client, factory, and defined connectivity errors       
│   ├── client.py                      # defines the abstract base class for the database client
│   ├── client_factory.py              # factory for creating database clients based on provided variables
│   ├── connection_exceptions.py       # defines the exceptions raised by the database client
│   ├── huggingface_client.py          # concrete implementation of the database client for HuggingFace Datasets
│   ├── local_client.py                # concrete implementation of the database client for local file storage
│   ├── mongo_client.py                # concrete implementation of the database client for MongoDB
│   └── utilities.py                   # utility functions for the database client
└── data_classes
    ├── README.md                      # documentation for the data schema used to represent a leaderboard run
    ├── __init__.py                    # exposes the RunConfig data class
    └── run_config.py                  # defines the RunConfig data class, which represents the object stored for an approximator run
```


## Usage

The storage module introduces additional dependencies to the `pymongo` and `huggingface_hub` libraries. These libraries are included in the default installation of the leaderboard component, so no additional installation steps are required.

We describe the usage of the CLI (with relevant examples) in the [CLI README](cli/README.md). The database client (and the factory required for its creation) are exposed in the [`__init__.py`](connection/__init__.py) file, and can be instantiated and used as follows:

```python
from pathlib import Path

# Create a local database client
output_path = Path("data") / "data.jsonl"
local_db = DatabaseClientFactory.create_client(
    "local", db_args={"LOCAL_DB_PATH": str(output_path)}
)
```

By default, the factory will make use of the environment variables defined in the `.env` file to create the database client. The `db_args` argument can be used to override these variables, as shown in the example above. The following environment variables are needed to configure the database client for the different storage systems:
- MongoDB: `MONGODB_URI` and `MONGODB_DB`
- HuggingFace Datasets: `HF_DATASET`, `HF_FILE`, and `HF_TOKEN`
- Local file storage: `LOCAL_DB_PATH`

## Functionality

### Database Objects

Every stored record represents a single run of an approximator against a benchmark game, keyed by an immutable [`RunConfig`](data_classes/README.md#runconfig) (game, number of players, approximator, index, max order, budget, ground-truth method, plus any game/approximator-specific parameters). Two runs that share a `RunConfig` differ only in their random seed and are treated as repetitions of the same experiment for metric aggregation. Full field-level documentation, along with serialisation helpers (`to_dict` / `from_dict`), lives in the [`data_classes` README](data_classes/README.md).

### Database Client

The [`DatabaseClient`](connection/README.md) abstract base class defines a single storage API point, that hides the underlying storage implementation and is able to handle:
- connectivity, 
- inserts (including duplicate-safe inserts), 
- deletes, 
- reads.

Concrete implementations (`MongoDBClient`, `LocalClient`, `HuggingFaceClient`) are created through the [`DatabaseClientFactory`](connection/README.md#databaseclientfactory) shown in the [Usage](#usage) section above, allowing application code to function through a unified interface. Full method reference, exception hierarchy, and how to register additional backends is documented in the [connection README](connection/README.md).

### Command Line Interface for Storage Interactions

The [Storage CLI](cli/README.md) exposes the database client interactively from the terminal. It is meant to assist with the synchronization of different backends, so it supports opening several storage connections at once (each identified through a short ID like `local1` or `mongodb1`), exploring a connection's contents with the chained `seq` query language, and synchronizing data between backends via `insert` and `delete`. See the [CLI README](cli/README.md) for the full command grammar, a walkthrough with examples, and the precise implementation details.

