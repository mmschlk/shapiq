# Storage CLI

We discuss:
1. [Module Overview](#module-overview)
2. [Implementation Details](#implementation-details)
3. [Usage](#usage)

## Module Overview

The module is structured as follows:

```
├── cli
│   ├── __init__.py           
│   ├── cli.py                         # entry point for the CLI
│   ├── formatting.py                  # formatting utilities for the CLI
│   ├── query_context.py               # context manager for the `seq` command of the CLI
│   ├── registry.py                    # registry for the available storage backends
│   └── repl.py                        # defines CLI grammar (commands, arguments, and options)
```

## Implementation Details

The CLI is a small custom Read-Eval-Print Loop (REPL).

Can you shorten this a bit and make it sound more natural as well

| File | Responsibility |
|---|---|
| [`cli.py`](cli.py) | Entry point. Parses top-level startup flags (`--backend`, `--no-color`) and constructs a `StorageREPL` and triggers its run. Through the optional flag `--backend` a connection is opened immediately. Handles `KeyboardInterrupt` by closing all connections before exiting. |
| [`repl.py`](repl.py) | Defines `StorageREPL` (grammar and dispatch loop). Tokenises each input line (`shlex`) and routes it to a `_cmd_*` handler for `list`, `add`, `close`, `insert`, `delete`, and `sequence`/`seq`. Also handles the additional parameters required for defining new connections, creating the client via the [`DatabaseClientFactory`](../connection/README.md#databaseclientfactory). |
| [`registry.py`](registry.py) | Defines `StorageRegistry`, which tracks every open `DatabaseClient` and assigns human-readable IDs (`local1`, `mongodb1`, `huggingface1`, ...). IDs are scoped per-backend-type, so opening two local files gives `local1` and `local2`.|
| [`query_context.py`](query_context.py) | Defines `QueryContext`, which powers the `seq` command described in [Exploring the Store](#3-exploring-the-store). It accumulates a list of `(verb, args)` pairs (`get`, `list`, `count`, `sort`, `show`) as the user types them, and only executes them against the storage once the end of the sequence is received (`eoc`). |
| [`formatting.py`](formatting.py) | Stateless ANSI-color and message-formatting helpers (`bold`, `ok`, `warn`, `error`, `header`, the prompt string, ...) shared by `repl.py`. Color can be disabled globally via `set_color(enabled=False)`, or by enabling the flag `--no-color`. |

### Command grammar

At the top level, the REPL recognises:

```
list storages                              List open connections and their IDs
add <backend>                              Open a new connection (prompts for params)
close <storage_id> | close all             Close one or every open connection

insert [safe] <src> to <dst> [using <mode>]  Transfer documents between storages
delete from <storage_id>                     Interactively delete entries from a storage
delete entries <src> from <dst>              Delete entries found in <src> from <dst>

sequence [<storage_id>]  (alias: seq)        Enter a query sequence (see query_context.py)
help                                          Show the command reference
exit | quit | q                              Close all connections and exit
```

Once inside a `sequence` block, a separate mini-grammar applies (`get`, `list`, `count`, `sort`, `show`, `eoc`, `help`, `abort`) - see the docstring at the top of [`query_context.py`](query_context.py) for the exact semantics of each sub-command.


### Adding a new backend to the CLI

Because the CLI ultimately delegates connection creation to `DatabaseClientFactory` (see the [connection README](../connection/README.md#databaseclientfactory)), wiring up a new storage backend in the CLI only requires:
1. Registering the backend with the factory (as described in the connection README).
2. Adding an entry to `_BACKEND_PARAMS` in [`repl.py`](repl.py) describing which environment-variable-backed parameters the CLI should prompt for.
3. Adding the backend name to the `choices` list for `--backend` in [`cli.py`](cli.py).

## Usage

We present a short tutorial on the usage of the CLI. The CLI is intended to be used for storage management and synchronization between different storage backends.

1. Open CLI
2. Add storage
3. Explore storage through `seq`
4. Storage Operations
	1. Insert
	2. Delete

### 1. Running the CLI

To run the CLI, please run:

```
python3 -m leaderboard.storage.cli.cli
```

Disclaimer. Running this for the first time may take some time.

### 2. Adding a Storage

Available Storage for interaction using the CLI are:
- local (relies on local `.jsonl` files)
- mongodb 
- huggingface

All available results store can be initiated with the default configurations as they are available in the `.env` file. 

>[!Note] 
>Keep in mind that MongoDB connections error can also happen due to the **IP whitelisting** requirement of MongoDB. 


Example (Adding a mongodb backend with default parameters):
<img width="677" height="377" alt="img_add_mongodb" src="https://github.com/user-attachments/assets/b6569838-09e5-4422-9fa8-1aa26357334d" />


Adding a storage will assign it an `id`. All future interactions with said storage happen through the `id`.

>[!Warning]
> It is possible to use multiple backends simultaneously. The application is intended to work with multiple **different** backends (backends are considered different if they do not point towards the same thing). However keep in mind that the application **can** and **will** show unexpected behaviour if the backends are identical.

Active connections are tracked through the `list` command:

<img width="757" height="147" alt="img_track_active_connections" src="https://github.com/user-attachments/assets/0426c473-5e56-40fb-bb4f-45ef1b77cc2a" />

### 3. Exploring the Store

The `seq` command allows for *simple* SQL-like operations on the database. It functions via **chaining** commands, which allows one to define a sequence of filters and list **distinct** values in fields of interest. `eoc` marks the end of a command and triggers running the sequence.

The `help` command runs inside `seq` provides an overview of the abilities of the command sequencer. 
<img width="1238" height="402" alt="img_help_seq" src="https://github.com/user-attachments/assets/ab6ff7f1-640c-481a-90c1-5b5e8dcb9b7d" />

We provide an example of interacting with `seq`. In this example, we want to list all budgets available for the "BikeSharing" game.

<img width="1110" height="511" alt="img_seq_interact" src="https://github.com/user-attachments/assets/fc017e52-0708-4cef-8d32-8c9a62bfb51b" />

>[!Important]
>The `seq` command relies on the exact match between the field to filter on. Using the wrong field name, will still trigger an execution, however nothing meaningful will be returned.


### 4. Storage Operations

The Storage CLI supports two very important storage interaction functionalities, namely **insert** and **delete**. 

#### 4.1. Insertion

Insertion serves as an easy way to synchronize two active storages. The insertion command goes as follows:

`Usage: insert [safe] <src> to <dst> [using <mode>]` 

Simple insertion performs a "copy" operation. It takes ALL the entries from the source storage (provided via its `id`) and copies them to the destination storage (also provided via its `id`). 

For example, to copy all the entries from the `MongoDB` database (with `id = mongodb1`) to a local `.jsonl` file (with `id = local1`) one should run:


```
insert mongodb1 to local1
```

**Safe** insertion allows for the handling of duplicates. This is significantly slower. For safe insertion, three modes are available (default is `merge`):
- `merge`: the "newest" object (the object to be inserted) overrides the common attributes. Attributes that only exist for the "older" (already existing) object are retained. 
- `skip`: if a duplicate object is found, insertion is skipped. 
- `override`: the duplicate object is overridden completely (no retaining of attributes). 


#### 4.2 Deletion

As this tool is intended to be used for storage management, deletion is also supported. 

>[!Warning]
>Objects, once deleted are NOT recoverable. Recommended to create a local snapshot before any deletion operations.

To delete, there exist two options:
- `delete from <id>` : deletes entries from an active storage
- `delete entries <src> from <dst>` deletes entries available in the source storage from a destination storage

The command `delete entries` matches the objects available in one storage to the objects available in the other and deletes them (no confirmation required). Due to this pairwise matching between the two storages deletion speed depends on the number of comparisons to be done.

The command `delete from <id>` allows for more customization. Options available are: `all` and `by config`. Deleting everything requires additional confirmation. To delete by config, a logic similar to `seq` is used. Field names and values should be provided exactly as they appear in the database. 

Example (Deleting budget 100 runs under BikeSharing):

<img width="1055" height="362" alt="img_del_example" src="https://github.com/user-attachments/assets/f55dec3f-86bc-4b63-95b2-661d18bd1792" />


## Observations

The Storage CLI is intended as a synchronization between local and MongoDB storages. HuggingFace datasets is also supported to the `huggingface` backend, however any operations done on the HuggingFace dataset will be done on a temporary local representation of it and will **not** affect the dataset available on HuggingFace.
