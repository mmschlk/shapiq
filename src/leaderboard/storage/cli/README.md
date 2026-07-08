# Storage CLI

A short tutorial on the interactions with the storage cli (now with images)

## Usage

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
