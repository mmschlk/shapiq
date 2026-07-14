# Data Classes

This module defines the data schema used to represent a single leaderboard run. It currently exposes one class, [`RunConfig`](#runconfig), which is used throughout the [database client](../connection/README.md) and the [CLI](../cli/README.md) to identify, filter, and de-duplicate runs.

This document is organised as follows:
- [Module Overview](#module-overview)
- [`RunConfig`](#runconfig) (data class definition, fields, and methods)
- [Run Object Schema](#run-object-schema) (the full schema of a run document stored in the database)
- [Usage](#usage) (example of how to create a `RunConfig` instance)


## Module Overview

```
├── data_classes
│   ├── README.md
│   ├── __init__.py           # exposes the RunConfig data class
│   └── run_config.py         # defines the RunConfig data class
```

## `RunConfig`

`RunConfig` (defined in [`run_config.py`](run_config.py)) is an immutable (`frozen=True`) dataclass that uniquely identifies a *family* of runs (repetitions of the same approximator run under a different seed). Documents that share the same `RunConfig` are treated as repeated trials and are aggregated together when the leaderboard computes metrics.

### Fields

| Field | Type | Description |
|---|---|---|
| `game_name` | `str` | Name of the benchmark game the run was performed on. |
| `game_params` | `dict[str, Any]` | Additional game-specific parameters (default: `{}`). |
| `n_players` | `int` | Number of players in the game. |
| `approximator_name` | `str` | Name of the Shapley value approximator used. |
| `approximator_params` | `dict[str, Any]` | Additional approximator-specific parameters (default: `{}`). |
| `index` | `str` | The interaction index computed |
| `max_order` | `int` | Maximum interaction order considered. |
| `budget` | `int` | Evaluation budget (number of game calls) allotted to the approximator. |
| `ground_truth_method` | `str` | Method used to compute the ground-truth values the approximation is compared against. |

The `RunConfig` intentionallty excludes run-specific fields (random seed, computed metrics, runtime, hardware, etc.) which are part of the full raw run information, but not part of the identity used for matching/grouping.


### Methods

| Method | Description |
|---|---|
| `to_dict() -> dict[str, Any]` | Serialise the config into a plain dictionary. Since this only contains the config fields (in the same shape they are stored), it can be used directly as a MongoDB query filter, or compared against a stored document's fields. |
| `from_dict(data: dict[str, Any]) -> RunConfig` (classmethod) | Reconstruct a `RunConfig` from a plain dictionary (e.g. a raw run document loaded from storage). Missing `game_params` / `approximator_params` default to `{}`. |
| `__repr__` | A concise, single-line representation used for logging and debugging. |


## Run Object Schema

Each line in the `.jsonl` local store (and each equivalent document in MongoDB / a HuggingFace dataset) represents a single **run** (one execution of one approximator, on one game instance, with one random seed). A run object is a flat JSON dictionary made up of two groups of fields:

- **Config fields**: everything captured by [`RunConfig`](data_classes/README.md#runconfig). These identify *which experiment* was run, and are what `get_by_config`, `delete_by_config`, and `get_unique_configs` match/group on.
- **Run-specific fields**: everything else: the seed, outcome, metrics, timing, environment, and free-text notes for *the particular execution* of that config.

### Fields

| Field | Type | Group | Description |
|---|---|---|---|
| `run_id` | `str` (UUID) | run-specific | Unique identifier for this run. |
| `game_name` | `str` | config | Name of the benchmark game. |
| `game_id` | `str` | run-specific | Identifier of the concrete game *instance* used for this run |
| `game_params` | `dict[str, Any]` | config | Game-specific parameters, e.g. `x` (instance index explained), `class_to_explain`, `model_name`, `imputer`, `normalize`, `verbose`, `random_state`. |
| `n_players` | `int` | config | Number of players (features) in the game. |
| `approximator_name` | `str` | config | Name of the Shapley/interaction-value approximator used. |
| `approximator_params` | `dict[str, Any]` | config | Approximator-specific parameters (empty `{}` if defaults were used). |
| `shapiq_version` | `str` | run-specific | Version (or dev build) of the `shapiq` package the run was executed with. |
| `index` | `str` | config | Interaction index computed. |
| `max_order` | `int` | config | Maximum interaction order considered. |
| `budget` | `int` | config | Evaluation budget (number of game calls) allotted to the approximator. |
| `approx_seed` | `int` | run-specific | Random seed for the particular run *instance*. |
| `ground_truth_method` | `str` | config | Method used to compute the ground-truth values the approximation is scored against |
| `run_failed` | `bool` | run-specific | Whether the run raised an error. |
| `error_message` | `str \| null` | run-specific | Captured error message if `run_failed` is `true`; `null` otherwise. |
| `metrics` | `dict[str, float]` | run-specific | Nested dict of computed accuracy metrics (see [metrics documentation](../../metrics/README.md)) |
| `runtime_seconds` | `float` | run-specific | Wall-clock time taken by the approximator for this run. |
| `timestamp` | `str` (ISO 8601) | run-specific | UTC timestamp of when the run completed. |
| `hardware` | `dict[str, Any]` | run-specific | Execution environment info: `cpu`, `ram_gb` (nullable), `python_version`. |
| `notes` | `str` | run-specific | Free-text notes field (often empty). |

### Example

```json
{
  "run_id": "2a7a3ba2-c7c2-4555-a1c8-d0c91c898a8b",
  "game_name": "Nursery",
  "game_id": "Nursery_LocalExplanation_Game_29825138",
  "game_params": {
    "x": 0,
    "class_to_explain": null,
    "model_name": "decision_tree",
    "imputer": "marginal",
    "normalize": true,
    "verbose": false,
    "random_state": 42
  },
  "n_players": 8,
  "approximator_name": "OwenSamplingSV",
  "approximator_params": {},
  "shapiq_version": "1.4.2.dev242+gc08b606cc",
  "index": "SV",
  "max_order": 1,
  "budget": 200,
  "approx_seed": 2,
  "ground_truth_method": "ExactComputer",
  "run_failed": false,
  "error_message": null,
  "metrics": {
    "mse": 0.00017300373913590924,
    "mae": 0.009756076388888858,
    "mse_normalized": 0.0050934584934776055,
    "spearman": 0.9761904761904763,
    "kendall_tau": 0.9285714285714285,
    "precision_at_k": 1.0
  },
  "runtime_seconds": 0.4674947079984122,
  "timestamp": "2026-06-16T23:40:39.952547+00:00",
  "hardware": { "cpu": "arm", "ram_gb": null, "python_version": "3.13.5" },
  "notes": ""
}
```

`RunConfig.from_dict(document)` extracts only the **config fields** listed above (dropping everything else). A more in depth description of the configuration fields and their meaning can be found in the [config documentation](../../config_manager/README.md).




## Usage

```python
from leaderboard.storage.data_classes import RunConfig

config = RunConfig(
    game_name="BikeSharing",
    n_players=10,
    approximator_name="KernelSHAPIQ",
    index="k-SII",
    max_order=2,
    budget=250,
    ground_truth_method="exact",
)
```

Because `RunConfig` is a frozen dataclass, instances are hashable and comparable by value, which is what allows the [`DatabaseClient`](../connection/README.md) to de-duplicate and group runs by configuration (e.g. in `get_unique_configs()`).
