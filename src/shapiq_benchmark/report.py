"""Create a portable static comparison page from compatible benchmark results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

from shapiq_benchmark.duplicates import remove_aliases
from shapiq_benchmark.order_metrics import order_scores
from shapiq_benchmark.record_store import RecordStore
from shapiq_benchmark.results_io import read_results
from shapiq_benchmark.runner import identity
from shapiq_benchmark.summary import summarize

SITE_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "site"

ROW_FIELDS = (
    "game_id",
    "method",
    "budget",
    "seed",
    "status",
    "nmse",
    "order_scores",
    "mse",
    "queries",
    "requested_queries",
    "seconds",
    "cache_lookup_seconds",
    "estimated_oracle_seconds",
    "estimated_uncached_seconds",
    "timing_scope",
    "official_timing",
    "zero_truth_energy",
    "wall_seconds",
    "timing_profile",
    "worker",
    "error_type",
    "failure_reason",
    "minimum_budget",
    "duplicate_of",
)
GAME_FIELDS = ("id", "family", "stratum", "n_players", "index", "order")
METADATA_FIELDS = (
    "quality_protocol",
    "model_validation_gate",
    "game_quality",
    "imputation_stability",
    "preparation_hardware",
    "model_profile",
    "model_parameters",
    "refit_protocol",
    "quality_scope",
    "prediction_batch_size",
    "oracle_precision",
    "oracle_validation",
    "signal_ratio_definition",
    "payoff_std",
    "payoff_range_upper_bound",
    "ensemble_members",
    "tree_rounds",
    "training_profile",
    "training_rows",
    "validation_rows",
    "test_rows",
    "quality",
    "structure",
    "fourier_spectrum",
    "dataset_source",
    "dataset_source_url",
    "dataset_target_note",
    "dataset_preprocessing",
    "output_class",
    "classes",
    "model_key",
    "model_artifact_sha256",
    "model_source_sha256",
    "model_packages",
    "best_iteration",
    "split_rules",
    "dataset",
    "data_sha256",
    "model",
    "test_r2",
    "background_size",
    "semantics",
    "truth_method",
    "test_accuracy",
    "active_players",
    "player_unit",
    "small_validation_players",
    "small_validation_max_error",
    "model_sha256",
    "n_neighbors",
    "nonzero_coefficients",
    "active_players_definition",
    "class_index",
    "output_scale",
    "point_label",
    "cluster_id",
    "zero_truth_energy",
    "synthetic",
    "stochastic_frozen",
    "case_id",
    "game_kind",
    "feature_indices",
    "feature_names",
    "row_selection",
    "instance_seed",
    "replicate_unit",
    "input_id",
    "class",
    "oracle_cost_protocol",
    "evaluation_timing",
    "score_eligible",
    "signal_ratio",
    "score_exclusion_reason",
)


def budget_failure(row: dict) -> dict:
    """Recognize documented budget guards without publishing exception messages."""
    if row["status"] != "failed":
        return {}
    method, error = row["method"], row.get("error")
    known_methods = {"OddSHAP", "ShaplEIG", "SPEX", "ProxySPEX"}
    if not error:
        # A reproduction bundle has already removed the original exception text.
        if method not in known_methods or row.get("failure_reason") != "insufficient_budget":
            return {}
        details = {"failure_reason": "insufficient_budget"}
        minimum = row.get("minimum_budget")
        if method in {"OddSHAP", "ShaplEIG"} and type(minimum) is int and minimum > row["budget"]:
            details["minimum_budget"] = minimum
        return details
    minimum = None
    if method == "OddSHAP":
        match = re.fullmatch(
            r"ValueError: The budget is too small for OddSHAP\. Received budget=(\d+), "
            r"but at least (\d+) evaluations are required\. Please increase the budget\.",
            error,
        )
        if match and int(match[1]) == row["budget"] < int(match[2]):
            minimum = int(match[2])
        else:
            return {}
    elif method == "ShaplEIG":
        match = re.fullmatch(
            r"ValueError: Budget \((\d+)\) must exceed the initial design size \((\d+)\)\.",
            error,
        )
        if match and int(match[1]) == row["budget"] <= int(match[2]):
            minimum = int(match[2]) + 1
        else:
            return {}
    elif method == "SPEX":
        if error != (
            "ValueError: Insufficient budget to compute the transform. "
            "Increase the budget or use a different approximator."
        ):
            return {}
    elif method == "ProxySPEX":
        match = re.fullmatch(
            r"ValueError: Cannot have number of splits n_splits=5 greater than "
            r"the number of samples: n_samples=([1-4])\.",
            error,
        )
        if not match:
            return {}
    else:
        return {}
    return {
        "failure_reason": "insufficient_budget",
        **({"minimum_budget": minimum} if minimum is not None else {}),
    }


def public_preparation(suite: dict) -> dict:
    """Keep preparation exclusions visible without private pilot diagnostics."""
    if "preparation_preflight" not in suite:
        return {}
    fields = (
        "seed",
        "status",
        "reason",
        "error_type",
        "setup_seconds",
        "oracle_seconds",
        "projected_seconds",
        "uniform_coalitions",
        "projected_constructions",
        "validation_coalitions",
        "imputation_stability",
        "model_validation_gate",
        "peak_rss_bytes",
        "actual_preparation_seconds",
        "artifact_bytes",
        "metadata_bytes",
    )

    def instances(rows: list[dict]) -> list[dict]:
        public = []
        for row in rows:
            entry = {key: row[key] for key in fields if key in row}
            if row.get("reason") in (
                "unstable_imputation",
                "model_not_better_than_validation_dummy",
            ):
                keys = {
                    "policy",
                    "metric",
                    "model_loss",
                    "dummy_loss",
                    "passed",
                    "protocol",
                    "scope",
                    "coalitions",
                    "repeats_per_level",
                    "levels",
                    "sample_size_drift_ratio",
                    "maximum_noise_ratio",
                    "status",
                    "limitation",
                }
                entry["details"] = {
                    key: value for key, value in row.get("details", {}).items() if key in keys
                }
            public.append(entry)
        return public

    preflight = suite["preparation_preflight"]
    return {
        "preparation_preflight": {
            **{
                key: preflight[key]
                for key in (
                    "version",
                    "safety_factor",
                    "scope",
                    "uncertainty",
                    "selection_rule",
                    "maximum_seconds_per_instance",
                    "pilot_timeout_seconds",
                    "structured_references",
                    "structured_timeout_seconds",
                    "structured_memory_gb",
                )
                if key in preflight
            },
            "families": [
                {"id": row["id"], "status": row["status"], "instances": instances(row["instances"])}
                for row in preflight.get("families", [])
            ],
        },
        "preparation_exclusions": [
            {
                "spec": {
                    key: row["spec"][key]
                    for key in (
                        "id",
                        "family",
                        "dataset",
                        "model_profile",
                        "n_players",
                        "device",
                        "oracle",
                        "index",
                        "order",
                        "quality_protocol",
                    )
                    if key in row["spec"]
                },
                "reason": row["reason"],
                **({"kind": row["kind"]} if "kind" in row else {}),
                "maximum_seconds_per_instance": row["maximum_seconds_per_instance"],
                "instances": instances(row["instances"]),
            }
            for row in suite.get("preparation_exclusions", [])
        ],
    }


def merge_results(paths: list[Path]) -> dict:
    """Join identical panels, rejecting conflicting methods and repeated run cells.

    A whitelist excludes truth, coefficients, artifact paths, and exception text
    from the shareable report. Candidate source code is never copied.
    """
    if not paths:
        message = "At least one result file is required."
        raise ValueError(message)
    first = read_results(paths[0])
    panel = ("schema_version", "snapshot_id", "suite", "games", "snapshot_provenance")
    methods, runs, cells = {}, {}, {}
    for position, path in enumerate(paths):
        result = first if position == 0 else read_results(path)
        if result.get("schema_version") != 1 or any(result[key] != first[key] for key in panel):
            message = "Results must have the same snapshot, games, suite, and snapshot provenance."
            raise ValueError(message)
        run_id = identity(result)
        runs[run_id] = result["run_provenance"]
        for name, metadata in result["methods"].items():
            if name in methods and methods[name] != metadata:
                message = f"Conflicting source or metadata for method {name}."
                raise ValueError(message)
            methods[name] = metadata
        game_ids = {game["id"] for game in result["games"]}
        for row in result["records"]:
            if (
                row["game_id"] not in game_ids
                or row["method"] not in result["methods"]
                or row["budget"]
                not in result["suite"]
                .get("budgets_by_game", {})
                .get(row["game_id"], result["suite"]["budgets"])
                or row["seed"] not in result["suite"]["seeds"]
                or row["status"] not in ("ok", "failed", "unsupported", "duplicate")
            ):
                message = "Result cell is outside its declared panel."
                raise ValueError(message)
            for field in (
                "nmse",
                "mse",
                "seconds",
                "queries",
                "requested_queries",
                "cache_lookup_seconds",
                "estimated_oracle_seconds",
                "estimated_uncached_seconds",
            ):
                value = row.get(field)
                if value is not None and (
                    not isinstance(value, int | float) or not math.isfinite(value) or value < 0
                ):
                    message = f"Invalid numeric result: {field}."
                    raise ValueError(message)
            if row["status"] == "ok" and row.get("mse") is None:
                message = "Successful records must include a finite MSE."
                raise ValueError(message)
            if row["status"] == "duplicate" and (
                not isinstance(row.get("duplicate_of"), str)
                or row["duplicate_of"] == row["game_id"]
            ):
                message = "Duplicate results must identify a different canonical game."
                raise ValueError(message)
            key = tuple(row[field] for field in ("game_id", "method", "budget", "seed"))
            cell = {"record": row, "run_id": run_id}
            if key in cells:
                previous = cells[key]
                if (
                    previous["record"] != row
                    or runs[previous["run_id"]] != result["run_provenance"]
                ):
                    message = f"Conflicting duplicate result cell: {key}."
                    raise ValueError(message)
            else:
                cells[key] = cell
    raw_games = {game["id"]: game for game in first["games"]}
    truths = {
        game["id"]: dict(
            zip(map(tuple, game["truth"]["coordinates"]), game["truth"]["values"], strict=True)
        )
        for game in first["games"]
        if "coordinates" in game["truth"]
    }
    for cell in cells.values():
        row = cell["record"]
        encoded = row.get("estimate", {})
        if row["status"] == "ok" and row["game_id"] in truths and "coordinates" in encoded:
            prediction = dict(
                zip(map(tuple, encoded["coordinates"]), encoded["values"], strict=True)
            )
            cell["record"] = {
                **row,
                "order_scores": order_scores(
                    truths[row["game_id"]], prediction, raw_games[row["game_id"]]
                ),
            }
        if "order_scores" in cell["record"]:
            # Truth energy and signal eligibility live once in game metadata.
            # Whitelist nested fields too; legacy sanitized imports need no coefficients.
            cleaned = {}
            for degree, values in cell["record"]["order_scores"].items():
                if degree not in {str(i) for i in range(1, raw_games[row["game_id"]]["order"] + 1)}:
                    message = "Invalid score order."
                    raise ValueError(message)
                cleaned[degree] = {key: values[key] for key in ("nmse", "mse")}
                if any(
                    value is not None
                    and (
                        not isinstance(value, int | float) or not math.isfinite(value) or value < 0
                    )
                    for value in cleaned[degree].values()
                ):
                    message = "Invalid order-specific score."
                    raise ValueError(message)
            cell["record"]["order_scores"] = cleaned
        details = budget_failure(cell["record"])
        cell["record"] = {
            key: value
            for key, value in cell["record"].items()
            if key not in ("failure_reason", "minimum_budget")
        }
        cell["record"].update(details)
        error = cell["record"].get("error")
        if error:
            prefix = error.split(":", 1)[0]
            cell["record"] = {
                **cell["record"],
                "error_type": prefix if prefix.isidentifier() else "EstimatorError",
            }
    games = [
        {
            **{key: game[key] for key in GAME_FIELDS},
            "metadata": {
                **{
                    key: value
                    for key, value in game.get("metadata", {}).items()
                    if key in METADATA_FIELDS
                },
                "zero_truth_energy": game["truth"].get("energy") == 0,
                **(
                    {"order_scores": order_scores(truths[game["id"]], truths[game["id"]], game)}
                    if game["id"] in truths
                    else {}
                ),
            },
        }
        for game in first["games"]
    ]
    return {
        "schema_version": 1,
        "snapshot_id": first["snapshot_id"],
        "snapshot_provenance": first["snapshot_provenance"],
        "suite": {
            **{
                key: first["suite"][key]
                for key in (
                    "name",
                    "protocol",
                    "phase_plan",
                    "min_players",
                    "min_signal_ratio",
                    "matrix_definition",
                    "matrix_coverage",
                    "budgets",
                    "seeds",
                    "game_seeds",
                    "methods",
                    "method_parameters",
                    "budgets_by_game",
                    "relative_budgets",
                )
                if key in first["suite"]
            },
            **public_preparation(first["suite"]),
        },
        "games": games,
        "coverage": first.get("coverage", []),
        "methods": {
            name: {
                key: value
                for key, value in metadata.items()
                if key in ("source_sha256", "software_sha256", "private", "factory", "parameters")
            }
            for name, metadata in methods.items()
        },
        "runs": runs,
        "records": [
            {
                **{key: cell["record"][key] for key in ROW_FIELDS if key in cell["record"]},
                "run_id": cell["run_id"],
            }
            for cell in cells.values()
        ],
    }


def encode_records(records: list[dict], *, nested: bool = False) -> dict:
    """Store lossless columns, optionally sharing repeated preset objects as well."""
    columns = {}
    for field in sorted({key for row in records for key in row}):
        values = [row.get(field) for row in records]
        column: dict = {"values": values}
        present = [value for value in values if value is not None]
        if present and all(isinstance(value, str) for value in present):
            dictionary = list(dict.fromkeys(present))
            codes = {value: i for i, value in enumerate(dictionary)}
            column = {"dictionary": dictionary, "values": [codes.get(value) for value in values]}
        elif nested and present and all(isinstance(value, dict | list) for value in present):
            keys = [json.dumps(value, separators=(",", ":"), allow_nan=False) for value in values]
            unique = list(dict.fromkeys(keys))
            codes = {key: i for i, key in enumerate(unique)}
            candidate = {
                "dictionary": [json.loads(key) for key in unique],
                "values": [codes[key] for key in keys],
            }
            if len(json.dumps(candidate, separators=(",", ":"))) < len(
                json.dumps(column, separators=(",", ":"))
            ):
                column = candidate
        missing = [i for i, row in enumerate(records) if field not in row]
        if missing:
            column["missing"] = missing
        columns[field] = column
    return {
        "codec": "columns-v2" if nested else "columns-v1",
        "count": len(records),
        "columns": columns,
    }


def compact_workers(data: dict, *, omit_null: bool = False) -> dict:
    """Share worker placement while keeping changing diagnostics in scalar columns."""
    workers, worker_ids, records = {}, {}, []
    for row in data["records"]:
        record = {key: value for key, value in row.items() if not omit_null or value is not None}
        worker = record.get("worker")
        if worker is None and record.get("worker_id"):
            worker = data["workers"][record["worker_id"]]
        if worker is not None:
            record.pop("worker", None)
            worker = dict(worker)
            for field in ("peak_rss_bytes", "process_cpu_seconds"):
                if field in worker:
                    record[f"worker_{field}"] = worker.pop(field)
            key = identity(worker)
            worker_id = worker_ids.setdefault(key, f"w{len(worker_ids)}")
            workers[worker_id] = worker
            record["worker_id"] = worker_id
        records.append(record)
    return {**data, "workers": workers, "records": records}


def _write_json(path: Path, value: dict) -> str:
    payload = (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _shard_report(data: dict, output: Path) -> dict:
    """Keep one explanation target's records and globally computed presets per asset."""
    games = {game["id"]: game for game in data["games"]}
    groups = {(game["index"], game["order"]): [] for game in data["games"]}
    capabilities = {}
    for row in data["records"]:
        game = games[row["game_id"]]
        key = (game["index"], game["order"])
        groups.setdefault(key, []).append(row)
        target = f"{key[0]} · order {key[1]}"
        status = "unsupported" if row["status"] == "unsupported" else "supported"
        capabilities.setdefault(row["method"], {}).setdefault(status, set()).add(target)
    shards = []
    for (index, order), records in groups.items():
        if not re.fullmatch(r"[A-Za-z-]+", index) or type(order) is not int or order < 1:
            message = "Invalid explanation target for record export."
            raise ValueError(message)
        filename = f"records-{index.lower()}-{order}.json"
        target = f"{index} · order {order}"
        payload = {
            **encode_records(records),
            "snapshot_id": data["snapshot_id"],
            "target": target,
            "presets": encode_records(
                [p for p in data["presets"] if p["index"] == index and p["order"] == order],
                nested=True,
            ),
        }
        checksum = _write_json(output / filename, payload)
        shards.append(
            {"target": target, "file": filename, "sha256": checksum, "count": len(records)}
        )
    return {
        **data,
        "records": [],
        "presets": [],
        "record_shards": shards,
        "record_count": len(data["records"]),
        "evaluated_count": sum(row["status"] != "unsupported" for row in data["records"]),
        "method_targets": {
            method: {status: sorted(targets) for status, targets in statuses.items()}
            for method, statuses in capabilities.items()
        },
    }


def report(paths: list[Path], output: Path, *, public: bool = False) -> dict:
    """Validate a single snapshot and write its portable report."""
    return write_report(merge_results(paths), output, public=public)


def write_report(
    data: dict, output: Path, *, public: bool = False, compact: bool | None = None
) -> dict:
    """Write a sanitized single-snapshot or authenticated composite report.

    Large reports load one explanation target at a time. Small reports retain
    their standalone JSON format, including private local candidate workflows.
    """
    if isinstance(data["records"], RecordStore):
        message = "Select a bounded panel before writing disk-backed records."
        raise TypeError(message)
    assets = SITE_DIR
    public = public or output.resolve() == assets.resolve()
    if public and any(method.get("private", True) for method in data["methods"].values()):
        message = "Public reports cannot include private candidate methods."
        raise ValueError(message)
    if public and (
        any({"truth", "artifact"} & game.keys() for game in data["games"])
        or any({"estimate", "error"} & row.keys() for row in data["records"])
    ):
        message = "Public report writing requires sanitized games and records."
        raise ValueError(message)
    if data.get("record_shards"):
        message = "Load record shards before writing another report."
        raise ValueError(message)
    aliases = {
        row["game_id"]: row["duplicate_of"]
        for row in data["records"]
        if row["status"] == "duplicate"
    }
    if aliases:
        remove_aliases(data, aliases)
    degrees = sorted(
        {
            int(degree)
            for game in data["games"]
            if game["order"] > 1
            for degree in game.get("metadata", {}).get("order_scores", {})
        }
    )
    controls = any(
        game.get("metadata", {}).get("game_quality", {}).get("role") == "control"
        for game in data["games"]
    )
    data = {
        **data,
        "presets": [
            preset
            for included in ([False, True] if controls else [False])
            for degree in [None, *degrees]
            for preset in summarize(data, score_order=degree, include_controls=included)
        ],
    }
    compact = len(data["records"]) >= 100_000 if compact is None else compact
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "index.html",
        "app.js",
        "records.js",
        "partitions.js",
        "partition-details.js",
        "query.js",
        "query-worker.js",
        "partition-client.js",
        "partition-download.js",
        "partition-about.js",
        "charts.js",
        "style.css",
        "shapiq.svg",
        "methods.js",
        "protocol.js",
        "about.html",
        "about.css",
        "about.js",
    ):
        source, destination = assets / name, output / name
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    exported = compact_workers(data, omit_null=not compact)
    if compact:
        exported = _shard_report(exported, output)
    _write_json(output / "data.json", exported)
    current_shards = {item["file"] for item in exported.get("record_shards", [])}
    for path in output.glob("records-*.json"):
        if (
            re.fullmatch(r"records-[a-z-]+-\d+\.json", path.name)
            and path.name not in current_shards
        ):
            path.unlink()  # Old target assets may contain a previous local candidate.

    # The guide needs game provenance, not the much larger evaluation records.
    (output / "about.json").write_text(
        json.dumps(
            {
                key: exported[key]
                for key in (
                    "schema_version",
                    "snapshot_id",
                    "snapshot_provenance",
                    "suite",
                    "games",
                )
            },
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    return data


def main() -> None:
    """Build a static report from one or more compatible result files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public", action="store_true", help="Reject private candidate results")
    args = parser.parse_args()
    report(args.results, args.output, public=args.public)


if __name__ == "__main__":
    main()
