"""Create a portable static comparison page from compatible benchmark results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

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
)
GAME_FIELDS = ("id", "family", "stratum", "n_players", "index", "order")
METADATA_FIELDS = (
    "preparation_hardware",
    "model_profile",
    "model_parameters",
    "refit_protocol",
    "quality_scope",
    "prediction_batch_size",
    "oracle_precision",
    "oracle_validation",
    "signal_ratio_definition",
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
    )

    def instances(rows: list[dict]) -> list[dict]:
        return [{key: row[key] for key in fields if key in row} for row in rows]

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
                    for key in ("id", "family", "dataset", "model_profile", "n_players", "device")
                    if key in row["spec"]
                },
                "reason": row["reason"],
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
    inputs = [json.loads(path.read_text()) for path in paths]
    first = inputs[0]
    panel = ("schema_version", "snapshot_id", "suite", "games", "snapshot_provenance")
    methods, runs, cells = {}, {}, {}
    for result in inputs:
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
                or row["status"] not in ("ok", "failed", "unsupported")
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
    for cell in cells.values():
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
                if key in ("source_sha256", "software_sha256", "private", "factory")
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


def encode_records(records: list[dict]) -> dict:
    """Store lossless columns; string dictionaries never round numeric measurements."""
    columns = {}
    for field in sorted({key for row in records for key in row}):
        values = [row.get(field) for row in records]
        column: dict = {"values": values}
        present = [value for value in values if value is not None]
        if present and all(isinstance(value, str) for value in present):
            dictionary = list(dict.fromkeys(present))
            codes = {value: i for i, value in enumerate(dictionary)}
            column = {"dictionary": dictionary, "values": [codes.get(value) for value in values]}
        missing = [i for i, row in enumerate(records) if field not in row]
        if missing:
            column["missing"] = missing
        columns[field] = column
    return {"codec": "columns-v1", "count": len(records), "columns": columns}


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
            "presets": [p for p in data["presets"] if p["index"] == index and p["order"] == order],
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
    data = {**data, "presets": summarize(data)}
    compact = len(data["records"]) >= 100_000 if compact is None else compact
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "index.html",
        "app.js",
        "records.js",
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
    # Store repeated hardware descriptions once; the browser restores row references.
    workers = dict(data.get("workers", {}))
    worker_ids = {identity(worker): name for name, worker in workers.items()}
    records = []
    for row in data["records"]:
        record = (
            dict(row)
            if compact
            else {key: value for key, value in row.items() if value is not None}
        )
        worker = record.get("worker")
        if worker is not None:
            record.pop("worker")
            key = identity(worker)
            if key not in worker_ids:
                number = len(workers)
                while f"w{number}" in workers:
                    number += 1
                worker_ids[key] = f"w{number}"
            worker_id = worker_ids[key]
            workers[worker_id] = worker
            record["worker_id"] = worker_id
        records.append(record)
    exported = {**data, "workers": workers, "records": records}
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
