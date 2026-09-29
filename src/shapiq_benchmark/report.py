"""Create a portable static comparison page from compatible benchmark results."""

from __future__ import annotations

import argparse
import json
import math
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
    "timing_scope",
    "official_timing",
    "zero_truth_energy",
    "wall_seconds",
    "timing_profile",
    "worker",
    "error_type",
)
GAME_FIELDS = ("id", "family", "stratum", "n_players", "index", "order")
METADATA_FIELDS = (
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
    "point_label",
    "cluster_id",
    "zero_truth_energy",
    "synthetic",
    "stochastic_frozen",
    "case_id",
    "class",
)


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
            for field in ("nmse", "mse", "seconds", "queries", "requested_queries"):
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
            key: first["suite"][key]
            for key in (
                "name",
                "budgets",
                "seeds",
                "methods",
                "budgets_by_game",
                "relative_budgets",
            )
            if key in first["suite"]
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


def report(paths: list[Path], output: Path, *, public: bool = False) -> dict:
    """Write data and the three dependency-free website files."""
    data = merge_results(paths)
    assets = SITE_DIR
    public = public or output.resolve() == assets.resolve()
    if public and any(method.get("private", True) for method in data["methods"].values()):
        message = "Public reports cannot include private candidate methods."
        raise ValueError(message)
    data["presets"] = summarize(data)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "app.js", "style.css"):
        source, destination = assets / name, output / name
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    (output / "data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
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
