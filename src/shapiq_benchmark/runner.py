"""Run counted, reproducible estimator comparisons against a frozen snapshot.

Candidate Python files are trusted local code, not sandboxed plugins. They export a
factory(n, index, order, seed) returning an object with approximate(budget, game).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from shapiq import InteractionValues
from shapiq.approximator import (
    SHAPIQ,
    SVARM,
    SVARMIQ,
    KernelSHAP,
    KernelSHAPIQ,
    PermutationSamplingSV,
)

if TYPE_CHECKING:
    from collections.abc import Callable

from shapiq_benchmark.games import load_game

METHODS = {
    cls.__name__: cls
    for cls in (KernelSHAP, PermutationSamplingSV, SVARM, KernelSHAPIQ, SHAPIQ, SVARMIQ)
}


def digest(path: Path) -> str:
    """Hash the exact bytes of an artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(value: dict) -> str:
    """Hash a canonical JSON object."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def provenance() -> dict:
    """Describe the software actually executing this run."""
    versions = {
        name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "shapiq")
    }
    source_root = Path(__file__).resolve().parents[1]
    source_hash = hashlib.sha256()
    for package in ("shapiq", "shapiq_benchmark"):
        for path in sorted(
            p for p in (source_root / package).rglob("*") if p.suffix in (".py", ".so")
        ):
            source_hash.update(str(path.relative_to(source_root)).encode())
            source_hash.update(path.read_bytes())
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 -- only reads revision metadata
            cwd=source_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        **versions,
        "python": platform.python_version(),
        "git_commit": commit,
        "source_sha256": source_hash.hexdigest(),
        "benchmark_files": {
            p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("prepare.py"))
        },
    }


def validate_suite(suite: dict) -> None:
    """Reject ambiguous or empty run matrices before doing any work."""
    for name in ("budgets", "seeds", "methods"):
        values = suite[name]
        if not values or len(values) != len(set(values)):
            message = f"Suite {name} must be nonempty and unique."
            raise ValueError(message)
    for name in ("budgets", "seeds"):
        if any(
            type(value) is not int or value < (1 if name == "budgets" else 0)
            for value in suite[name]
        ):
            message = f"Suite {name} must contain valid integers."
            raise ValueError(message)
    if any(name not in METHODS for name in suite["methods"]):
        message = "Suite contains an unknown method."
        raise ValueError(message)


def load_snapshot(path: Path) -> tuple[dict, Path]:
    """Verify snapshot identity and every artifact before executing estimators."""
    path = path / "snapshot.json" if path.is_dir() else path
    snapshot = json.loads(path.read_text())
    unsigned = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
    if snapshot.get("schema_version") != 1 or identity(unsigned) != snapshot.get("snapshot_id"):
        message = "Snapshot schema or identity mismatch."
        raise ValueError(message)
    validate_suite(snapshot["suite"])
    root = path.parent.resolve()
    for relative, expected in snapshot["artifacts"].items():
        artifact = (root / relative).resolve()
        if not artifact.is_relative_to(root) or digest(artifact) != expected:
            message = f"Artifact hash mismatch: {relative}"
            raise ValueError(message)
    for game in snapshot["games"]:
        if game["artifact"] not in snapshot["artifacts"]:
            message = "Game artifact is not authenticated by the snapshot."
            raise ValueError(message)
    return snapshot, root


class BudgetExceededError(RuntimeError):
    """The estimator requested more coalition rows than its budget."""


class CountedGame:
    """Count every requested row, including duplicates and endpoint queries."""

    def __init__(self, game: Callable, n_players: int, budget: int) -> None:
        """Wrap a callable; denied requests leave a sticky budget violation."""
        self._game = game
        self.n_players = n_players
        self.budget = budget
        self.queries = 0
        self.requested = 0
        self.exceeded = False

    def __call__(self, coalitions: np.ndarray) -> np.ndarray:
        """Validate binary coalitions and charge before calling the oracle."""
        coalitions = np.asarray(coalitions)
        if coalitions.ndim == 1:
            coalitions = coalitions[None, :]
        if coalitions.ndim != 2 or coalitions.shape[1] != self.n_players:
            message = "Expected a coalition matrix with n_players columns."
            raise ValueError(message)
        if not np.all((coalitions == 0) | (coalitions == 1)):
            message = "Coalitions must be binary."
            raise ValueError(message)
        self.requested += len(coalitions)
        if self.exceeded or self.queries + len(coalitions) > self.budget:
            self.exceeded = True
            message = "Coalition-query budget exceeded."
            raise BudgetExceededError(message)
        self.queries += len(coalitions)
        values = np.asarray(self._game(coalitions.astype(bool)))
        if values.shape != (len(coalitions),) or not np.all(np.isfinite(values)):
            message = "Game returned invalid values."
            raise ValueError(message)
        return values


def table_game(values: np.ndarray, n_players: int) -> Callable:
    """Return the table oracle, with player zero as the least significant bit."""
    if values.shape != (2**n_players,) or not np.all(np.isfinite(values)):
        message = "Invalid coalition table."
        raise ValueError(message)
    weights = 2 ** np.arange(n_players, dtype=np.int64)
    return lambda coalitions: values[np.asarray(coalitions, dtype=np.int64) @ weights]


def score(estimate: InteractionValues, game: dict) -> dict:
    """Align coordinates and compute error over all nonempty target coefficients.

    Missing sparse coordinates mean zero. Nonzero estimates outside truth support
    are included. Zero-energy truth has undefined nMSE and is retained as null.
    """
    if not isinstance(estimate, InteractionValues):
        message = "Estimator must return InteractionValues."
        raise TypeError(message)
    n, order = game["n_players"], game["order"]
    if (estimate.n_players, estimate.index, estimate.max_order) != (n, game["index"], order):
        message = "Estimator returned the wrong player count, index, or order."
        raise ValueError(message)
    truth = {
        tuple(key): float(value)
        for key, value in zip(game["truth"]["coordinates"], game["truth"]["values"], strict=True)
    }
    prediction = estimate.dict_values
    for mapping in (truth, prediction):
        for coordinate, value in mapping.items():
            if (
                len(coordinate) > order
                or tuple(sorted(set(coordinate))) != coordinate
                or any(not isinstance(i, (int, np.integer)) or i < 0 or i >= n for i in coordinate)
            ):
                message = "Invalid interaction coordinate."
                raise ValueError(message)
            if not np.isfinite(value):
                message = "Nonfinite coefficient."
                raise ValueError(message)
    coordinates = (truth.keys() | prediction.keys()) - {()}
    squared_error = math.fsum(
        (float(prediction.get(key, 0.0)) - truth.get(key, 0.0)) ** 2 for key in coordinates
    )
    energy = math.fsum(value**2 for key, value in truth.items() if key)
    if not math.isfinite(squared_error) or not math.isfinite(energy):
        message = "Nonfinite error or truth energy."
        raise ValueError(message)
    if energy and not math.isfinite(squared_error / energy):
        message = "Nonfinite normalized error."
        raise ValueError(message)
    count = sum(math.comb(n, degree) for degree in range(1, order + 1))
    return {
        "mse": squared_error / count,
        "nmse": squared_error / energy if energy else None,
        "truth_energy": energy,
        "normalization": "nonempty_l2_energy",
        "zero_truth_energy": energy == 0,
    }


def candidate_factory(spec: str) -> tuple[str, Any, dict]:
    """Load an explicitly selected local candidate, recording its source hash."""
    filename, function = spec.rsplit(":", 1)
    path = Path(filename).resolve()
    module_name = "benchmark_candidate_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        message = "Cannot load candidate module."
        raise ValueError(message)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    name = f"candidate:{path.stem}:{function}"
    return (
        name,
        getattr(module, function),
        {
            "source_sha256": digest(path),
            "factory": function,
            "filename": path.name,
            "private": True,
        },
    )


def run(snapshot_path: Path, output: Path, candidate: str | None = None) -> dict:
    """Execute the snapshot's finite matrix and preserve failed runs as records."""
    snapshot, root = load_snapshot(snapshot_path)
    factories = {
        name: (
            lambda n, index, order, seed, cls=METHODS[name]: cls(
                n=n, index=index, max_order=order, random_state=seed
            )
        )
        for name in snapshot["suite"]["methods"]
    }
    software = provenance()
    methods = {
        name: {"source_sha256": software["source_sha256"], "private": False} for name in factories
    }
    if candidate:
        name, factory, metadata = candidate_factory(candidate)
        factories = {name: factory}
        methods = {name: metadata}
    result = {
        "schema_version": 1,
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_provenance": snapshot["provenance"],
        "run_provenance": software,
        "methods": methods,
        "suite": snapshot["suite"],
        "games": snapshot["games"],
        "records": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    for game in snapshot["games"]:
        oracle = load_game(game, root)
        for name, factory in factories.items():
            for budget in snapshot["suite"]["budgets"]:
                for seed in snapshot["suite"]["seeds"]:
                    counted = CountedGame(oracle, game["n_players"], budget)
                    record = {
                        "game_id": game["id"],
                        "method": name,
                        "budget": budget,
                        "seed": seed,
                        "status": "failed",
                        "nmse": None,
                        "mse": None,
                        "error": None,
                        "timing_scope": f"estimator_with_{game.get('oracle', 'table')}_oracle",
                        "official_timing": False,
                    }
                    if not candidate and game["index"] not in METHODS[name].valid_indices:
                        record.update(
                            status="unsupported", queries=0, requested_queries=0, seconds=None
                        )
                        result["records"].append(record)
                        write_results(result, output)
                        continue
                    start = time.perf_counter()
                    try:
                        estimator = factory(
                            n=game["n_players"], index=game["index"], order=game["order"], seed=seed
                        )
                        estimate = estimator.approximate(budget=budget, game=counted)
                        if counted.exceeded:
                            message = "Estimator caught a budget exception."
                            raise BudgetExceededError(message)  # noqa: TRY301
                        record["seconds"] = time.perf_counter() - start
                        record.update(score(estimate, game))
                        record["status"] = "ok"
                        record["estimate"] = {
                            "coordinates": [list(key) for key in estimate.dict_values],
                            "values": [float(value) for value in estimate.dict_values.values()],
                        }
                    except Exception as error:  # noqa: BLE001 -- preserve failed candidate runs
                        record["error"] = f"{type(error).__name__}: {error}"
                    record.update(
                        queries=counted.queries,
                        requested_queries=counted.requested,
                        seconds=record.get("seconds", time.perf_counter() - start),
                    )
                    result["records"].append(record)
                    write_results(result, output)
    return result


def write_results(result: dict, output: Path) -> None:
    """Atomically checkpoint JSON and write a convenient flat CSV companion."""
    temporary = output / "results.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "results.json")
    fields = sorted({key for row in result["records"] for key in row})
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["records"])


def main() -> None:
    """Run the command-line benchmark entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate", help="Trusted local Python file:factory (runs only this candidate)"
    )
    args = parser.parse_args()
    run(args.snapshot, args.output, args.candidate)


if __name__ == "__main__":
    main()
