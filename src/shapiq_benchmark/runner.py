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
import inspect
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import shapiq.approximator as approximators
from shapiq import InteractionValues

if TYPE_CHECKING:
    from collections.abc import Callable

from shapiq_benchmark.games import load_game
from shapiq_benchmark.order_metrics import order_scores
from shapiq_benchmark.results_io import Checkpoint, read_results

METHOD_NAMES = (
    "PermutationSamplingSII",
    "PermutationSamplingSTII",
    "PermutationSamplingSV",
    "StratifiedSamplingSV",
    "OwenSamplingSV",
    "KernelSHAP",
    "LeverageSHAP",
    "RegressionFSII",
    "RegressionFBII",
    "KernelSHAPIQ",
    "InconsistentKernelSHAPIQ",
    "ProxySPEX",
    "ProxySHAP",
    "OddSHAP",
    "RegressionMSR",
    "ShaplEIG",
    "SHAPIQ",
    "SVARM",
    "SVARMIQ",
    "kADDSHAP",
    "SPEX",
    "UnbiasedKernelSHAP",
)
METHODS = {name: getattr(approximators, name) for name in METHOD_NAMES}


def method_catalog() -> dict:
    """Describe supported targets without constructing estimators or fitting models."""
    return {
        name: {
            "indices": ["SV"]
            if name == "ShaplEIG"
            else (["SV", "BV"] if name == "RegressionMSR" else list(cls.valid_indices)),
            "import_available": not hasattr(cls, "_import_error"),
            "availability_note": "Import check only; optional backend dependencies may fail at construction.",
        }
        for name, cls in METHODS.items()
    }


def validate_method_parameters(
    name: str, parameters: dict, *, check_constructor: bool = True
) -> None:
    """Validate settings; only execution needs the current constructor's signature."""
    reserved = {"n", "index", "max_order", "random_state"}
    if (
        not isinstance(parameters, dict)
        or any(not isinstance(key, str) for key in parameters)
        or parameters.keys() & reserved
    ):
        message = f"Invalid explicit constructor parameters for {name}."
        raise ValueError(message)
    json.dumps(parameters, allow_nan=False)
    if not check_constructor:
        return
    accepted = {
        key
        for key, parameter in inspect.signature(METHODS[name]).parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    } - reserved
    if not parameters.keys() <= accepted:
        message = f"Invalid explicit constructor parameters for {name}."
        raise ValueError(message)


def builtin_factory(
    name: str, game: dict, seed: int, parameters: dict | None = None
) -> approximators.Approximator:
    """Pass only explicitly accepted common constructor parameters."""
    cls = METHODS[name]
    if hasattr(cls, "_import_error"):
        raise ImportError(str(cls._import_error))
    accepted = inspect.signature(cls).parameters
    overrides = parameters if parameters is not None else {}
    validate_method_parameters(name, overrides)
    arguments = {
        "n": game["n_players"],
        "index": game["index"],
        "max_order": game["order"],
        "random_state": seed,
    }
    return cls(**{key: value for key, value in arguments.items() if key in accepted}, **overrides)


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
    for package in ("shapiq", "shapiq_benchmark", "shapiq_games"):
        for path in sorted(
            p for p in (source_root / package).rglob("*") if p.suffix in (".py", ".so")
        ):
            source_hash.update(str(path.relative_to(source_root)).encode())
            source_hash.update(path.read_bytes())
    try:
        commit = subprocess.check_output(  # noqa: S603 -- fixed read-only git commands
            ["git", "rev-parse", "HEAD"],  # noqa: S607 -- only reads revision metadata
            cwd=source_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(  # noqa: S603 -- fixed read-only git commands
                [  # noqa: S607 -- fixed read-only git executable
                    "git",
                    "status",
                    "--porcelain",
                    "--",
                    "src/shapiq",
                    "src/shapiq_benchmark",
                    "src/shapiq_games",
                ],
                cwd=source_root.parent,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        **versions,
        "rng_protocol": "cell-seed-python-numpy-v1",
        "python": platform.python_version(),
        "installed_packages": {
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
        },
        "git_commit": commit,
        "source_dirty": dirty,
        "source_sha256": source_hash.hexdigest(),
        "benchmark_files": {
            p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("prepare.py"))
        },
    }


def cell_timeout_policy(suite: dict) -> dict | None:
    """Validate the optional frozen resource policy without changing legacy suites."""
    if "cell_timeout_policy" not in suite:
        return None
    policy = suite["cell_timeout_policy"]
    keys = {"ordinary_seconds", "extended_seconds", "min_players", "min_relative_budget"}
    if (
        not isinstance(policy, dict)
        or set(policy) != keys
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in policy.values()
        )
        or type(policy["min_players"]) is not int
        or policy["extended_seconds"] < policy["ordinary_seconds"]
    ):
        message = (
            "Invalid cell_timeout_policy: require positive finite limits and integer min_players."
        )
        raise ValueError(message)
    return policy


def cell_timeout(game: dict, budget: int, timeout: float, policy: dict | None) -> float:
    """Use the requested query cap, not observed queries, to select a frozen time limit."""
    if policy is None:
        return timeout
    extended = (
        game["n_players"] >= policy["min_players"]
        and budget >= policy["min_relative_budget"] * game["n_players"]
    )
    return policy["extended_seconds" if extended else "ordinary_seconds"]


def validate_suite(suite: dict, *, check_constructors: bool = True) -> None:
    """Reject ambiguous or empty run matrices before doing any work."""
    cell_timeout_policy(suite)
    minimum = suite.get("min_players", 1)
    if type(minimum) is not int or minimum < 1:
        message = "min_players must be a positive integer."
        raise ValueError(message)
    signal = suite.get("min_signal_ratio")
    if "min_signal_ratio" in suite and (
        not isinstance(signal, int | float)
        or isinstance(signal, bool)
        or not math.isfinite(signal)
        or signal <= 0
    ):
        message = "min_signal_ratio must be finite and positive."
        raise ValueError(message)
    for name in (
        "seeds",
        "methods",
        *(["budgets"] if "budgets" in suite else ["relative_budgets"]),
    ):
        values = suite[name]
        if not values or len(values) != len(set(values)):
            message = f"Suite {name} must be nonempty and unique."
            raise ValueError(message)
    for name in ("budgets", "seeds"):
        if any(
            type(value) is not int or value < (1 if name == "budgets" else 0)
            for value in suite.get(name, [])
        ):
            message = f"Suite {name} must contain valid integers."
            raise ValueError(message)
    for ratio in suite.get("relative_budgets", []):
        if type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio <= 0:
            message = "Relative budgets must be finite positive numbers."
            raise ValueError(message)
    for grid in suite.get("budgets_by_game", {}).values():
        if (
            not grid
            or len(set(grid)) != len(grid)
            or any(type(b) is not int or b <= 0 or b not in suite["budgets"] for b in grid)
        ):
            message = "Per-game budget grids must contain unique declared positive integers."
            raise ValueError(message)
    if any(name not in METHODS for name in suite["methods"]):
        message = "Suite contains an unknown method."
        raise ValueError(message)
    parameters = suite.get("method_parameters", {})
    if not isinstance(parameters, dict) or not parameters.keys() <= set(suite["methods"]):
        message = "Method parameters must name declared suite methods."
        raise ValueError(message)
    for name, overrides in parameters.items():
        validate_method_parameters(name, overrides, check_constructor=check_constructors)


def load_snapshot(path: Path, *, historical: bool = False) -> tuple[dict, Path]:
    """Authenticate artifacts; historical export need not match today's constructors.

    Execution remains constructor-strict by default. Historical mode preserves
    structural and reserved-parameter checks; callers authenticate the recorded
    source and method settings before publishing those measurements.
    """
    path = path / "snapshot.json" if path.is_dir() else path
    snapshot = json.loads(path.read_text())
    unsigned = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
    if snapshot.get("schema_version") != 1 or identity(unsigned) != snapshot.get("snapshot_id"):
        message = "Snapshot schema or identity mismatch."
        raise ValueError(message)
    validate_suite(snapshot["suite"], check_constructors=not historical)
    root = path.parent.resolve()
    for relative, expected in snapshot["artifacts"].items():
        artifact = (root / relative).resolve()
        if not artifact.is_relative_to(root) or digest(artifact) != expected:
            message = f"Artifact hash mismatch: {relative}"
            raise ValueError(message)
    for game in snapshot["games"]:
        if game["n_players"] < snapshot["suite"].get("min_players", 1):
            message = "Game is below the suite min_players constraint."
            raise ValueError(message)
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
        self.evaluation_seconds = getattr(game, "evaluation_seconds", None)
        self.estimated_oracle_seconds = 0.0
        self.cache_lookup_seconds = 0.0

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
        binary = coalitions.astype(bool)
        start = time.perf_counter() if self.evaluation_seconds is not None else None
        values = np.asarray(self._game(binary))
        if start is not None and self.evaluation_seconds is not None:
            self.cache_lookup_seconds += time.perf_counter() - start
            positions = binary.astype(np.int64) @ (2 ** np.arange(self.n_players, dtype=np.int64))
            self.estimated_oracle_seconds += float(self.evaluation_seconds[positions].sum())
        if values.shape != (len(coalitions),) or not np.all(np.isfinite(values)):
            message = "Game returned invalid values."
            raise ValueError(message)
        return values


def table_game(
    values: np.ndarray, n_players: int, evaluation_seconds: np.ndarray | None = None
) -> Callable:
    """Return a table oracle with optional measured, batch-amortized coalition costs."""
    if values.shape != (2**n_players,) or not np.all(np.isfinite(values)):
        message = "Invalid coalition table."
        raise ValueError(message)
    if evaluation_seconds is not None and (
        evaluation_seconds.shape != values.shape
        or not np.all(np.isfinite(evaluation_seconds))
        or np.any(evaluation_seconds < 0)
    ):
        message = "Invalid coalition evaluation costs."
        raise ValueError(message)
    weights = 2 ** np.arange(n_players, dtype=np.int64)

    def oracle(coalitions: np.ndarray) -> np.ndarray:
        return values[np.asarray(coalitions, dtype=np.int64) @ weights]

    setattr(oracle, "evaluation_seconds", evaluation_seconds)  # noqa: B010 -- callable metadata
    return oracle


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
                or any(not isinstance(i, int | np.integer) or i < 0 or i >= n for i in coordinate)
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
        "order_scores": order_scores(truth, prediction, game),
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


def run_one(
    game: dict,
    root: Path,
    method: str,
    budget: int,
    seed: int,
    candidate: str | None = None,
    *,
    parameters: dict | None = None,
) -> dict:
    """Time the estimator excluding imports, oracle reconstruction, and scoring.

    Saved batch-amortized oracle costs yield a separate uncached-time estimate:
    measured estimator time minus cache calls plus charged original-game costs.
    This estimate is not a measurement of execution against the original game.
    """
    counted = CountedGame(load_game(game, root), game["n_players"], budget)
    # Optional backends such as sparse-transform use global RNGs rather than the
    # estimator's Generator. Each isolated cell must seed both before construction.
    random.seed(seed)
    np.random.seed(seed % 2**32)  # noqa: NPY002 -- backend uses the legacy 32-bit RNG
    factory = candidate_factory(candidate)[1] if candidate else None
    record: dict = {"status": "failed", "nmse": None, "mse": None, "error": None}
    start = time.perf_counter()
    try:
        estimator = (
            factory(n=game["n_players"], index=game["index"], order=game["order"], seed=seed)
            if factory
            else builtin_factory(method, game, seed, parameters)
        )
        estimate = estimator.approximate(budget=budget, game=counted)
        record["seconds"] = time.perf_counter() - start
        if counted.exceeded:
            message = "Estimator caught a budget exception."
            raise BudgetExceededError(message)  # noqa: TRY301
        record.update(score(estimate, game))
        record["estimate"] = {
            "coordinates": [[int(i) for i in key] for key in estimate.dict_values],
            "values": [float(value) for value in estimate.dict_values.values()],
        }
        record["status"] = "ok"
    except Exception as error:  # noqa: BLE001 -- preserve failed candidate runs
        record.update(status="failed", nmse=None, mse=None)
        record["error"] = f"{type(error).__name__}: {error}"
    record.update(
        queries=counted.queries,
        requested_queries=counted.requested,
        seconds=record.get("seconds", time.perf_counter() - start),
    )
    if record["status"] == "ok" and counted.evaluation_seconds is not None:
        record.update(
            cache_lookup_seconds=counted.cache_lookup_seconds,
            estimated_oracle_seconds=counted.estimated_oracle_seconds,
            estimated_uncached_seconds=max(0.0, record["seconds"] - counted.cache_lookup_seconds)
            + counted.estimated_oracle_seconds,
        )
    return record


def run(
    snapshot_path: Path,
    output: Path,
    candidate: str | None = None,
    *,
    resume: bool = False,
    timeout: float = 60,
    memory_gb: float | None = None,
    max_runs: int | None = None,
    max_seconds: float = 600,
    timing_profile: str = "diagnostic",
    method_names: list[str] | None = None,
    game_ids: list[str] | None = None,
) -> dict:
    """Checkpoint a finite matrix of isolated cells and safely resume identical campaigns."""
    from shapiq_benchmark.execution import (
        hardware,
        isolated,
        verify_profile,
    )

    if (
        not math.isfinite(timeout)
        or not math.isfinite(max_seconds)
        or timeout <= 0
        or max_seconds <= timeout
        or (memory_gb is not None and (not math.isfinite(memory_gb) or memory_gb <= 0))
        or (max_runs is not None and (type(max_runs) is not int or max_runs < 1))
    ):
        message = "Campaign limits must be finite and positive; max_seconds must exceed a full cell timeout."
        raise ValueError(message)
    verify_profile(timing_profile)
    import fcntl

    output.mkdir(parents=True, exist_ok=True)
    with (output / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot_path = snapshot_path.resolve()
        # Private factories do not execute the snapshot's historical baselines.
        snapshot, artifact_root = load_snapshot(snapshot_path, historical=bool(candidate))
        timeout_policy = cell_timeout_policy(snapshot["suite"])
        if timeout_policy is not None and max_seconds <= timeout_policy["extended_seconds"]:
            message = "max_seconds must exceed the longest cell timeout in the suite policy."
            raise ValueError(message)
        duplicates = {}
        if not candidate and snapshot["suite"].get("duplicate_registry"):
            from shapiq_benchmark.duplicates import claim_games

            duplicates = claim_games(
                snapshot, artifact_root, Path(snapshot["suite"]["duplicate_registry"])
            )
        if method_names is not None and (
            candidate
            or not method_names
            or len(set(method_names)) != len(method_names)
            or not set(method_names) <= set(snapshot["suite"]["methods"])
        ):
            message = "--methods must select unique suite methods and cannot accompany a candidate."
            raise ValueError(message)
        selected_games = snapshot["games"]
        if game_ids is not None:
            if (
                not game_ids
                or len(set(game_ids)) != len(game_ids)
                or not set(game_ids) <= {g["id"] for g in selected_games}
            ):
                message = "--games must select unique snapshot game IDs."
                raise ValueError(message)
            selected_games = [g for g in selected_games if g["id"] in game_ids]
        software = provenance()
        methods = {
            name: {
                "source_sha256": software["source_sha256"],
                "software_sha256": identity(software),
                "private": False,
                **(
                    {"parameters": snapshot["suite"]["method_parameters"][name]}
                    if snapshot["suite"].get("method_parameters", {}).get(name)
                    else {}
                ),
            }
            for name in (method_names or snapshot["suite"]["methods"])
        }
        if candidate:
            filename, function = candidate.rsplit(":", 1)
            path = Path(filename).resolve()
            candidate = f"{path}:{function}"
            name = f"candidate:{path.stem}:{function}"
            methods = {
                name: {
                    "source_sha256": digest(path),
                    "software_sha256": identity(software),
                    "factory": function,
                    "filename": path.name,
                    "private": True,
                }
            }
        execution = {
            "timeout": timeout,
            "memory_gb": memory_gb,
            "threads": 1,
            "timing_profile": timing_profile,
            "hardware": hardware(),
            "game_ids": [game["id"] for game in selected_games],
        }
        if timeout_policy is not None:
            execution["cell_timeout_policy"] = timeout_policy
        result: dict = {
            "schema_version": 1,
            "snapshot_id": snapshot["snapshot_id"],
            "snapshot_provenance": snapshot["provenance"],
            "run_provenance": {**software, "execution": execution},
            "methods": methods,
            "suite": snapshot["suite"],
            "games": snapshot["games"],
            "coverage": snapshot.get("coverage", []),
            "records": [],
        }
        result["resume_key"] = identity(
            {key: value for key, value in result.items() if key != "records"}
        )
        existing = output / "results.json"
        if existing.exists():
            previous = read_results(existing, recover=True)
            if not resume or (
                previous.get("resume_key") != result["resume_key"]
                or identity(
                    {
                        key: value
                        for key, value in previous.items()
                        if key not in ("records", "campaign", "resume_key")
                    }
                )
                != result["resume_key"]
            ):
                message = "Existing campaign requires --resume with identical snapshot, code, methods, hardware, and resource limits."
                raise ValueError(message)
            result = previous
        elif resume:
            message = "No existing campaign to resume."
            raise ValueError(message)
        keys = ("game_id", "method", "budget", "seed")
        completed = {tuple(row[key] for key in keys) for row in result["records"]}
        if len(completed) != len(result["records"]):
            message = "Checkpoint contains duplicate cells."
            raise ValueError(message)
        planned = [
            (game, name, budget, seed)
            for game in selected_games
            for name in methods
            for budget in snapshot["suite"]
            .get("budgets_by_game", {})
            .get(game["id"], snapshot["suite"]["budgets"])
            for seed in snapshot["suite"]["seeds"]
        ]
        planned_keys = {(game["id"], name, budget, seed) for game, name, budget, seed in planned}
        if not completed <= planned_keys or any(
            row["status"] not in ("ok", "failed", "unsupported", "duplicate")
            for row in result["records"]
        ):
            message = "Checkpoint contains cells outside the planned matrix."
            raise ValueError(message)
        start, added = time.monotonic(), 0
        result["campaign"] = {
            "planned": len(planned),
            "completed": len(result["records"]),
            "complete": len(result["records"]) == len(planned),
        }
        checkpoint = Checkpoint(output, snapshot_path, result)
        for game, name, budget, seed in planned:
            if (game["id"], name, budget, seed) in completed:
                continue
            remaining = max_seconds - (time.monotonic() - start)
            selected_timeout = cell_timeout(game, budget, timeout, timeout_policy)
            if remaining < selected_timeout or (max_runs is not None and added >= max_runs):
                break
            record = {
                "game_id": game["id"],
                "method": name,
                "budget": budget,
                "seed": seed,
                "nmse": None,
                "mse": None,
                "error": None,
                "timing_scope": f"estimator_with_{game.get('oracle', 'table')}_oracle",
                "official_timing": False,
                "timing_profile": timing_profile,
            }
            if game["id"] in duplicates:
                record.update(
                    status="duplicate",
                    duplicate_of=duplicates[game["id"]],
                    queries=0,
                    requested_queries=0,
                    seconds=None,
                    wall_seconds=0,
                )
            elif not candidate and game["index"] not in method_catalog()[name]["indices"]:
                record.update(
                    status="unsupported",
                    queries=0,
                    requested_queries=0,
                    seconds=None,
                    wall_seconds=0,
                )
            else:
                request = {
                    "snapshot": str(snapshot_path),
                    "authenticated_game": game,
                    "artifact_root": str(artifact_root),
                    "artifact_sha256": snapshot["artifacts"][game["artifact"]],
                    "game_id": game["id"],
                    "method": name,
                    "budget": budget,
                    "seed": seed,
                    "candidate": candidate,
                    "timing_profile": timing_profile,
                    "expected_snapshot_id": snapshot["snapshot_id"],
                    "expected_source_hash": software["source_sha256"],
                    "candidate_sha256": methods[name]["source_sha256"] if candidate else None,
                    "method_parameters": methods[name].get("parameters"),
                }
                record.update(isolated(request, selected_timeout, memory_gb))
            result["records"].append(record)
            added += 1
            result["campaign"] = {
                "planned": len(planned),
                "completed": len(result["records"]),
                "complete": len(result["records"]) == len(planned),
            }
            checkpoint.append(record)
        checkpoint.finish(result)
        _write_csv(result, output)
        return result


def write_results(result: dict, output: Path) -> None:
    """Atomically checkpoint JSON and write a convenient flat CSV companion."""
    temporary = output / "results.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "results.json")
    _write_csv(result, output)


def _write_csv(result: dict, output: Path) -> None:
    """Write the optional human-readable companion once per bounded run."""
    fields = sorted({key for row in result["records"] for key in row})
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["records"])


def main() -> None:
    """Run the command-line benchmark entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--candidate", help="Trusted local Python file:factory (runs only this candidate)"
    )
    parser.add_argument("--methods", nargs="+", help="Run only these suite methods")
    parser.add_argument("--games", nargs="+", help="Run only these snapshot game IDs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument(
        "--timeout", type=float, default=60, help="Whole-worker wall seconds per cell"
    )
    parser.add_argument("--memory-gb", type=float, help="POSIX worker address-space cap in GiB")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--timing-profile", default="diagnostic")
    args = parser.parse_args()
    if args.list_methods:
        sys.stdout.write(json.dumps(method_catalog(), indent=2) + "\n")
        return
    if args.snapshot is None or args.output is None:
        parser.error("--snapshot and --output are required unless --list-methods is used")
    run(
        args.snapshot,
        args.output,
        args.candidate,
        resume=args.resume,
        timeout=args.timeout,
        memory_gb=args.memory_gb,
        max_runs=args.max_runs,
        max_seconds=args.max_seconds,
        timing_profile=args.timing_profile,
        method_names=args.methods,
        game_ids=args.games,
    )


if __name__ == "__main__":
    main()
