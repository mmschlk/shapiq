"""Bound enumeration costs before launching full payoff tables.

A pilot tests finite endpoint/singleton payoffs and times fixed uniform coalitions.
It never inspects ground-truth attributions or estimator performance. Predictions
are planning estimates, not guarantees; exact-reference preparation remains a
separate gate. Each pilot owns a fresh process and cannot advance the real game.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import queue
import random
import resource
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from shapiq_benchmark.datasets import load_raw_dataset
from shapiq_benchmark.execution import THREAD_VARIABLES, hardware
from shapiq_benchmark.families import make_family
from shapiq_benchmark.games import prepare_structured
from shapiq_benchmark.materialize import CHUNK_SIZE, _synchronize
from shapiq_benchmark.media import EXTRA_CATALOG, make_extra, preparation_backend
from shapiq_benchmark.quality import QUALITY_PROTOCOL, QualityExclusion, imputation_stability
from shapiq_benchmark.runner import provenance

PILOT_PROTOCOL: dict = {
    "version": 2,
    "uniform_coalitions": 32,
    "coalition_seed": 0,
    "safety_factor": 2,
    "chunk_size": CHUNK_SIZE,
    "scope": "construction and payoff enumeration; exact-target solver cost is additional",
    "uncertainty": "Small-batch projection, not a runtime guarantee; every game seed is measured.",
}


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _pilot(spec: dict, seed: int, model_cache: str) -> dict:
    """Validate without a score-based selection rule, then measure the uniform workload."""
    random.seed(seed)
    np.random.seed(seed % 2**32)  # noqa: NPY002 -- optional backend global RNGs
    factory = make_extra if spec["family"] in EXTRA_CATALOG else make_family
    options = {
        key: spec[key]
        for key in (
            "dataset",
            "n_players",
            "model_profile",
            "device",
            "quality_protocol",
            "input_features",
            "feature_rule",
        )
        if key in spec
    }
    if factory is make_extra:
        options.pop("quality_protocol", None)
    if "model_profile" in spec:
        options["model_cache"] = model_cache
    _synchronize(spec)
    started = time.perf_counter()
    game, metadata = factory(spec["family"], instance_seed=seed, **options)
    _synchronize(spec)
    setup_seconds = time.perf_counter() - started
    stability = None
    if spec.get("quality_protocol") == QUALITY_PROTOCOL and spec["family"] in (
        "local_gaussian",
        "local_copula",
        "local_conditional",
    ):
        stability = imputation_stability(game, seed=seed)
        if stability["status"] != "stable":
            reason = "unstable_imputation"
            raise QualityExclusion(reason, stability)
    n = game.n_players
    if n != spec["n_players"] or not 1 <= n <= 20:
        message = "Constructed player count disagrees with the bounded recipe."
        raise ValueError(message)
    checks = np.vstack((np.zeros(n), np.ones(n), np.eye(n))).astype(bool)
    values = np.asarray(game(checks), dtype=float)
    if values.shape != (len(checks),) or not np.isfinite(values).all():
        message = "Endpoint or singleton payoff validation failed."
        raise ValueError(message)
    coalitions = (
        np.random.default_rng(PILOT_PROTOCOL["coalition_seed"])
        .integers(0, 2, size=(PILOT_PROTOCOL["uniform_coalitions"], n))
        .astype(bool)
    )
    _synchronize(spec)
    started = time.perf_counter()
    values = np.asarray(game(coalitions), dtype=float)
    _synchronize(spec)
    oracle_seconds = time.perf_counter() - started
    if values.shape != (len(coalitions),) or not np.isfinite(values).all():
        message = "Uniform-coalition payoff validation failed."
        raise ValueError(message)
    # Tables above twelve players reconstruct the game for each canonical chunk.
    constructions = 1 if n <= 12 else 1 + math.ceil(2**n / CHUNK_SIZE)
    projected = PILOT_PROTOCOL["safety_factor"] * (
        setup_seconds * constructions + oracle_seconds / len(coalitions) * 2**n
    )
    return {
        "status": "measured",
        "setup_seconds": setup_seconds,
        "oracle_seconds": oracle_seconds,
        "uniform_coalitions": len(coalitions),
        "projected_seconds": projected,
        "projected_constructions": constructions,
        "validation_coalitions": len(checks),
        "hardware": hardware(),
        "preparation_hardware": metadata.get("preparation_hardware", {"device": "cpu"}),
        "model_key": metadata.get("model_key"),
        "data_sha256": metadata.get("data_sha256"),
        "model_validation_gate": metadata.get("model_validation_gate"),
        "imputation_stability": stability,
    }


def _structured_pilot(spec: dict, seed: int, model_cache: str) -> dict:
    """Measure the actual requested solver, including its correctness qualification."""
    random.seed(seed)
    np.random.seed(seed % 2**32)  # noqa: NPY002 -- optional backend global RNGs
    cache = Path(model_cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="structured-pilot-", dir=cache.parent) as directory:
        game = prepare_structured(
            [{**spec, "id": f"{spec['id']}-i{seed}", "instance_seed": seed}], Path(directory)
        )[0]
        # Truth coordinates can dwarf the numeric oracle at native dimensions.
        # Include their real serialization, not only a small-model solver proxy.
        _write(Path(directory) / "game.json", game)
        seconds = time.perf_counter() - started
        if game["n_players"] != spec.get("n_players", game["n_players"]):
            message = "Structured pilot did not construct the requested player count."
            raise ValueError(message)
        artifact_bytes = (Path(directory) / game["artifact"]).stat().st_size
        metadata_bytes = (Path(directory) / "game.json").stat().st_size
    return {
        "status": "measured",
        "projected_seconds": seconds,
        "actual_preparation_seconds": seconds,
        "n_players": game["n_players"],
        "index": game["index"],
        "order": game["order"],
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "artifact_bytes": artifact_bytes,
        "metadata_bytes": metadata_bytes,
        "hardware": hardware(),
        "scope": "Actual requested-size construction, exact solver, validation and serialization",
    }


def _run_pilot(
    identity: dict, directory: Path, model_cache: str, timeout: float, cpu: int | None
) -> dict:
    """Kill the whole pilot process group on timeout; exception details stay private."""
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    result_path = directory / f"{key}.json"
    # Pretrained assets can change outside the source tree. Re-probe these rather
    # than authenticate an old pilot only by a model name or package version.
    reusable = (
        identity["spec"].get("family") not in {*EXTRA_CATALOG, "_dataset_identity"}
        and identity["spec"].get("model_profile") != "tabpfn_prediction"
    )
    if reusable and result_path.exists():
        cached = json.loads(result_path.read_text())
        if cached.get("identity") == identity:
            return cached["result"]
    private = directory / "private"
    private.mkdir(exist_ok=True)
    request_path = private / f"{key}-request.json"
    response_path = private / f"{key}-response.json"
    response_path.unlink(missing_ok=True)
    _write(
        request_path,
        {
            "spec": identity["spec"],
            "seed": identity["seed"],
            "model_cache": model_cache,
            "cpu": cpu,
            "kind": identity.get("kind", "families"),
            "memory_gb": identity.get("structured_memory_gb"),
        },
    )
    environment = {**os.environ, **dict.fromkeys(THREAD_VARIABLES, "1")}
    with (private / f"{key}.log").open("w") as log:
        process = subprocess.Popen(  # noqa: S603 -- fixed module; recipes passed as JSON, never shell code
            [
                sys.executable,
                "-m",
                "shapiq_benchmark.qualification",
                "--pilot",
                str(request_path),
                str(response_path),
            ],
            env=environment,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            result = {"status": "failed", "reason": "pilot_timeout", "limit_seconds": timeout}
        else:
            result = (
                json.loads(response_path.read_text())
                if process.returncode == 0 and response_path.exists()
                else {
                    "status": "failed",
                    "reason": "pilot_process_failed",
                    "exit_code": process.returncode,
                }
            )
    _write(result_path, {"identity": identity, "result": result})
    return result


def _dataset_identity(name: str) -> dict:
    x, y, names = load_raw_dataset(name)
    digest = hashlib.sha256()
    for array in (np.asarray(x), np.asarray(y)):
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    digest.update(json.dumps(list(map(str, names))).encode())
    return {"dataset": name, "data_sha256": digest.hexdigest()}


def _bounded_dataset_identity(name: str, path: Path, timeout: float, cpu: int | None) -> dict:
    result = _run_pilot(
        {"spec": {"family": "_dataset_identity", "dataset": name}, "seed": 0},
        path,
        "",
        timeout,
        cpu,
    )
    return (
        result["dataset_identity"]
        if result["status"] == "measured"
        else {
            "error_type": result.get("error_type", "DatasetPreflightError"),
            "reason": result["reason"],
        }
    )


def qualify_suite(
    suite: dict,
    path: Path,
    model_cache: str | Path,
    max_seconds: float = 28800,
    *,
    pilot_timeout: float = 480,
    workers: int = 1,
    structured_timeout: float = 480,
    structured_memory_gb: float = 12,
) -> dict:
    """Keep all four instances together and retain every exclusion with its measured reason.

    ``path`` is a checkpoint directory. Structured references execute their actual
    requested solver in time- and memory-bounded processes. Pretrained media
    pilots deliberately rerun on resume because their external assets may change.
    """
    if (
        max_seconds <= 0
        or not math.isfinite(max_seconds)
        or pilot_timeout <= 0
        or not math.isfinite(pilot_timeout)
        or type(workers) is not int
        or workers < 1
        or not math.isfinite(structured_timeout)
        or structured_timeout <= 0
        or not math.isfinite(structured_memory_gb)
        or structured_memory_gb <= 0
    ):
        message = "Cost limits and worker count must be positive and finite."
        raise ValueError(message)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    private = path / "private"
    private.mkdir(exist_ok=True)
    source = provenance()
    observed = hardware()
    cpu_ids = observed["affinity"] or [None]
    has_cuda = any(spec.get("device") == "cuda" for spec in suite.get("families", []))
    if has_cuda and workers != 1:
        message = "CUDA pilots require one worker per allocated GPU."
        raise ValueError(message)
    worker_count = min(workers, len(cpu_ids))
    cpus = queue.Queue()
    for cpu in cpu_ids[:worker_count]:
        cpus.put(cpu)
    datasets, identities, input_errors = {}, [], {}
    recipes = [(kind, spec) for kind in ("families", "games") for spec in suite.get(kind, [])]
    for kind, spec in recipes:
        dataset = spec.get("dataset")
        if dataset and dataset not in datasets:
            try:
                datasets[dataset] = _bounded_dataset_identity(
                    dataset, path, pilot_timeout, cpu_ids[0]
                )
            except Exception as error:  # noqa: BLE001 -- retain failures as explicit inventory
                datasets[dataset] = {"error_type": type(error).__name__}
                (
                    private / (hashlib.sha256(dataset.encode()).hexdigest() + "-dataset.log")
                ).write_text(traceback.format_exc())
        if dataset and "error_type" in datasets[dataset]:
            input_errors[spec["id"]] = datasets[dataset]
        try:
            backend = (
                preparation_backend("cuda")
                if spec.get("device") == "cuda"
                else {"cpu_model": observed["cpu_model"]}
            )
        except Exception as error:  # noqa: BLE001 -- preserve unavailable backend as inventory
            backend = {"error_type": type(error).__name__}
            input_errors[spec["id"]] = {"reason": "backend_unavailable", **backend}
            (
                private / (hashlib.sha256(spec["id"].encode()).hexdigest() + "-backend.log")
            ).write_text(traceback.format_exc())
        identities.extend(
            {
                "spec": spec,
                "kind": kind,
                "seed": seed,
                "source": source,
                "inputs": datasets.get(dataset),
                "backend": backend,
                "protocol": PILOT_PROTOCOL,
                "pilot_timeout": pilot_timeout,
                "structured_timeout": structured_timeout,
                "structured_memory_gb": structured_memory_gb if kind == "games" else None,
            }
            for seed in suite.get("game_seeds", [0])
        )

    def evaluate(identity: dict) -> dict:
        if identity["spec"]["id"] in input_errors:
            return {
                "status": "failed",
                "reason": "dataset_unavailable",
                **input_errors[identity["spec"]["id"]],
            }
        cpu = cpus.get()
        try:
            timeout = (
                min(structured_timeout, max_seconds)
                if identity["kind"] == "games"
                else pilot_timeout
            )
            return _run_pilot(identity, path, str(model_cache), timeout, cpu)
        finally:
            cpus.put(cpu)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        pilots = list(executor.map(evaluate, identities))
    if provenance() != source:
        message = "Source changed during preparation preflight; discard its qualification."
        raise ValueError(message)
    results = copy.deepcopy(suite)
    results["families"] = []
    if "games" in results:
        results["games"] = []
    exclusions = []
    summaries = []
    for kind, spec in recipes:
        instances = [
            {"seed": identity["seed"], **result}
            for identity, result in zip(identities, pilots, strict=True)
            if identity["spec"]["id"] == spec["id"] and identity["kind"] == kind
        ]
        failed = any(item["status"] != "measured" for item in instances)
        costly = any(item.get("projected_seconds", 0) > max_seconds for item in instances)
        summary = {
            "id": spec["id"],
            "kind": kind,
            "instances": instances,
            "status": "excluded" if failed or costly else "qualified",
        }
        summaries.append(summary)
        if failed or costly:
            exclusions.append(
                {
                    "spec": spec,
                    "kind": kind,
                    "reason": "preflight_failed" if failed else "projected_cost_limit",
                    "maximum_seconds_per_instance": max_seconds,
                    "instances": instances,
                }
            )
        else:
            results[kind].append(spec)
    results["preparation_exclusions"] = [*suite.get("preparation_exclusions", []), *exclusions]
    results["preparation_preflight"] = {
        **PILOT_PROTOCOL,
        "source": source,
        "requested_suite_sha256": hashlib.sha256(
            json.dumps(suite, sort_keys=True).encode()
        ).hexdigest(),
        "maximum_seconds_per_instance": max_seconds,
        "pilot_timeout_seconds": pilot_timeout,
        "families": summaries,
        "structured_references": "Actual requested-size solver, validation and serialization in bounded processes for every seed.",
        "structured_timeout_seconds": min(structured_timeout, max_seconds),
        "structured_memory_gb": structured_memory_gb,
        "selection_rule": "Any failed or over-limit seed excludes the entire recipe; no attribution or estimator score filtering.",
    }
    _write(path / "qualified-suite.json", results)
    return results


def main() -> None:
    """Private bounded worker entry point; the controller owns all public inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    request = json.loads(args.pilot.read_text())
    if request["cpu"] is not None:
        os.sched_setaffinity(0, {request["cpu"]})
    try:
        if request.get("memory_gb") is not None:
            limit = int(request["memory_gb"] * 1024**3)
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        if request["spec"].get("family") == "_dataset_identity":
            result = {
                "status": "measured",
                "dataset_identity": _dataset_identity(request["spec"]["dataset"]),
            }
        elif request.get("kind") == "games":
            result = _structured_pilot(request["spec"], request["seed"], request["model_cache"])
        else:
            result = _pilot(request["spec"], request["seed"], request["model_cache"])
    except Exception as error:  # noqa: BLE001 -- worker preserves a sanitized failure record
        traceback.print_exc()
        result = {
            "status": "failed",
            "reason": getattr(error, "reason", "construction_or_payoff_validation_failed"),
            "error_type": type(error).__name__,
        }
        if hasattr(error, "details"):
            result["details"] = error.details
    _write(args.output, result)


if __name__ == "__main__":
    main()
