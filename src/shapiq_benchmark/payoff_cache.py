"""Authenticated payoff checkpoints and batch-amortized oracle timing records."""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


CHUNK_SIZE = 4096
CHUNK_PROTOCOL = "ascending-4096-fresh-seeded-recipe-v1"
COST_PROTOCOL = "batch-amortized-wall-seconds-v1"


def cost_batch(start: int, stop: int, seconds: float) -> dict:
    """Record measured batch work without local host names or paths."""
    from shapiq_benchmark.execution import THREAD_VARIABLES, hardware

    observed = hardware()
    return {
        "start": start,
        "stop": stop,
        "seconds": seconds,
        "cpu_model": observed["cpu_model"],
        "affinity": observed["affinity"],
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
    }


def chunk_identity(spec: dict, seed: int, start: int, source: dict) -> dict:
    """Bind a checkpoint to its recipe, construction seed, protocol and software."""
    software = {
        key: source.get(key)
        for key in ("python", "numpy", "scikit-learn", "shapiq", "installed_packages")
    }
    return {
        "spec": spec,
        "instance_seed": seed,
        "start": start,
        "stop": min(start + CHUNK_SIZE, 2 ** spec["n_players"]),
        "protocol": CHUNK_PROTOCOL,
        "source_sha256": source["source_sha256"],
        "software_sha256": hashlib.sha256(
            json.dumps(software, sort_keys=True).encode()
        ).hexdigest(),
    }


def read_chunk(path: Path, expected: dict) -> tuple:
    """Reject incomplete, corrupt, or incompatible checkpointed payoff work."""
    with np.load(path, allow_pickle=False) as saved:
        values, costs = saved["values"], saved["evaluation_seconds"]
        manifest = json.loads(str(saved["manifest"]))
    count = expected["stop"] - expected["start"]
    batch = manifest["batch"]
    if (
        manifest["identity"] != expected
        or values.shape != (count,)
        or costs.shape != (count,)
        or not np.isfinite(values).all()
        or not np.isfinite(costs).all()
        or np.any(costs < 0)
        or batch["start"] != expected["start"]
        or batch["stop"] != expected["stop"]
        or not math.isfinite(batch["seconds"])
        or batch["seconds"] < 0
        or not math.isclose(batch["seconds"], float(np.sum(costs)), rel_tol=1e-12, abs_tol=0)
        or hashlib.sha256(values.tobytes() + costs.tobytes()).hexdigest()
        != manifest["payload_sha256"]
    ):
        message = "Cached payoff chunk is incomplete or does not match its source/recipe/protocol."
        raise ValueError(message)
    return values, costs, manifest


def write_chunk(path: Path, values: np.ndarray, costs: np.ndarray, manifest: dict) -> None:
    """Publish a complete payoff chunk atomically with its payload checksum."""
    manifest = {
        **manifest,
        "payload_sha256": hashlib.sha256(values.tobytes() + costs.tobytes()).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            values=values,
            evaluation_seconds=costs,
            manifest=json.dumps(manifest, allow_nan=False),
        )
    temporary.replace(path)
