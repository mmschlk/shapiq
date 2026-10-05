"""Immutable normalization acceleration; this is not a publication approval gate."""

from __future__ import annotations

import json
import os
import platform
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from shapiq_benchmark.runner import digest, identity

if TYPE_CHECKING:
    from collections.abc import Callable


def normalization_policy() -> dict:
    """Conservatively bind derived values to exporter code and numerical runtime."""
    return {
        "version": 1,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "files": {str(path): digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
    }


def normalized_batch(
    cache_dir: Path, inputs: dict[str, str], policy: dict, build: Callable[[], dict]
) -> dict:
    """Reuse an exact normalized batch after its caller has validated raw shards.

    Inputs include original shard companions and payoff artifacts. Entries retain
    every public row and original run ID; alias resolution belongs to the caller.
    Corrupt or incomplete entries fail closed instead of silently rebuilding.
    """
    binding = {"inputs": inputs, "policy": policy}
    key = identity(binding)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{key}.json"

    def authenticate() -> None:
        for name, expected in {**inputs, **policy["files"]}.items():
            if digest(Path(name)) != expected:
                msg = f"Normalization input changed: {name}"
                raise ValueError(msg)

    def read() -> dict:
        entry = json.loads(destination.read_text())
        if entry.get("binding") != binding or identity(entry["value"]) != entry.get("sha256"):
            msg = "Normalization cache identity or output checksum changed"
            raise ValueError(msg)
        return entry["value"]

    authenticate()
    if destination.exists():
        value = read()
    else:
        value = build()
        authenticate()
        entry = {"binding": binding, "value": value, "sha256": identity(value)}
        # Link publishes a fully written file atomically without replacing a
        # concurrent producer's immutable entry. Interrupted temporary files
        # cannot be mistaken for cache entries.
        with tempfile.NamedTemporaryFile(mode="w", dir=cache_dir, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                json.dump(entry, handle, allow_nan=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    if identity(read()) != identity(value):
                        msg = "Concurrent normalization produced conflicting output"
                        raise ValueError(msg) from None
            finally:
                temporary.unlink(missing_ok=True)
    authenticate()
    return value
