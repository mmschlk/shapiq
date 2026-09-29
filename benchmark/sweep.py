"""Distribute disjoint game panels over fixed CPU cores; rerun unchanged to resume."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("snapshot", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--workers", type=int, default=128)
parser.add_argument("--timeout", type=float, default=120)
parser.add_argument("--seconds", type=float, default=1620)
args = parser.parse_args()
manifest = args.snapshot / "snapshot.json"
snapshot = json.loads(manifest.read_text())
cpus = sorted(os.sched_getaffinity(0))[: args.workers]
if not cpus or args.workers <= 0:
    message = "At least one allocated CPU is required."
    raise ValueError(message)
ids = [game["id"] for game in snapshot["games"]]
random.Random(0).shuffle(ids)  # noqa: S311 -- deterministic experiment scheduling  # Avoid assigning only SV or only unsupported targets to one core.
args.output.mkdir(parents=True, exist_ok=True)
allocation = {"snapshot_id": snapshot["snapshot_id"], "cpus": cpus, "game_ids": ids}
plan = args.output / "allocation.json"
if plan.exists() and json.loads(plan.read_text()) != allocation:
    message = "Resume requires the same snapshot and allocated core IDs."
    raise ValueError(message)
plan.write_text(json.dumps(allocation, indent=2) + "\n")


def shard(slot: int) -> int:
    """A fixed shard retains a stable hardware identity across resumptions."""
    selected = ids[slot :: len(cpus)]
    if not selected:
        return 0
    output = args.output / f"shard-{slot:03}"
    command = [
        "taskset",
        "-c",
        str(cpus[slot]),
        sys.executable,
        "-m",
        "shapiq_benchmark.runner",
        "--snapshot",
        str(args.snapshot),
        "--output",
        str(output),
        "--games",
        *selected,
        "--timeout",
        str(args.timeout),
        "--memory-gb",
        "12",
        "--max-seconds",
        str(args.seconds),
    ]
    if (output / "results.json").exists():
        command.append("--resume")
    with (args.output / f"shard-{slot:03}.log").open("a") as log:
        return subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)  # noqa: S603 -- argument list, no shell


with concurrent.futures.ThreadPoolExecutor(max_workers=len(cpus)) as pool:
    codes = list(pool.map(shard, range(len(cpus))))
if any(codes):
    message = f"{sum(code != 0 for code in codes)} shards failed; see their logs."
    raise SystemExit(message)
