"""Prepare a dataset matrix on fixed CPU cores, resuming completed tables and chunks.

Run from an immutable source checkout. Each task writes to its own directory;
only the parent combines qualified fragments into the final authenticated snapshot.
"""

from __future__ import annotations

# ruff: noqa: PLC0415
# Worker imports follow CPU affinity and address-space limits.
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import random
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path


def task_suite(suite: dict, task: dict) -> dict:
    """Keep estimator settings, but avoid duplicating the full matrix inventory per task."""
    result = {
        key: value
        for key, value in suite.items()
        if not key.startswith("matrix_") and key not in ("families", "games")
    }
    result["game_seeds"] = [task["seed"]]
    result["games"] = [task["spec"]] if task["kind"] == "structured" else []
    # An empty families list selects the wrong preparation path for structured games.
    if task["kind"] != "structured":
        result["families"] = [task["spec"]]
    return result


def cases(suite: dict) -> list[dict]:
    """Retain the serial preparation's family-then-structured ordering."""
    return [
        {"kind": "family", "seed": seed, "spec": spec}
        for seed in suite["game_seeds"]
        for spec in suite["families"]
    ] + [
        {"kind": "structured", "seed": seed, "spec": spec}
        for spec in suite["games"]
        for seed in suite["game_seeds"]
    ]


def execute(task: dict, suite: dict, stage: Path, expected: dict) -> None:
    """Reuse only complete authenticated fragments from this exact source and recipe."""
    from shapiq_benchmark.materialize import prepare_family_chunk
    from shapiq_benchmark.prepare import prepare
    from shapiq_benchmark.runner import load_snapshot, provenance

    if provenance() != expected:
        message = "Source/environment changed since preparation was planned"
        raise ValueError(message)
    output = stage / f"case-{task['case']:04}"
    output.mkdir(exist_ok=True)
    fragment = task_suite(suite, task)
    if (output / "snapshot.json").exists():
        snapshot, _ = load_snapshot(output)
        declared = {
            k: v for k, v in snapshot["suite"].items() if k not in ("budgets", "budgets_by_game")
        }
        if (
            declared
            != {k: v for k, v in fragment.items() if k not in ("budgets", "budgets_by_game")}
            or snapshot["provenance"] != expected
        ):
            message = "Cached fragment belongs to another recipe or source"
            raise ValueError(message)
        if any(row["status"] != "measured" for row in snapshot.get("coverage", [])):
            message = "A cached fragment did not qualify"
            raise ValueError(message)
        return
    if "start" in task:
        prepare_family_chunk(task["spec"], task["seed"], task["start"], output)
        return
    path = output / "suite.json"
    path.write_text(json.dumps(fragment, indent=2) + "\n")
    result = prepare(path, output)
    if any(row["status"] != "measured" for row in result.get("coverage", [])):
        message = "A recipe did not qualify; inspect its local error file"
        raise ValueError(message)


def assemble(
    suite: dict, tasks: list[dict], stage: Path, destination: Path, expected: dict
) -> None:
    """Publish only a complete canonical snapshot; unfinished chunks remain private."""
    from shapiq_benchmark.prepare import write_snapshot
    from shapiq_benchmark.runner import load_snapshot, provenance

    assembling = destination.with_name(destination.name + ".assembling")
    assembling.mkdir(parents=True, exist_ok=True)
    games, coverage, artifacts = [], [], set()
    for task in tasks:
        snapshot, root = load_snapshot(stage / f"case-{task['case']:04}")
        if snapshot["provenance"] != expected:
            message = "Fragment provenance differs"
            raise ValueError(message)
        count = len(suite["targets"]) if task["kind"] == "family" else 1
        entries = snapshot.get(
            "coverage",
            [
                {
                    "family": game["id"],
                    "status": "measured",
                    "source": "src/shapiq_benchmark/games.py",
                    "game_ids": [game["id"]],
                }
                for game in snapshot["games"]
            ],
        )
        if len(snapshot["games"]) != count or any(row["status"] != "measured" for row in entries):
            message = "Incomplete fragment"
            raise ValueError(message)
        games.extend(snapshot["games"])
        coverage.extend(entries)
        for name in snapshot["artifacts"]:
            if name in artifacts or Path(name).name != name:
                message = "Repeated or non-flat artifact name"
                raise ValueError(message)
            artifacts.add(name)
            shutil.copyfile(root / name, assembling / name)
    if len({game["id"] for game in games}) != len(games) or provenance() != expected:
        message = "Duplicate games or changed source"
        raise ValueError(message)
    result = write_snapshot(suite, games, assembling, coverage=coverage)
    if result["provenance"] != expected:
        message = "Source changed while writing snapshot"
        raise ValueError(message)
    load_snapshot(assembling)
    assembling.rename(destination)
    print(  # noqa: T201 -- command-line preparation summary
        json.dumps(
            {"snapshot_id": result["snapshot_id"], "games": len(games), "artifacts": len(artifacts)}
        ),
        flush=True,
    )


def plan_tasks(suite: dict) -> tuple[list[dict], list[dict], int]:
    """Order reusable evaluation tasks before large-table qualification and assembly."""
    from shapiq_benchmark.materialize import CHUNK_SIZE

    specs: list[dict] = [{**task, "case": index} for index, task in enumerate(cases(suite))]
    evaluation, qualification = [], []
    for task in specs:
        if task["kind"] == "family" and task["spec"]["n_players"] > 12:
            evaluation.extend(
                {**task, "start": start}
                for start in range(0, 2 ** task["spec"]["n_players"], CHUNK_SIZE)
            )
            qualification.append(task)
        else:
            evaluation.append(task)
    random.Random(0).shuffle(evaluation)  # noqa: S311 -- reproducible task ordering
    return specs, evaluation + qualification, len(evaluation)


def run_tasks(indices: range, cpus: list[int], args: argparse.Namespace, deadline: float) -> None:
    """Run one phase on pinned cores, retaining completed tasks after a deadline or failure."""

    def slot(indices: range, cpu: int) -> int:
        failed = 0
        for index in indices:
            remaining = deadline - time.monotonic()
            if remaining < 30:
                return failed + 1
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                str(args.suite),
                str(args.stage),
                str(args.destination),
                "--task",
                str(index),
                "--cpu",
                str(cpu),
            ]
            with (args.stage / f"task-{index:05}.log").open("a") as log:
                try:
                    completed = subprocess.run(  # noqa: S603 -- fixed interpreter/script and structured arguments
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=min(28800, remaining),
                        check=False,
                    )
                    failed += completed.returncode != 0
                except subprocess.TimeoutExpired:
                    failed += 1
        return failed

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        failures = sum(pool.map(slot, (indices[i :: len(cpus)] for i in range(len(cpus))), cpus))
    if failures:
        message = f"{failures} unfinished/failed preparation tasks; rerun unchanged to resume"
        raise SystemExit(message)


def main() -> None:
    """Dispatch bounded preparation tasks, then authenticate the complete matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("stage", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=250000)
    parser.add_argument("--task", type=int)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if args.task is not None:
        os.sched_setaffinity(0, {args.cpu})
        resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
        plan = json.loads((args.stage / "plan.json").read_text())
        if plan["driver_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            message = "Preparation driver changed after planning"
            raise ValueError(message)
        execute(plan["tasks"][args.task], plan["suite"], args.stage, plan["provenance"])
        return

    from shapiq_benchmark.runner import provenance, validate_suite

    suite = json.loads(args.suite.read_text())
    validate_suite(suite)
    if args.destination.exists():
        message = "Final snapshot already exists; do not submit preparation twice"
        raise FileExistsError(message)
    expected = provenance()
    if expected["source_dirty"] is not False or not expected["git_commit"]:
        message = "Preparation requires clean committed source"
        raise ValueError(message)
    cpus = sorted(os.sched_getaffinity(0))[: args.workers]
    if args.workers <= 0 or len(cpus) != args.workers or args.seconds <= 0:
        message = "Insufficient allocated cores or invalid preparation limits"
        raise ValueError(message)
    specs, tasks, evaluation_count = plan_tasks(suite)
    plan = {
        "suite": suite,
        "provenance": expected,
        "tasks": tasks,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.stage.mkdir(parents=True, exist_ok=True)
    # Retain this descriptor until the process exits; children never acquire it.
    lock = (args.stage / "prepare.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.stage / "allocation.json").write_text(json.dumps({"cpus": cpus}) + "\n")
    plan_path = args.stage / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        message = "Resume requires the same suite, source and preparation driver"
        raise ValueError(message)
    temporary = plan_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(plan, indent=2) + "\n")
    temporary.replace(plan_path)
    deadline = time.monotonic() + args.seconds

    from shapiq_benchmark.datasets import warm_dataset_caches

    warm_dataset_caches(suite)
    run_tasks(range(evaluation_count), cpus, args, deadline)
    run_tasks(range(evaluation_count, len(tasks)), cpus, args, deadline)
    assemble(suite, specs, args.stage, args.destination, expected)


if __name__ == "__main__":
    main()
