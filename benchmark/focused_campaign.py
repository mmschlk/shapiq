"""Qualify a frozen focused suite and account for bounded Slurm reservations.

This command never submits jobs. Book every task's allocated cores times its
Slurm wall-time limit before submission; reclaim unused time only after verified
terminal accounting. Active or missing accounting retains the full reservation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path


def budget_status(reservations: list[dict], settled: dict, cap_hours: float = 2048) -> dict:
    """Fail closed on ambiguous tasks and sum settled use plus outstanding reservations.

    Reservations contain ``id``, ``cpus`` and ``seconds`` for each exact task;
    ``settled`` maps verified terminal task IDs to Slurm CPUTimeRAW seconds.
    The caller must verify terminal state and final accounting before settling.
    """
    if not math.isfinite(cap_hours) or cap_hours <= 0:
        message = "The allocated CPU-hour cap must be positive and finite"
        raise ValueError(message)
    reserved = {}
    for row in reservations:
        task, cpus, seconds = row["id"], row["cpus"], row["seconds"]
        if (
            not isinstance(task, str)
            or not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", task)
            or task in reserved
            or type(cpus) is not int
            or cpus < 1
            or isinstance(seconds, bool)
            or not isinstance(seconds, int | float)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            message = "Reservations need unique exact task IDs and positive CPU/time limits"
            raise ValueError(message)
        reserved[task] = cpus * seconds
    if any(task.split("_")[0] in reserved for task in reserved if "_" in task):
        message = "Do not reserve both an array root and its tasks"
        raise ValueError(message)
    if not set(settled) <= set(reserved):
        message = "Terminal accounting contains an unreserved task"
        raise ValueError(message)
    for seconds in settled.values():
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, int | float)
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            message = "Settled CPUTimeRAW values must be finite nonnegative seconds"
            raise ValueError(message)
    used = math.fsum(settled.values())
    outstanding = math.fsum(value for task, value in reserved.items() if task not in settled)
    total = used + outstanding
    return {
        "cap_cpu_hours": cap_hours,
        "settled_cpu_hours": used / 3600,
        "reserved_cpu_hours": outstanding / 3600,
        "committed_cpu_hours": total / 3600,
        "remaining_cpu_hours": (cap_hours * 3600 - total) / 3600,
        "within_cap": total <= cap_hours * 3600,
    }


def immutable_json(path: Path, value: dict) -> None:
    """Create once; a changed request cannot reuse an existing pilot directory."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            message = f"Changed frozen request: {path.name}"
            raise ValueError(message)
        return
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def slurm_allocation(workers: int) -> tuple[str, dict]:
    """Authenticate one exact task's shared CPU allocation, without requiring a full node."""
    job = os.environ["SLURM_JOB_ID"]
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is not None:
        job = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{task}"
    if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job):
        message = "Slurm task identity must be numeric"
        raise ValueError(message)
    rows = (
        subprocess.check_output(  # noqa: S603 -- validated task ID, read-only Slurm command
            ["scontrol", "show", "job", job, "-o"],  # noqa: S607 -- fixed read-only Slurm executable
            text=True,
            timeout=10,
        )
        .strip()
        .splitlines()
    )
    if len(rows) != 1:
        message = "Slurm must return exactly one allocation for the selected task"
        raise ValueError(message)
    fields = dict(part.split("=", 1) for part in rows[0].split() if "=" in part)
    observed_id = (
        f"{fields.get('ArrayJobId')}_{fields.get('ArrayTaskId')}"
        if task is not None
        else fields.get("JobId")
    )
    tres = dict(
        part.split("=", 1) for part in fields.get("AllocTRES", "").split(",") if "=" in part
    )
    gpu = any("gpu" in key.lower() and value != "0" for key, value in tres.items())
    gpu = gpu or any("gpu" in fields.get(key, "").lower() for key in ("Gres", "TresPerNode"))
    if (
        observed_id != job
        or fields.get("NumNodes") != "1"
        or fields.get("JobState") != "RUNNING"
        or not workers <= int(fields.get("NumCPUs", "0")) <= 128
        or int(tres.get("cpu", "0")) != int(fields.get("NumCPUs", "0"))
        or gpu
    ):
        message = "Preflight requires its running single-node CPU allocation with no GPUs"
        raise ValueError(message)
    stable_fields = (
        "JobId",
        "ArrayJobId",
        "ArrayTaskId",
        "NumNodes",
        "NumCPUs",
        "NodeList",
        "AllocTRES",
        "Gres",
        "TresPerNode",
    )
    return job, {key: fields[key] for key in stable_fields if key in fields}


def preflight(args: argparse.Namespace) -> None:
    """Reuse existing bounded qualification workers without full-table preparation."""
    # Import the scientific environment only for compute-node execution.
    if not os.environ.get("SLURM_JOB_ID"):
        message = "Run focused preflight inside a Slurm CPU allocation"
        raise RuntimeError(message)
    job, slurm = slurm_allocation(args.workers)
    from shapiq_benchmark.execution import THREAD_VARIABLES, hardware
    from shapiq_benchmark.qualification import qualify_suite
    from shapiq_benchmark.runner import identity, provenance

    source = json.loads(args.expected_source.read_text())
    if provenance() != source:
        message = "Imported source/environment differs from the frozen expected source"
        raise ValueError(message)
    allocation = hardware()
    if (
        not 1 <= args.workers <= len(allocation["affinity"])
        or "EPYC 9754" not in allocation["cpu_model"]
        or any(os.environ.get(name) != "1" for name in THREAD_VARIABLES)
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        message = "Preflight needs single-threaded EPYC CPU workers and disabled CUDA"
        raise ValueError(message)
    suite = json.loads(args.suite.read_text())
    if suite.get("game_seeds") != [0, 1, 2, 3]:
        message = "Focused preflight must qualify all four construction seeds"
        raise ValueError(message)
    if any(spec.get("device") == "cuda" for spec in suite.get("families", [])):
        message = "Focused preflight is CPU-only"
        raise ValueError(message)
    limits = {
        "max_seconds": args.max_preparation_seconds,
        "pilot_timeout": args.pilot_timeout,
        "structured_timeout": args.structured_timeout,
        "structured_memory_gb": args.memory_gb,
        "workers": args.workers,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    immutable_json(
        args.output / "request.json",
        {"suite_sha256": identity(suite), "source": source, "limits": limits},
    )
    # Record each allocation separately; retries remain visible in CPU accounting.
    immutable_json(args.output / f"allocation-{job}.json", {"hardware": allocation, "slurm": slurm})
    qualified = qualify_suite(suite, args.output, model_cache=args.output / "models", **limits)
    summary = {
        "requested_recipes": len(suite.get("families", [])) + len(suite.get("games", [])),
        "qualified_recipes": len(qualified.get("families", [])) + len(qualified.get("games", [])),
        "exclusions": len(qualified.get("preparation_exclusions", [])),
        "scope": "Preparation preflight only; not complete game tables or estimator results",
    }
    print(json.dumps(summary))  # noqa: T201 -- CLI summary


def main() -> None:
    """Run explicit preflight or report an existing reservation ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pilot = commands.add_parser("preflight")
    pilot.add_argument("suite", type=Path)
    pilot.add_argument("output", type=Path)
    pilot.add_argument("--expected-source", type=Path, required=True)
    pilot.add_argument("--workers", type=int, default=16)
    pilot.add_argument("--max-preparation-seconds", type=float, required=True)
    pilot.add_argument("--pilot-timeout", type=float, required=True)
    pilot.add_argument("--structured-timeout", type=float, required=True)
    pilot.add_argument("--memory-gb", type=float, default=12)
    budget = commands.add_parser("budget")
    budget.add_argument("ledger", type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args)
    else:
        ledger = json.loads(args.ledger.read_text())
        status = budget_status(
            ledger["reservations"], ledger.get("settled", {}), ledger.get("cap_cpu_hours", 2048)
        )
        print(json.dumps(status, indent=2))  # noqa: T201 -- CLI report
        if not status["within_cap"]:
            message = "Allocated CPU-hour commitments exceed the campaign cap"
            raise SystemExit(message)


if __name__ == "__main__":
    main()
