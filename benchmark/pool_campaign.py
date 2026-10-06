"""Pipeline frozen instance preparation and evaluation on an admitted CPU allocation.

The external bootstrap stages dependencies before this stdlib-only launcher starts.
An atomic directory claims each instance once across hosts; claims are never stolen.
Partial preparation, launch intents and compact result journals remain for audit.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import resource
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

THREADS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def digest(path: Path) -> str:
    """Hash a frozen input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict, *, replace: bool = False) -> None:
    """Durably create evidence; only progress files are replaceable."""
    target = path.with_name(path.name + f".{os.getpid()}.tmp") if replace else path
    with target.open("w" if replace else "x") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if replace:
        target.replace(path)


def append(path: Path, value: dict) -> None:
    """Flush one durable journal entry."""
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def tasks(suite: dict) -> list[dict]:
    """Group native targets by construction, then interleave dataset/application strata."""
    from prepare_matrix import cases

    grouped = {}
    for task in cases(suite):
        recipe = task["spec"].get("basecase_id", task["spec"]["id"])
        key = (task["kind"], recipe, task["seed"])
        if key in grouped:
            if task["kind"] != "structured":
                message = "Repeated family instance"
                raise ValueError(message)
            grouped[key]["specs"].append(task["spec"])
        else:
            grouped[key] = {**task, "recipe": recipe, "specs": [task["spec"]]}
    if not grouped:
        message = "Empty instance inventory"
        raise ValueError(message)
    recipes = {row["id"]: row for row in suite.get("focused_design", {}).get("recipes", [])}
    buckets = {}
    for task in grouped.values():
        spec = task["spec"]
        application = recipes.get(task["recipe"], {}).get(
            "application", spec.get("family", spec.get("oracle"))
        )
        buckets.setdefault((spec.get("dataset", ""), application), []).append(task)
    rng = random.Random(0)  # noqa: S311 -- reproducible queue ordering
    keys = sorted(buckets, key=str)
    rng.shuffle(keys)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    result = []
    for offset in range(max(map(len, buckets.values()))):
        for key in keys:
            if offset < len(buckets[key]):
                result.append({**buckets[key][offset], "case": len(result)})
    return result


def fragment(suite: dict, task: dict) -> dict:
    """Keep only one construction and its full target panel."""
    from prepare_matrix import task_suite

    result = task_suite(suite, task)
    if task["kind"] == "structured":
        result["games"] = task["specs"]
    if "focused_design" in result:
        selected = [
            row for row in result["focused_design"]["recipes"] if row["id"] == task["recipe"]
        ]
        if len(selected) != 1:
            message = "Missing or ambiguous selected recipe"
            raise ValueError(message)
        result["focused_design"] = {**result["focused_design"], "recipes": selected}
    return result


def claim(root: Path, task: dict, owner: dict) -> Path | None:
    """Mkdir is cross-host exclusion; BeeGFS flock is deliberately not used."""
    directory = root / f"case-{task['case']:06d}"
    try:
        directory.mkdir()
    except FileExistsError:
        return None
    write(directory / "claim.json", {**owner, "task": task, "claimed_at": time.time()})
    return directory


def bounded(command: list[str], logfile: Path, seconds: float) -> dict:
    """Preparation children cannot exceed the wall bound on their one CPU."""
    started = time.monotonic()
    with logfile.open("xb") as log:
        child = subprocess.Popen(  # noqa: S603 -- fixed current-interpreter command
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        timed_out = False
        try:
            child.wait(timeout=max(0.01, seconds))
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Also stop descendants if the leader failed or reached its deadline.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    return {
        "returncode": child.returncode,
        "timed_out": timed_out,
        "wall_seconds": time.monotonic() - started,
    }


def bind(cpu: int, memory_gb: float) -> None:
    """Constrain scientific processes before imports."""
    os.sched_setaffinity(0, {cpu})
    size = int(memory_gb * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (size, size))
    if any(os.environ.get(key) != "1" for key in THREADS):
        message = "Bootstrap must set all native thread limits before Python starts"
        raise ValueError(message)


def authenticate(args: argparse.Namespace) -> tuple[dict, dict]:
    """Verify admitted input bytes and reject unsafe shared registries."""
    suite = json.loads(args.suite.read_text())
    source = json.loads(args.source.read_text())
    if digest(args.suite) != args.suite_sha256 or digest(args.source) != args.source_sha256:
        message = "Frozen suite/source changed"
        raise ValueError(message)
    if suite.get("duplicate_registry"):
        message = "Pooled execution cannot use a cross-host duplicate registry"
        raise ValueError(message)
    return suite, source


def scientific(args: argparse.Namespace) -> None:
    """Only freshly started, pre-staged, CPU-bound children import scientific code."""
    bind(args.cpu, args.memory_gb)
    _, expected = authenticate(args)
    from shapiq_benchmark import execution, runner

    if runner.provenance() != expected:
        message = "Scientific runtime differs from the frozen source/environment"
        raise ValueError(message)
    directory = args.task_directory
    fragment = directory / "suite.json"
    if args.phase == "prepare":
        from shapiq_benchmark.prepare import prepare

        snapshot = prepare(fragment, directory / "prepared")
        if not snapshot["games"] or any(
            row["status"] != "measured" for row in snapshot.get("coverage", [])
        ):
            message = "Preparation did not qualify; preserved coverage needs review"
            raise ValueError(message)
        write(
            directory / "prepared.json",
            {
                "snapshot_id": snapshot["snapshot_id"],
                "snapshot_sha256": digest(directory / "prepared" / "snapshot.json"),
                "targets": len(snapshot["games"]),
                "source": expected,
            },
        )
        return
    original = execution.isolated
    seen: set[tuple] = set()
    intents = directory / "intents.jsonl"
    if intents.exists():
        message = "An evaluation attempt already exists; automatic retries are forbidden"
        raise ValueError(message)

    def isolated(request: dict, timeout: float, memory_gb: float | None) -> dict:
        key = tuple(request[name] for name in ("game_id", "method", "budget", "seed"))
        if key in seen:
            message = "Repeated evaluation cell"
            raise ValueError(message)
        if time.time() + timeout + 15 >= args.deadline:
            message = "Allocation deadline reached before launching a cell"
            raise TimeoutError(message)
        seen.add(key)
        intent = {
            name: request[name]
            for name in (
                "game_id",
                "method",
                "budget",
                "seed",
                "expected_snapshot_id",
                "expected_source_hash",
                "method_parameters",
            )
        }
        intent.update(
            sequence=len(seen),
            timeout=timeout,
            created_at=time.time(),
            slurm_job_id=os.environ["SLURM_JOB_ID"],
            cpu=args.cpu,
        )
        append(intents, intent)  # fsync completes before execution can begin
        result = original(request, timeout, memory_gb)
        append(
            directory / "responses.jsonl",
            {"sequence": len(seen), "cell": list(key), "response": result},
        )
        return result

    execution.isolated = isolated
    try:
        remaining = args.deadline - time.time() - 30
        if remaining <= 120:
            message = "Too little allocation time remains for evaluation"
            raise TimeoutError(message)
        result = runner.run(
            directory / "prepared",
            directory / "results",
            timeout=30,
            memory_gb=args.memory_gb,
            max_seconds=remaining,
            timing_profile="diagnostic",
        )
        write(directory / "evaluation.json", result["campaign"])
    finally:
        execution.isolated = original


def child_command(args: argparse.Namespace, phase: str, directory: Path) -> list[str]:
    """Start a fresh interpreter in the inherited staged environment."""
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--suite",
        str(args.suite),
        "--source",
        str(args.source),
        "--suite-sha256",
        args.suite_sha256,
        "--source-sha256",
        args.source_sha256,
        "--output",
        str(args.output),
        "--phase",
        phase,
        "--cpu",
        str(args.cpu),
        "--memory-gb",
        str(args.memory_gb),
        "--deadline",
        str(args.deadline),
        "--task-directory",
        str(directory),
    ]


def worker(args: argparse.Namespace) -> None:
    """Claim and pipeline whole instances until the allocation deadline."""
    bind(args.cpu, args.memory_gb)
    suite, _ = authenticate(args)
    owner = {
        "job": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(),
        "cpu": args.cpu,
        "suite_sha256": args.suite_sha256,
        "source_sha256": args.source_sha256,
    }
    inventory = tasks(suite)
    # Spread the initial probes; every worker still traverses the full inventory.
    offset = (int(owner["job"]) * args.workers + args.worker_index) % len(inventory)
    inventory = inventory[offset:] + inventory[:offset]
    progress = args.output / "workers" / f"{owner['job']}-{args.cpu}.json"
    completed = 0
    for task in inventory:
        if time.time() + 180 >= args.deadline:
            break
        directory = claim(args.output / "tasks", task, owner)
        if directory is None:
            continue
        write(directory / "suite.json", fragment(suite, task))
        status = {**owner, "case": task["case"], "phase": "preparing", "finished_tasks": completed}
        write(progress, status, replace=True)
        prep_limit = min(args.preparation_seconds, args.deadline - time.time() - 60)
        prep = bounded(
            child_command(args, "prepare", directory), directory / "prepare.log", prep_limit
        )
        write(directory / "preparation-execution.json", prep)
        if prep["returncode"] or not (directory / "prepared.json").exists():
            # This is an explicit operational/qualification outcome, never a fabricated score.
            outcome = {
                "status": "preparation_timeout" if prep["timed_out"] else "preparation_failed",
                "requires_review": True,
                "preparation": prep,
            }
        elif time.time() + 180 >= args.deadline:
            outcome = {"status": "prepared_evaluation_not_started", "remaining": True}
        else:
            write(progress, {**status, "phase": "evaluating"}, replace=True)
            evaluation = bounded(
                child_command(args, "evaluate", directory),
                directory / "evaluate.log",
                args.deadline - time.time() - 5,
            )
            write(directory / "evaluation-execution.json", evaluation)
            checkpoint = directory / "evaluation.json"
            complete = checkpoint.exists() and json.loads(checkpoint.read_text())["complete"]
            outcome = {
                "status": "complete"
                if complete and evaluation["returncode"] == 0
                else "evaluation_partial",
                "remaining": not complete,
                "evaluation": evaluation,
            }
        write(directory / "outcome.json", {**outcome, "finished_at": time.time()})
        completed += 1
        write(progress, {**status, "phase": "idle", "finished_tasks": completed}, replace=True)
    write(progress, {**owner, "phase": "stopped", "finished_tasks": completed}, replace=True)


def main() -> None:
    """Launch one bound worker per allocated CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite-sha256")
    parser.add_argument("--source-sha256")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=71400)
    parser.add_argument("--preparation-seconds", type=float, default=7200)
    parser.add_argument("--memory-gb", type=float, default=8)
    parser.add_argument(
        "--phase", choices=("pool", "worker", "prepare", "evaluate"), default="pool"
    )
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--task-directory", type=Path)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or not 1 <= args.workers <= 64
        or any(
            not math.isfinite(x) or x <= 0
            for x in (args.seconds, args.preparation_seconds, args.memory_gb)
        )
        or args.preparation_seconds > 7200
        or args.memory_gb > 8
    ):
        parser.error("Require admitted Slurm, 1-64 workers, preparation ≤7200s, memory ≤8GiB")
    args.suite, args.source, args.output = (
        args.suite.resolve(),
        args.source.resolve(),
        args.output.resolve(),
    )
    if args.phase != "pool":
        if args.cpu is None or args.deadline is None:
            parser.error("Internal worker requires affinity CPU and absolute deadline")
        (worker if args.phase == "worker" else scientific)(args)
        return
    if not args.suite_sha256 or not args.source_sha256:
        parser.error("External admission must pin --suite-sha256 and --source-sha256")
    authenticate(args)
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < args.workers or any(os.environ.get(key) != "1" for key in THREADS):
        parser.error("Insufficient allocated affinity or missing bootstrap thread limits")
    for name in ("tasks", "workers", "allocations"):
        (args.output / name).mkdir(parents=True, exist_ok=True)
    allocation = args.output / "allocations" / os.environ["SLURM_JOB_ID"]
    allocation.mkdir()  # no duplicate launch in the same allocation
    args.deadline = time.time() + args.seconds
    write(
        allocation / "started.json",
        {
            "job": os.environ["SLURM_JOB_ID"],
            "hostname": socket.gethostname(),
            "cpus": cpus[: args.workers],
            "deadline": args.deadline,
            "suite_sha256": args.suite_sha256,
            "source_sha256": args.source_sha256,
            "workers": args.workers,
        },
    )
    children = []
    for index, cpu in enumerate(cpus[: args.workers]):
        args.cpu = cpu
        command = child_command(args, "worker", allocation) + [  # noqa: RUF005
            "--workers",
            str(args.workers),
            "--worker-index",
            str(index),
            "--preparation-seconds",
            str(args.preparation_seconds),
        ]
        log = (allocation / f"worker-{cpu}.log").open("xb")
        children.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))  # noqa: S603
    codes = []
    for child, log in children:
        codes.append(child.wait())
        log.close()
    write(allocation / "finished.json", {"returncodes": codes, "finished_at": time.time()})
    if any(codes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
