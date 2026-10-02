"""Bounded subprocess workers and explicit CPU provenance for benchmark campaigns."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
PROFILE = "hopper-epyc9754-1c-v1"


def hardware() -> dict:
    """Record observed CPU placement, excluding volatile job IDs from resume identity."""
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    cpuinfo = Path("/proc/cpuinfo")
    models = (
        [
            line.split(":", 1)[1].strip()
            for line in cpuinfo.read_text().splitlines()
            if line.startswith("model name")
        ]
        if cpuinfo.exists()
        else []
    )
    return {
        "cpu_model": models[0] if models else platform.processor(),
        "affinity": affinity,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def verify_profile(profile: str) -> None:
    """Require the declared CPU, binding, single-thread environment, and exclusive Slurm job."""
    if profile == "diagnostic":
        return
    observed = hardware()
    if (
        profile != PROFILE
        or "EPYC 9754" not in observed["cpu_model"]
        or len(observed["affinity"]) != 1
    ):
        message = "Timing profile requires an AMD EPYC 9754 worker pinned to one CPU."
        raise ValueError(message)
    if any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
        message = "Timing profile requires all declared thread environment limits to equal one."
        raise ValueError(message)
    job = os.environ.get("SLURM_JOB_ID")
    # A task can share the array-root job ID, whose query returns multiple siblings.
    array, task = os.environ.get("SLURM_ARRAY_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    if array and task:
        job = f"{array}_{task}"
    if not job:
        message = "Timing profile requires an exclusive Slurm allocation."
        raise ValueError(message)
    description = subprocess.check_output(  # noqa: S603 -- fixed read-only Slurm command
        ["scontrol", "show", "job", job, "-o"],  # noqa: S607
        text=True,
        timeout=10,
    )
    if len(description.strip().splitlines()) != 1:
        message = "Slurm returned an ambiguous job allocation."
        raise ValueError(message)
    fields = dict(part.split("=", 1) for part in description.split() if "=" in part)
    if fields.get("OverSubscribe") != "NO" and fields.get("Shared") != "0":
        message = "Slurm did not confirm an exclusive allocation."
        raise ValueError(message)
    if fields.get("NumNodes") != "1" or not fields.get("NodeList"):
        message = "Timing profile requires one exclusively allocated node."
        raise ValueError(message)
    node = subprocess.check_output(  # noqa: S603 -- fixed read-only Slurm command
        ["scontrol", "show", "node", fields["NodeList"], "-o"],  # noqa: S607
        text=True,
        timeout=10,
    )
    node_fields = dict(part.split("=", 1) for part in node.split() if "=" in part)
    if (
        fields.get("NumCPUs") != node_fields.get("CPUTot")
        or node_fields.get("ThreadsPerCore") != "1"
    ):
        message = "Timing profile requires the complete physical node allocation without SMT."
        raise ValueError(message)


def isolated(request: dict, timeout: float, memory_gb: float | None) -> dict:
    """Run one cell in a process group; kill the whole group when its wall limit expires."""
    with tempfile.TemporaryDirectory(prefix="shapiq-worker-") as directory:
        root = Path(directory)
        source, destination = root / "request.json", root / "response.json"
        request = {**request, "memory_gb": memory_gb}
        source.write_text(json.dumps(request))
        env = {**os.environ, **dict.fromkeys(THREAD_VARIABLES, "1"), "PYTHONHASHSEED": "0"}
        start = time.perf_counter()
        with (root / "worker.log").open("w+b") as log:
            process = subprocess.Popen(  # noqa: S603 -- fixed Python module, JSON arguments
                [sys.executable, "-m", "shapiq_benchmark.execution", str(source), str(destination)],
                env=env,
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
                return {
                    "status": "failed",
                    "error": "TimeoutError: worker wall-time limit exceeded.",
                    "queries": None,
                    "requested_queries": None,
                    "seconds": None,
                    "wall_seconds": time.perf_counter() - start,
                }
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if not destination.exists():
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - 2000))
                return {
                    "status": "failed",
                    "error": f"WorkerError: exit {process.returncode}; {log.read(2000).decode(errors='replace')}",
                    "queries": None,
                    "requested_queries": None,
                    "seconds": None,
                    "wall_seconds": time.perf_counter() - start,
                }
        try:
            result: dict = json.loads(destination.read_text())
            if not isinstance(result, dict) or result.get("status") not in ("ok", "failed"):
                message = "Invalid worker response."
                raise ValueError(message)  # noqa: TRY301
            json.dumps(result, allow_nan=False)
        except (ValueError, OSError) as error:
            result = {
                "status": "failed",
                "error": f"WorkerError: invalid response ({type(error).__name__}).",
                "queries": None,
                "requested_queries": None,
                "seconds": None,
            }
        result["wall_seconds"] = time.perf_counter() - start
        return result


def worker(source: Path, destination: Path) -> None:
    """Apply resource limits before importing scientific libraries or candidate code."""
    request = json.loads(source.read_text())
    result: dict = {"status": "failed", "queries": None, "requested_queries": None, "seconds": None}
    try:
        import resource

        if request["memory_gb"] is not None:
            size = int(request["memory_gb"] * 1024**3)
            resource.setrlimit(resource.RLIMIT_AS, (size, size))
        from threadpoolctl import threadpool_info, threadpool_limits

        from shapiq_benchmark.runner import (
            digest,
            load_snapshot,
            provenance,
            run_one,
        )

        if "authenticated_game" in request:
            # The locked parent authenticated the complete snapshot once. The
            # worker receives only its selected game and authenticates that file.
            root = Path(request["artifact_root"]).resolve()
            game = request["authenticated_game"]
            artifact = (root / game["artifact"]).resolve()
            if (
                game["id"] != request["game_id"]
                or not artifact.is_relative_to(root)
                or digest(artifact) != request["artifact_sha256"]
            ):
                message = "Selected game artifact changed during the campaign."
                raise ValueError(message)  # noqa: TRY301
        else:
            # Retain compatibility with previously written isolated requests.
            snapshot, root = load_snapshot(Path(request["snapshot"]))
            if snapshot["snapshot_id"] != request["expected_snapshot_id"]:
                message = "Snapshot changed during the campaign."
                raise ValueError(message)  # noqa: TRY301
            game = next(game for game in snapshot["games"] if game["id"] == request["game_id"])
        if provenance()["source_sha256"] != request["expected_source_hash"]:
            message = "Snapshot or benchmark source changed during the campaign."
            raise ValueError(message)  # noqa: TRY301
        if (
            request.get("candidate")
            and digest(Path(request["candidate"].rsplit(":", 1)[0])) != request["candidate_sha256"]
        ):
            message = "Candidate source changed during the campaign."
            raise ValueError(message)  # noqa: TRY301
        with threadpool_limits(limits=1):
            verify_profile(request["timing_profile"])
            result = run_one(
                game,
                root,
                request["method"],
                request["budget"],
                request["seed"],
                request.get("candidate"),
                parameters=request.get("method_parameters"),
            )
            pools = [
                {key: pool.get(key) for key in ("internal_api", "num_threads", "version")}
                for pool in threadpool_info()
            ]
            if request["timing_profile"] != "diagnostic" and any(
                pool["num_threads"] != 1 for pool in pools
            ):
                message = "Native thread pools did not preserve the required single-thread limit."
                raise ValueError(message)  # noqa: TRY301
            result["worker"] = {
                **hardware(),
                "thread_pools": pools,
                "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
                "process_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
                + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            }
    except Exception as error:  # noqa: BLE001 -- return import/resource/setup failures as data
        result.update(
            status="failed", nmse=None, mse=None, error=f"{type(error).__name__}: {error}"
        )
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, allow_nan=False))
    temporary.replace(destination)


if __name__ == "__main__":
    worker(Path(sys.argv[1]), Path(sys.argv[2]))
