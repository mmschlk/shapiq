"""Stop an owned GPU command when sustained utilization falls below 80%.

This guard measures occupancy, not speedup. It never creates artificial GPU work.
Only CUDA-visible whole GPUs are sampled; MIG and unavailable telemetry fail closed.
Use after qualifying the recipe/backend, with a new receipt path for each attempt::

    python benchmark/gpu_utilization.py --receipt gpu.json -- python prepare.py
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections import deque
from pathlib import Path
from typing import NoReturn

LOW_UTILIZATION = 75
MONITOR_FAILURE = 76
INSUFFICIENT_OBSERVATION = 77


def fail(message: str) -> NoReturn:
    """Reject unsafe or unverifiable GPU monitoring."""
    raise ValueError(message)


def visible_uuids() -> list[str]:
    """Resolve CUDA ordinals using the driver, never host nvidia-smi index order."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = visible.split(",")
    if not visible or any(not (x.isdecimal() or x.startswith("GPU-")) for x in tokens):
        fail("Explicit numeric or GPU UUID CUDA_VISIBLE_DEVICES is required; no MIG.")
    driver = ctypes.CDLL("libcuda.so.1")

    def check(status: int) -> None:
        if status:
            fail(f"CUDA device discovery failed with status {status}.")

    check(driver.cuInit(0))
    count = ctypes.c_int()
    check(driver.cuDeviceGetCount(ctypes.byref(count)))
    if count.value != len(tokens):
        fail("CUDA-visible device count does not match the explicit allocation.")
    identifiers = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        check(driver.cuDeviceGet(ctypes.byref(device), ordinal))
        raw = (ctypes.c_ubyte * 16)()
        # v2 retains the instance identity on MIG; such an ID will not pass the
        # whole-GPU identity/MIG check below. Do not fall back to a parent UUID.
        check(driver.cuDeviceGetUuid_v2(ctypes.byref(raw), device))
        identifiers.append("GPU-" + str(uuid.UUID(bytes=bytes(raw))))
    if not identifiers or len(set(identifiers)) != len(identifiers):
        fail("CUDA allocation identities must be nonempty and unique.")
    return identifiers


def sample(identifiers: list[str]) -> dict[str, float]:
    """Read only allocated UUIDs, rejecting missing metrics and MIG-enabled GPUs."""
    program = shutil.which("nvidia-smi")
    if program is None:
        fail("nvidia-smi is required for utilization monitoring.")
    output = subprocess.check_output(  # noqa: S603 -- fixed program and validated UUIDs
        [
            program,
            "--id=" + ",".join(identifiers),
            "--query-gpu=uuid,utilization.gpu,mig.mode.current",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    values = {}
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3 or fields[0] not in identifiers or fields[0] in values:
            fail("GPU telemetry returned an unexpected or duplicate identity.")
        if fields[2] not in {"Disabled", "[N/A]", "N/A", "[Not Supported]"}:
            fail("MIG-enabled devices are not supported by the utilization guard.")
        value = float(fields[1])
        if not math.isfinite(value) or not 0 <= value <= 100:
            fail("GPU utilization must be a finite percentage.")
        values[fields[0]] = value
    if set(values) != set(identifiers):
        fail("GPU utilization is missing for an allocated device.")
    return values


def rolling_means(samples: deque) -> dict[str, float]:
    """Average each allocated GPU separately so a busy device cannot mask an idle one."""
    return {
        identifier: sum(row["utilization"][identifier] for row in samples) / len(samples)
        for identifier in samples[0]["utilization"]
    }


def stop_owned_group(child: subprocess.Popen, grace: float = 15) -> None:
    """Terminate only the session/process group created for our command."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        child.poll()
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait()


def run(args: argparse.Namespace) -> int:
    """Run the command and persist bounded, truthful telemetry even on failure."""
    count = math.ceil(args.window / args.interval) + 1
    samples = deque(maxlen=count)
    receipt = {
        "threshold_percent": args.threshold,
        "warmup_seconds": args.warmup,
        "window_seconds": args.window,
        "interval_seconds": args.interval,
        "policy": "Per allocated GPU, including I/O and CPU stalls after startup warmup.",
        "command_completed": False,
        "utilization_verified": False,
        "samples_observed": 0,
        "started_at_unix": time.time(),
    }
    child = None
    started = time.monotonic()
    exit_code = MONITOR_FAILURE

    def persist() -> None:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["recent_samples"] = list(samples)
        temporary = args.receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(args.receipt)

    try:
        # Isolate CUDA driver initialization and bound discovery time. The parent
        # never owns a CUDA context or changes the command's device mapping.
        identifiers = json.loads(
            subprocess.check_output(  # noqa: S603
                [sys.executable, str(Path(__file__).resolve()), "--resolve-devices"],
                text=True,
                timeout=30,
            )
        )
        receipt["gpu_uuids"] = identifiers
        sample(identifiers)  # Fail before launching if identity/telemetry is invalid.
        child = subprocess.Popen(args.command, start_new_session=True)  # noqa: S603
        started = time.monotonic()
        while child.poll() is None:
            sampled_at = time.monotonic()
            utilization = sample(identifiers)
            elapsed = sampled_at - started
            if elapsed >= args.warmup:
                if samples and elapsed - samples[-1]["elapsed_seconds"] > 2 * args.interval:
                    fail("GPU telemetry sampling was interrupted.")
                samples.append({"elapsed_seconds": elapsed, "utilization": utilization})
                receipt["samples_observed"] += 1
                receipt["rolling_mean_percent"] = rolling_means(samples)
                if len(samples) == count:
                    if min(receipt["rolling_mean_percent"].values()) < args.threshold:
                        receipt["reason"] = "sustained_low_utilization"
                        exit_code = LOW_UTILIZATION
                        break
                    receipt["utilization_verified"] = True
            persist()
            time.sleep(max(0, args.interval - (time.monotonic() - sampled_at)))
        else:
            receipt["command_completed"] = True
            receipt["command_returncode"] = child.returncode
            if child.returncode:
                receipt["reason"] = "command_failed"
                exit_code = child.returncode if child.returncode > 0 else 128 - child.returncode
            elif not receipt["utilization_verified"]:
                receipt["reason"] = "insufficient_observation"
                exit_code = INSUFFICIENT_OBSERVATION
            else:
                receipt["reason"] = "completed"
                exit_code = 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        receipt["reason"] = "telemetry_failure"
        receipt["error"] = str(error)
    except KeyboardInterrupt:
        receipt["reason"] = "interrupted"
        exit_code = 130
    finally:
        if child is not None:
            stop_owned_group(child)
            receipt["command_returncode"] = child.returncode
        if exit_code:
            receipt["utilization_verified"] = False
        receipt["guard_returncode"] = exit_code
        persist()
    return exit_code


def main() -> int:
    """Parse the monitor settings without permitting a threshold below 80%."""
    if sys.argv[1:] == ["--resolve-devices"]:
        print(json.dumps(visible_uuids()))  # noqa: T201 -- internal discovery protocol
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=80)
    parser.add_argument("--warmup", type=float, default=120)
    parser.add_argument("--window", type=float, default=300)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command.pop(0)
    if (
        not args.command
        or not 80 <= args.threshold <= 100
        or args.warmup < 0
        or args.interval <= 0
        or args.window < args.interval
        or not all(
            math.isfinite(x) for x in (args.threshold, args.warmup, args.interval, args.window)
        )
    ):
        parser.error("Need a command, threshold 80..100, and finite nonnegative timing settings.")
    if args.receipt.exists():
        parser.error("Use a new receipt path; existing attempt receipts must be preserved.")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)

    def interrupt(*_: object) -> NoReturn:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
