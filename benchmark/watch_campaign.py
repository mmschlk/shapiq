"""Wake an authorized Codex thread on job completion or idle campaign work.

This watcher never submits Slurm jobs or publishes results. The resumed agent
reads the durable handoff, verifies outputs, and continues the accepted plan.
Run with a campaign directory containing watch.json and WAKEUP.md. Acknowledge
each delivered wake-up with --acknowledge; refresh this heartbeat while working.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
}


def write_json(path: Path, value: dict) -> None:
    """Replace state atomically so a crash cannot leave a partial JSON document."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def job_states(config: dict) -> dict:
    """Combine terminal accounting with expanded live tasks; missing jobs stay unknown."""
    jobs = config.get("jobs", [])
    if not jobs:
        return {}
    output = subprocess.check_output(  # noqa: S603 -- trusted local campaign configuration
        [
            "/usr/bin/sacct",
            "--starttime=" + config["started_on"],
            "-X",
            "--array",
            "-n",
            "-P",
            "-j",
            ",".join(sorted({job.split("_", 1)[0] for job in jobs})),
            "-o",
            "JobID%40,State%40",
        ],
        text=True,
        timeout=30,
    )
    states = {}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) >= 2 and fields[0] in jobs:
            states[fields[0]] = fields[1].split()[0].rstrip("+")
    # sacct --array still compresses pending tasks that have no individual
    # accounting record. squeue expands those tasks before they start. Query
    # our live queue, since filtering on completed root IDs can make squeue fail.
    live = subprocess.check_output(  # noqa: S603 -- fixed executable and current numeric user ID
        [
            "/usr/bin/squeue",
            "--array",
            "--noheader",
            "--user=" + str(os.getuid()),
            "--format=%i|%T",
        ],
        text=True,
        timeout=30,
    )
    for line in live.splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) < 2 or fields[0] not in jobs or not fields[1]:
            continue
        status = fields[1].split()[0].rstrip("+")
        if status not in TERMINAL:
            # A live requeue or COMPLETING state overrides stale accounting.
            # Only sacct establishes terminal completion, never queue absence.
            states[fields[0]] = status
    if set(states) != set(jobs):
        message = "Accounting is incomplete; retry on the next watcher tick"
        raise RuntimeError(message)
    return states


def wake_reason(config: dict, state: dict, jobs: dict, now: float) -> str | None:
    """Deduplicate notifications and provide a heartbeat when no jobs can advance work."""
    if config.get("status") != "active" or state.get("pending_delivery"):
        return None
    terminal = {job: status for job, status in jobs.items() if status in TERMINAL}
    if any(state.get("notified_jobs", {}).get(job) != status for job, status in terminal.items()):
        return "A campaign job finished or failed; verify its actual outputs and continue the accepted plan."
    idle = now - state.get("last_progress_at", config["created_at"])
    if (not jobs or all(status in TERMINAL for status in jobs.values())) and idle >= config.get(
        "idle_seconds", 1800
    ):
        return "No active campaign jobs remain; continue the remaining implementation, runs, or audits."
    if (
        idle >= config.get("idle_seconds", 1800)
        and any(status in TERMINAL - {"COMPLETED"} for status in jobs.values())
        and all(status in TERMINAL | {"PENDING"} for status in jobs.values())
    ):
        return "Failed campaign jobs may block pending dependencies; inspect and repair the remaining jobs."
    heartbeat = config.get("heartbeat_seconds")
    if heartbeat is not None and heartbeat > 0 and idle >= heartbeat:
        return "Scheduled progress check; inspect the checklist and active jobs, then continue authorized work."
    return None


def check(root: Path, *, acknowledge: bool = False) -> None:
    """Queue at most one unacknowledged wake-up, preserving a delivery receipt."""
    root = root.resolve()
    with (root / "watch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        config = json.loads((root / "watch.json").read_text())
        state_path = root / "watch-state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        now = time.time()
        if acknowledge:
            state.update(pending_delivery=False, last_progress_at=now)
            write_json(state_path, state)
            return
        if config.get("status") != "active":
            return
        jobs = job_states(config)
        reason = wake_reason(config, state, jobs, now)
        write_json(root / "watch-status.json", {"checked_at": now, "jobs": jobs, "reason": reason})
        if reason is None:
            return
        prompt = (
            "User-authorized benchmark continuation. "
            + reason
            + " Read "
            + str(root / "WAKEUP.md")
            + " and the campaign state before acting. Acknowledge this delivery with "
            + "benchmark/watch_campaign.py "
            + str(root)
            + " --acknowledge. "
            "Follow the active scientific plan named in the campaign handoff: implement and qualify, queue/resume "
            "the jobs, independently audit completed outputs, and publish the verified cohort. "
            "Continue remaining authorized work without waiting for another user request. "
            "Do not duplicate submissions or publication, revive cancelled campaigns, or "
            "treat scheduler completion as proof that all planned cells finished. "
            "Stop the watcher when the accepted benchmark is verified complete or the user pauses/cancels."
        )
        result = subprocess.run(  # noqa: S603 -- trusted local campaign configuration
            [config["codex"], "queue", "--thread", config["thread_id"], "--message", prompt],
            cwd=config["cwd"],
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode:
            # Keep the delivery unacknowledged so cron retries; preserve the
            # actual CLI diagnosis instead of logging only its exit status.
            message = "Continuation delivery failed: " + result.stderr.strip()
            raise RuntimeError(message)
        state.update(
            pending_delivery=True,
            queued_at=now,
            response=result.stdout,
            notified_jobs={
                **state.get("notified_jobs", {}),
                **{job: value for job, value in jobs.items() if value in TERMINAL},
            },
        )
        write_json(state_path, state)
        print("Queued benchmark continuation", flush=True)  # noqa: T201 -- watcher log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--acknowledge", action="store_true")
    args = parser.parse_args()
    check(args.campaign, acknowledge=args.acknowledge)
