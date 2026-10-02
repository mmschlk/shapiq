"""Wake an authorized Codex thread on job completion or an idle implementation stage.

This watcher never submits Slurm jobs or publishes results. The resumed agent
reads the durable handoff, audits the current phase, and starts the next one.
Run with a campaign directory containing watch.json and WAKEUP.md. Acknowledge
each delivered wake-up with --acknowledge; refresh this heartbeat while working.
"""

from __future__ import annotations

import argparse
import fcntl
import json
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
    """Require accounting for every configured job; a missing row is not completion."""
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
        return "A campaign job finished or failed; verify its actual outputs and advance the phase."
    idle = now - state.get("last_progress_at", config["created_at"])
    if (not jobs or all(status in TERMINAL for status in jobs.values())) and idle >= config.get(
        "idle_seconds", 1800
    ):
        return "No active campaign jobs remain; continue implementation, audits, or the next phase."
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
            "User-authorized benchmark phase continuation. "
            + reason
            + " Read "
            + str(root / "WAKEUP.md")
            + " and the campaign state before acting. Acknowledge this delivery with "
            + "benchmark/watch_campaign.py "
            + str(root)
            + " --acknowledge. "
            "Continue through the remaining ROADMAP phases: implement and qualify, queue/resume "
            "the jobs, independently audit each completed phase, publish verified results, "
            "then start the next phase without waiting for another user request. "
            "Do not duplicate submissions or publication, revive cancelled campaigns, or "
            "treat scheduler completion as proof that all planned cells finished. "
            "Stop the watcher when all phases are verified complete or the user pauses/cancels."
        )
        result = subprocess.run(  # noqa: S603 -- trusted local campaign configuration
            [config["codex"], "queue", "--thread", config["thread_id"], "--message", prompt],
            cwd=config["cwd"],
            text=True,
            capture_output=True,
            timeout=60,
            check=True,
        )
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
        print("Queued benchmark phase continuation", flush=True)  # noqa: T201 -- watcher log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--acknowledge", action="store_true")
    args = parser.parse_args()
    check(args.campaign, acknowledge=args.acknowledge)
