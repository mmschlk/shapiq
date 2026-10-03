"""Run one authenticated batch through qualification, cached preparation and evaluation."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from queue_phases import immutable_write, scripts

from shapiq_benchmark.execution import PROFILE, THREAD_VARIABLES, hardware, verify_profile
from shapiq_benchmark.runner import identity, load_snapshot, provenance


def verify_allocation(batch: dict, step: str, root: Path, workers: int | None = None) -> None:
    """Require legacy isolation by default, or an explicit shared CPU worker allocation."""
    if workers is not None and ((step == "prepare" and batch["device"] == "cuda") or workers < 1):
        message = "--workers must be positive and cannot override CUDA preparation"
        raise ValueError(message)
    if batch["device"] == "cuda" and step == "prepare":
        return  # CUDA device/precision and single-worker checks belong to preparation.
    policy_path = root / f"{step}-resource-policy.json"
    if workers is None and policy_path.exists():
        message = "Resuming shared CPU work requires the recorded explicit --workers count"
        raise ValueError(message)
    observed = hardware()
    cpus = observed["affinity"]
    if workers is not None:
        if (
            workers > len(cpus)
            or "EPYC 9754" not in observed["cpu_model"]
            or observed["hostname"].split(".")[0] not in {"himem01", "himem02"}
        ):
            message = "Shared CPU work requires enough AMD EPYC 9754 cores on himem01 or himem02"
            raise ValueError(message)
        if any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
            message = "Shared CPU work requires all declared thread limits to equal one"
            raise ValueError(message)
        immutable_write(
            policy_path,
            {"workers": workers, "timing_profile": "diagnostic", "allocation": "shared-cpu"},
        )
        immutable_write(root / f"{step}-allocation.json", observed)
        return
    if len(cpus) != 128 or observed["hostname"].split(".")[0] != batch["node"]:
        message = "Batch requires its declared full 128-core standardized node"
        raise ValueError(message)
    try:
        os.sched_setaffinity(0, {cpus[0]})
        verify_profile(PROFILE)
    finally:
        os.sched_setaffinity(0, cpus)
    immutable_write(root / f"{step}-allocation.json", observed)


def verify_complete(root: Path, snapshot: dict) -> None:
    """A normal runner exit at its deadline may still contain pending cells."""
    allocation = json.loads((root / "sweep" / "allocation.json").read_text())
    count = len(allocation["cpus"])
    expected = {game["id"] for game in snapshot["games"]}
    if set(allocation["game_ids"]) != expected:
        message = "Sweep allocation differs from the prepared games"
        raise ValueError(message)
    for slot in range(min(count, len(expected))):
        path = root / "sweep" / f"shard-{slot:03}" / "results.json"
        if not path.is_file() or not json.loads(path.read_text()).get("campaign", {}).get(
            "complete"
        ):
            message = "Sweep has pending cells; resume this same batch before auditing publication"
            raise RuntimeError(message)


def verify_snapshot(snapshot: dict, suite: dict, source: dict) -> None:
    """Match the recipe and recompute only the two budget fields derived at freezing."""
    expected = dict(suite)
    if suite.get("relative_budgets"):
        expected["budgets_by_game"] = {
            game["id"]: sorted(
                {math.ceil(ratio * game["n_players"]) for ratio in suite["relative_budgets"]}
            )
            for game in snapshot["games"]
        }
        expected["budgets"] = sorted(
            {budget for grid in expected["budgets_by_game"].values() for budget in grid}
        )
    if snapshot["provenance"] != source or snapshot["suite"] != expected:
        message = (
            "Prepared snapshot differs from the qualified source, suite or relative budget grid"
        )
        raise ValueError(message)


def run_batch(manifest: Path, step: str, index: int, workers: int | None = None) -> None:
    """Reject changed suites, manifests, launch code and editable imports before execution."""
    campaign = json.loads((manifest.parent / "campaign.json").read_text())
    journal = json.loads((manifest.parent / "jobs.json").read_text())
    if identity(campaign) != journal["plan_sha256"]:
        message = "Campaign differs from its submission journal"
        raise ValueError(message)
    if provenance() != campaign["source"] or scripts() != campaign["scripts"]:
        message = "Source/environment or launch scripts differ from the submitted campaign"
        raise ValueError(message)
    group = json.loads(manifest.read_text())
    expected_group = [
        b
        for b in campaign["batches"]
        if f"phase-{b['phase']}-{b['device']}-{b['node']}" == manifest.stem
    ]
    if index < 0 or index >= len(group) or group != expected_group:
        message = "Array manifest differs from the authenticated campaign"
        raise ValueError(message)
    batch = group[index]
    root = Path(batch["directory"])
    original = json.loads((root / "suite.json").read_text())
    if identity(original) != batch["suite_sha256"]:
        message = "Batch suite changed after submission"
        raise ValueError(message)
    if batch["device"] == "cpu" or step == "evaluate":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    verify_allocation(batch, step, root, workers)
    workers = (
        workers
        if workers is not None
        else ((1 if batch["device"] == "cuda" else 16) if step == "prepare" else 128)
    )
    qualified_path = root / "qualified-suite.json"
    decision_path = root / "qualification-decision.json"
    suite = json.loads(qualified_path.read_text()) if qualified_path.exists() else None
    if suite is not None:
        decision = json.loads(decision_path.read_text())
        if decision != {
            "requested_suite_sha256": identity(original),
            "qualified_suite_sha256": identity(suite),
            "source": campaign["source"],
        }:
            message = "Frozen qualification decision or its inputs changed"
            raise ValueError(message)
    if step == "prepare":
        if suite is None:
            qualification = importlib.import_module("shapiq_benchmark.qualification")
            suite = qualification.qualify_suite(
                original,
                root / "qualification",
                model_cache=root / "stage" / ".models",
                workers=workers,
            )
            immutable_write(qualified_path, suite)
            immutable_write(
                decision_path,
                {
                    "requested_suite_sha256": identity(original),
                    "qualified_suite_sha256": identity(suite),
                    "source": campaign["source"],
                },
            )
        if not suite.get("families") and not suite.get("games"):
            immutable_write(
                root / "excluded.json",
                {
                    "suite_sha256": identity(suite),
                    "reason": "Every recipe failed the declared preparation gate",
                },
            )
            return
        if (root / "prepared" / "snapshot.json").exists():
            snapshot, _ = load_snapshot(root / "prepared")
            verify_snapshot(snapshot, suite, campaign["source"])
            return
        command = [
            sys.executable,
            "benchmark/prepare_matrix.py",
            str(qualified_path),
            str(root / "stage"),
            str(root / "prepared"),
            "--workers",
            str(workers),
            "--seconds",
            "250000",
        ]
    else:
        if suite is None:
            message = "Evaluation requires a completed qualification decision"
            raise ValueError(message)
        if (root / "excluded.json").exists():
            excluded = json.loads((root / "excluded.json").read_text())
            if (
                suite.get("families")
                or suite.get("games")
                or excluded["suite_sha256"] != identity(suite)
            ):
                message = "Excluded-batch marker disagrees with its qualified suite"
                raise ValueError(message)
            return
        snapshot, _ = load_snapshot(root / "prepared")
        verify_snapshot(snapshot, suite, campaign["source"])
        command = [
            sys.executable,
            "benchmark/sweep.py",
            str(root / "prepared"),
            str(root / "sweep"),
            "--workers",
            str(workers),
            "--timeout",
            "600",
            "--seconds",
            "250000",
        ]
    subprocess.run(command, check=True)  # noqa: S603 -- fixed scripts and argument lists
    if step == "evaluate":
        verify_complete(root, snapshot)


def main() -> None:
    """Run the one Slurm array task selected by its immutable manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("step", choices=("prepare", "evaluate"))
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        help="Workers on shared himem01/02 CPU allocation; timings are diagnostic (not CUDA prep)",
    )
    args = parser.parse_args()
    run_batch(args.manifest, args.step, args.index, args.workers)


if __name__ == "__main__":
    main()
