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

from shapiq_benchmark.execution import PROFILE, hardware, verify_profile
from shapiq_benchmark.runner import identity, load_snapshot, provenance


def verify_allocation(batch: dict, step: str, root: Path) -> None:
    """Require the actual standardized exclusive node, and stable CPU IDs on resume."""
    if batch["device"] == "cuda" and step == "prepare":
        return  # CUDA device/precision and single-worker checks belong to preparation.
    observed = hardware()
    cpus = observed["affinity"]
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


def run_batch(manifest: Path, step: str, index: int) -> None:
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
    verify_allocation(batch, step, root)
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
                workers=1 if batch["device"] == "cuda" else 16,
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
            "1" if batch["device"] == "cuda" else "16",
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
            "128",
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
    args = parser.parse_args()
    run_batch(args.manifest, args.step, args.index)


if __name__ == "__main__":
    main()
