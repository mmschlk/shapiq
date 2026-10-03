"""Plan small disjoint batches, then queue preparation/evaluation behind audit gates.

Planning is read-only apart from writing manifests. Submit from a clean frozen
checkout with --submit. Phase three starts; later preparation arrays are held
until their preceding phase's independent audit. Slurm aftercorr dependencies
pair each evaluation task with its own preparation task. Job IDs are journaled
after every submission; reruns reuse that journal instead of duplicating work.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import subprocess
from pathlib import Path

from shapiq_benchmark.planning import select_core
from shapiq_benchmark.protocol import build_phase
from shapiq_benchmark.runner import digest, identity, method_catalog, provenance

NODES = ("himem01", "himem02", "gpu15")


def write(path: Path, value: dict | list) -> None:
    """Persist the submission journal after each successful scheduler response."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def scripts() -> dict:
    """Authenticate launch code outside the package provenance fingerprint."""
    directory = Path(__file__).resolve().parent
    return {
        name: digest(directory / name)
        for name in (
            "queue_phases.py",
            "phase_batch.py",
            "phase_batch.sbatch",
            "prepare_matrix.py",
            "sweep.py",
        )
    }


def immutable_write(path: Path, value: dict | list) -> None:
    """Resume identical plans only; never rewrite a submitted experiment's meaning."""
    if path.exists() and json.loads(path.read_text()) != value:
        message = f"Persisted campaign input changed: {path.name}"
        raise ValueError(message)
    write(path, value)


def plan(
    root: Path,
    base: dict,
    *,
    batch_size: int = 16,
    bounded_core: bool = False,
    nodes: tuple[str, ...] = NODES,
) -> dict:
    """Do not rerun the same declared recipe in each cumulative phase."""
    if batch_size < 1:
        message = "Batch size must be positive"
        raise ValueError(message)
    if not nodes or len(set(nodes)) != len(nodes) or not set(nodes) <= set(NODES):
        message = "Select unique nodes from the verified CPU pool."
        raise ValueError(message)
    root.mkdir(parents=True, exist_ok=True)
    seen, batches = {}, []
    for phase in range(3, 8):
        suite = build_phase(phase, base)
        if bounded_core:
            suite["duplicate_registry"] = str((root / "duplicate-games.json").resolve())
            suite["method_parameters"] = {"OddSHAP": {"ridge": 0.001}}
            for kind in ("families", "games"):
                for spec in suite[kind]:
                    spec["quality_protocol"] = "quality-v2"
                    spec["id"] += "-quality-v2"
        immutable_write(root / f"phase-{phase}-inventory.json", suite)
        entries = []
        for kind in ("families", "games"):
            for spec in suite[kind]:
                key = (kind, spec["id"])
                if key not in seen:
                    entries.append((kind, spec))
                    seen[key] = spec
                elif seen[key] != spec:
                    message = f"Recipe ID changed meaning across phases: {spec['id']}"
                    raise ValueError(message)
        if bounded_core:
            entries, selection = select_core(entries, suite, method_catalog())
            immutable_write(root / f"phase-{phase}-core.json", selection)
        for device in ("cpu", "cuda"):
            group = [(kind, spec) for kind, spec in entries if spec.get("device", "cpu") == device]
            for offset in range(0, len(group), batch_size):
                selected = group[offset : offset + batch_size]
                name = f"phase-{phase}-{device}-{offset // batch_size:04d}"
                directory = root / name
                directory.mkdir(exist_ok=True)
                batch = copy.deepcopy(suite)
                batch["name"] = name
                for kind in ("families", "games"):
                    batch[kind] = [spec for entry_kind, spec in selected if entry_kind == kind]
                selected_ids = {spec["id"] for _, spec in selected}
                batch["phase_plan"] = {
                    "inventory_sha256": identity(suite["phase_plan"]),
                    "selected_recipe_ids": sorted(selected_ids),
                    "note": "Disjoint batch of the full phase inventory; selected does not mean qualified.",
                }
                path = directory / "suite.json"
                if path.exists() and json.loads(path.read_text()) != batch:
                    message = f"Existing batch changed: {name}"
                    raise ValueError(message)
                write(path, batch)
                batches.append(
                    {
                        "id": name,
                        "phase": phase,
                        "device": device,
                        "node": nodes[(offset // batch_size) % len(nodes)],
                        "directory": str(directory.resolve()),
                        "recipes": len(selected),
                        "suite_sha256": identity(batch),
                    }
                )
    return {"source": provenance(), "scripts": scripts(), "batches": batches}


def submit(root: Path, campaign: dict) -> None:
    """Submit bounded arrays and record every ID; future phases wait for audit release."""
    if campaign["source"]["source_dirty"] or not campaign["source"]["git_commit"]:
        message = "Submission requires a clean immutable checkout"
        raise ValueError(message)
    if provenance() != campaign["source"] or scripts() != campaign["scripts"]:
        message = "Campaign source or launch scripts changed before submission"
        raise ValueError(message)
    for batch in campaign["batches"]:
        if (
            identity(json.loads((Path(batch["directory"]) / "suite.json").read_text()))
            != batch["suite_sha256"]
        ):
            message = "A planned suite changed before submission"
            raise ValueError(message)
    dirty_launch = subprocess.check_output(  # noqa: S603 -- fixed Git read, authenticated paths
        [
            "/usr/bin/git",
            "status",
            "--porcelain",
            "--",
            *["benchmark/" + name for name in campaign["scripts"]],
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        timeout=30,
    ).strip()
    if dirty_launch:
        message = "Submission requires committed launch scripts"
        raise ValueError(message)
    journal_path = root / "jobs.json"
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else {}
    if journal and journal.get("plan_sha256") != identity(campaign):
        message = "Submitted campaign plan changed"
        raise ValueError(message)
    journal.setdefault("plan_sha256", identity(campaign))
    jobs = journal.setdefault("groups", {})
    groups = sorted({(b["phase"], b["device"], b["node"]) for b in campaign["batches"]})
    for phase, device, node in groups:
        name = f"phase-{phase}-{device}-{node}"
        group = [
            b
            for b in campaign["batches"]
            if (b["phase"], b["device"], b["node"]) == (phase, device, node)
        ]
        if len(group) > 1000:
            message = "Split the campaign before exceeding Hopper's array limit"
            raise ValueError(message)
        manifest = root / f"{name}.json"
        immutable_write(manifest, group)
        record = jobs.setdefault(name, {"phase": phase, "batches": len(group)})
        for step in ("prepare", "evaluate"):
            if step in record:
                continue
            if record.get(step + "_submission"):
                message = (
                    f"Unacknowledged submission for {name}/{step}; inspect Slurm before retrying"
                )
                raise RuntimeError(message)
            gpu = device == "cuda" and step == "prepare"
            command = [
                "sbatch",
                "--parsable",
                "--partition=main",
                "--nodes=1",
                "--ntasks=1",
                "--time=3-00:00:00",
                "--hint=nomultithread",
                f"--job-name={identity(campaign)[:8]}-{name}-{step}",
                f"--array=0-{len(group) - 1}%1",
                f"--output={root}/{name}-{step}-%A_%a.log",
                "--cpus-per-task=1" if gpu else "--cpus-per-task=128",
                "--mem=32G" if gpu else "--mem=384G",
            ]
            command += ["--gres=gpu:l40s:1"] if gpu else [f"--nodelist={node}", "--exclusive"]
            if step == "prepare" and phase > 3:
                command.append("--hold")
            if step == "evaluate":
                command.append("--dependency=aftercorr:" + record["prepare"])
            command += ["benchmark/phase_batch.sbatch", str(manifest.resolve()), step]
            # Write ahead: an interrupted/ambiguous sbatch response must never
            # cause an automatic duplicate submission on the next invocation.
            record[step + "_submission"] = {"command": command}
            write(journal_path, journal)
            job = subprocess.check_output(command, text=True, timeout=60).strip().split(";")[0]  # noqa: S603 -- fixed Slurm command
            if not job.isdigit():
                message = "Unexpected scheduler response; inspect Slurm before retrying"
                raise RuntimeError(message)
            record[step] = job
            record.pop(step + "_submission")
            write(journal_path, journal)
            print(f"{name} {step}: {job}", flush=True)  # noqa: T201 -- submission receipt


def watch_jobs(journal: dict, phase: int) -> list[str]:
    """Expand only the active phase into exact task IDs for the completion watcher."""
    return [
        f"{record[step]}_{index}"
        for record in journal.get("groups", {}).values()
        if record["phase"] == phase
        for step in ("prepare", "evaluate")
        if step in record
        for index in range(record["batches"])
    ]


def main() -> None:
    """Write reviewable manifests, or submit the reviewed plan from frozen source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base", type=Path, default=Path("benchmark/suites/all-families.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--nodes", nargs="+", choices=NODES, default=list(NODES))
    parser.add_argument(
        "--bounded-core",
        action="store_true",
        help="Qualify a resource-capped panel before matrix expansion",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = plan(
            args.output,
            json.loads(args.base.read_text()),
            batch_size=args.batch_size,
            bounded_core=args.bounded_core,
            nodes=tuple(args.nodes),
        )
        immutable_write(args.output / "campaign.json", campaign)
        if args.submit:
            submit(args.output, campaign)
        else:
            print(json.dumps({"batches": len(campaign["batches"])}))  # noqa: T201 -- CLI summary


if __name__ == "__main__":
    main()
