"""Campaign launch plans are immutable, resumable and separated by audit gates."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from shapiq_benchmark.prepare import prepare

DIRECTORY = Path(__file__).resolve().parents[2] / "benchmark"


def module(name: str) -> object:
    """Load a CLI without executing it or contacting the scheduler."""
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


queue = module("queue_phases")
batch = module("phase_batch")


@pytest.fixture
def campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Two tiny phases, including CPU and GPU preparation; no real Slurm calls."""
    monkeypatch.setattr(
        queue, "provenance", lambda: {"source_dirty": False, "git_commit": "frozen"}
    )
    monkeypatch.setattr(queue, "scripts", lambda: {"phase_batch.py": "hash"})

    def phase(number: int, base: dict) -> dict:
        return {
            "families": [{"id": f"p{number}", "device": "cuda" if number == 5 else "cpu"}],
            "games": [],
            "phase_plan": {"phase": number},
        }

    monkeypatch.setattr(queue, "build_phase", phase)
    planned = queue.plan(tmp_path, {})
    queue.write(tmp_path / "campaign.json", planned)
    return planned


def test_submit_once_and_audit_holds(
    campaign: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated submission reuses receipts; later preparation waits for explicit audit release."""
    commands = []

    def submit(command: list[str], **kwargs: object) -> str:
        if command[0] == "/usr/bin/git":
            return ""
        commands.append(command)
        return str(100 + len(commands))

    monkeypatch.setattr(queue.subprocess, "check_output", submit)
    queue.submit(tmp_path, campaign)
    queue.submit(tmp_path, campaign)
    assert len(commands) == 10
    for command in commands:
        phase = int(next(x for x in command if x.startswith("--job-name=")).split("phase-")[1][0])
        if command[-1] == "prepare":
            assert ("--hold" in command) == (phase > 3)
        else:
            assert any(arg.startswith("--dependency=aftercorr:") for arg in command)
    gpu = next(c for c in commands if c[-1] == "prepare" and "--gres=gpu:l40s:1" in c)
    assert "--cpus-per-task=1" in gpu
    journal = json.loads((tmp_path / "jobs.json").read_text())
    assert queue.watch_jobs(journal, 3) == ["101_0", "102_0"]


def test_ambiguous_submission_never_automatically_duplicates(
    campaign: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persist intent before sbatch so a lost response requires scheduler reconciliation."""

    def submit(command: list[str], **kwargs: object) -> str:
        if command[0] == "/usr/bin/git":
            return ""
        raise queue.subprocess.TimeoutExpired(command, 60)

    monkeypatch.setattr(queue.subprocess, "check_output", submit)
    with pytest.raises(queue.subprocess.TimeoutExpired):
        queue.submit(tmp_path, campaign)
    with pytest.raises(RuntimeError, match="Unacknowledged submission"):
        queue.submit(tmp_path, campaign)


def test_mutated_suite_is_rejected_before_submission(
    campaign: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest path alone is not an authenticated experiment definition."""
    path = Path(campaign["batches"][0]["directory"]) / "suite.json"
    suite = json.loads(path.read_text())
    suite["families"][0]["id"] = "changed"
    queue.write(path, suite)
    monkeypatch.setattr(
        queue.subprocess, "check_output", lambda *a, **k: pytest.fail("unexpected submission")
    )
    with pytest.raises(ValueError, match="planned suite changed"):
        queue.submit(tmp_path, campaign)


def test_campaign_and_inventory_are_immutable(campaign: dict, tmp_path: Path) -> None:
    """A resumed plan cannot overwrite the audit record with new inputs."""
    with pytest.raises(ValueError, match="Persisted campaign input changed"):
        queue.immutable_write(tmp_path / "campaign.json", {**campaign, "source": {}})
    with pytest.raises(ValueError, match="Persisted campaign input changed"):
        queue.immutable_write(tmp_path / "phase-3-inventory.json", {})


def test_changed_cross_phase_recipe_is_not_silently_deduplicated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same recipe ID must mean the same game in every cumulative phase."""
    monkeypatch.setattr(
        queue,
        "build_phase",
        lambda number, base: {
            "families": [{"id": "same", "n_players": number}],
            "games": [],
            "phase_plan": {},
        },
    )
    with pytest.raises(ValueError, match="changed meaning across phases"):
        queue.plan(tmp_path, {})


def test_evaluation_deadline_is_not_completion(tmp_path: Path) -> None:
    """Normal process exit with pending cells must not advance a phase."""
    sweep = tmp_path / "sweep"
    (sweep / "shard-000").mkdir(parents=True)
    queue.write(sweep / "allocation.json", {"cpus": [0], "game_ids": ["game"]})
    result = sweep / "shard-000/results.json"
    queue.write(result, {"campaign": {"complete": False}})
    with pytest.raises(RuntimeError, match="pending cells"):
        batch.verify_complete(tmp_path, {"games": [{"id": "game"}]})
    queue.write(result, {"campaign": {"complete": True}})
    batch.verify_complete(tmp_path, {"games": [{"id": "game"}]})


def test_worker_checks_manifest_and_source(
    campaign: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed array group cannot redirect a worker outside its frozen campaign."""
    queue.write(tmp_path / "jobs.json", {"plan_sha256": queue.identity(campaign)})
    manifest = tmp_path / "group.json"
    queue.write(manifest, [{**campaign["batches"][0], "directory": str(tmp_path / "wrong")}])
    monkeypatch.setattr(batch, "provenance", lambda: campaign["source"])
    monkeypatch.setattr(batch, "scripts", lambda: campaign["scripts"])
    with pytest.raises(ValueError, match="Array manifest differs"):
        batch.run_batch(manifest, "prepare", 0)
    monkeypatch.setattr(batch, "provenance", dict)
    with pytest.raises(ValueError, match="Source/environment"):
        batch.run_batch(manifest, "prepare", 0)


def test_allocation_restores_affinity_and_rejects_wrong_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full-node claims require observed hardware and the existing exclusive-profile verifier."""
    observed = {"hostname": "himem01", "affinity": list(range(128)), "cpu_model": "AMD EPYC 9754"}
    monkeypatch.setattr(batch, "hardware", lambda: observed)
    affinity = []
    monkeypatch.setattr(
        batch.os, "sched_setaffinity", lambda pid, cpus: affinity.append(list(cpus))
    )
    verified = []
    monkeypatch.setattr(batch, "verify_profile", verified.append)
    spec = {"device": "cpu", "node": "himem01"}
    batch.verify_allocation(spec, "prepare", tmp_path)
    assert verified == [batch.PROFILE]
    assert affinity == [[0], list(range(128))]
    with pytest.raises(ValueError, match="declared full 128-core"):
        batch.verify_allocation({**spec, "node": "himem02"}, "prepare", tmp_path)


def test_excluded_batch_freezes_decision_and_never_evaluates(
    campaign: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete exclusion inventory is preserved without fabricating a measurement snapshot."""
    spec = campaign["batches"][0]
    directory = Path(spec["directory"])
    manifest = tmp_path / f"phase-{spec['phase']}-{spec['device']}-{spec['node']}.json"
    queue.write(manifest, [spec])
    queue.write(tmp_path / "jobs.json", {"plan_sha256": queue.identity(campaign)})
    monkeypatch.setattr(batch, "provenance", lambda: campaign["source"])
    monkeypatch.setattr(batch, "scripts", lambda: campaign["scripts"])
    monkeypatch.setattr(batch, "verify_allocation", lambda *a: None)
    calls = []

    def qualify(suite: dict, *args: object, **kwargs: object) -> dict:
        calls.append(suite)
        return {**suite, "families": [], "preparation_exclusions": [{"reason": "cost"}]}

    monkeypatch.setattr(
        batch.importlib,
        "import_module",
        lambda _: type("Qualification", (), {"qualify_suite": staticmethod(qualify)}),
    )
    monkeypatch.setattr(batch.subprocess, "run", lambda *a, **k: pytest.fail("unexpected compute"))
    batch.run_batch(manifest, "prepare", 0)
    batch.run_batch(manifest, "prepare", 0)
    batch.run_batch(manifest, "evaluate", 0)
    assert len(calls) == 1
    assert (directory / "excluded.json").is_file()
    assert not (directory / "prepared").exists()
    qualified = json.loads((directory / "qualified-suite.json").read_text())
    qualified["preparation_exclusions"] = []
    queue.write(directory / "qualified-suite.json", qualified)
    with pytest.raises(ValueError, match="Frozen qualification decision"):
        batch.run_batch(manifest, "evaluate", 0)


def test_real_snapshot_budget_derivation_resumes_and_evaluates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise a real freeze: prepared suites gain authenticated integer query grids."""
    root = tmp_path / "batch"
    root.mkdir()
    suite = {
        "name": "tiny",
        "families": [{"id": "tiny", "family": "dummy", "n_players": 3}],
        "games": [],
        "targets": [{"index": "SV", "order": 1}],
        "methods": ["KernelSHAP"],
        "relative_budgets": [0.5, 1, 2],
        "seeds": [0],
    }
    queue.write(root / "suite.json", suite)
    snapshot = prepare(root / "suite.json", root / "prepared")
    assert snapshot["suite"]["budgets"] == [2, 3, 6]
    assert "budgets" not in suite
    record = {
        "id": "tiny",
        "phase": 3,
        "device": "cpu",
        "node": "himem01",
        "directory": str(root),
        "suite_sha256": queue.identity(suite),
    }
    campaign = {"source": snapshot["provenance"], "scripts": {}, "batches": [record]}
    queue.write(tmp_path / "campaign.json", campaign)
    queue.write(tmp_path / "jobs.json", {"plan_sha256": queue.identity(campaign)})
    manifest = tmp_path / "phase-3-cpu-himem01.json"
    queue.write(manifest, [record])
    queue.write(root / "qualified-suite.json", suite)
    queue.write(
        root / "qualification-decision.json",
        {
            "requested_suite_sha256": queue.identity(suite),
            "qualified_suite_sha256": queue.identity(suite),
            "source": snapshot["provenance"],
        },
    )
    monkeypatch.setattr(batch, "provenance", lambda: snapshot["provenance"])
    monkeypatch.setattr(batch, "scripts", dict)
    monkeypatch.setattr(batch, "verify_allocation", lambda *a: None)
    calls = []

    def evaluate(command: list[str], **kwargs: object) -> None:
        calls.append(command)
        (root / "sweep/shard-000").mkdir(parents=True)
        queue.write(
            root / "sweep/allocation.json",
            {"cpus": [0], "game_ids": [g["id"] for g in snapshot["games"]]},
        )
        queue.write(root / "sweep/shard-000/results.json", {"campaign": {"complete": True}})

    monkeypatch.setattr(batch.subprocess, "run", evaluate)
    batch.run_batch(manifest, "prepare", 0)
    assert not calls  # Existing real snapshot is reused after checking derived budgets.
    batch.run_batch(manifest, "evaluate", 0)
    assert len(calls) == 1 and calls[0][1] == "benchmark/sweep.py"
    changed = {**snapshot, "suite": {**snapshot["suite"], "budgets": [2, 3, 6, 99]}}
    with pytest.raises(ValueError, match="relative budget grid"):
        batch.verify_snapshot(changed, suite, snapshot["provenance"])
