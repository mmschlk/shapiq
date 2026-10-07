"""Small authentic compact checkpoints exercise terminal pool assembly."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from shapiq import InteractionValues
from shapiq_benchmark import runner
from shapiq_benchmark.execution import THREAD_VARIABLES
from shapiq_benchmark.record_store import RecordStore
from shapiq_benchmark.results_io import Checkpoint

spec = importlib.util.spec_from_file_location(
    "collect_pool_tested", Path(__file__).resolve().parents[2] / "benchmark/collect_pool.py"
)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")
    return {"path": str(path), "sha256": collector.digest(path)}


def lines(path, values):
    path.write_text("".join(json.dumps(x) + "\n" for x in values))


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    source = {"source_sha256": "frozen", "python": "fixture"}
    monkeypatch.setattr(runner, "provenance", lambda: copy.deepcopy(source))
    accounting = {
        "1": {
            "state": "COMPLETED",
            "exit_code": "0:0",
            "cpus": 1,
            "cpu_seconds": 10,
            "node": "node",
        }
    }
    monkeypatch.setattr(collector, "terminal_accounting", lambda jobs: copy.deepcopy(accounting))
    family = {"id": "recipe", "family": "local_baseline", "dataset": "fixture", "n_players": 4}
    recipe = {"id": "recipe", "application": "local", "subtype": "", "n_players": "4"}
    suite = {
        "name": "comprehensive-fixture",
        "families": [family],
        "games": [],
        "methods": ["PermutationSamplingSV", "PermutationSamplingSTII"],
        "method_parameters": {},
        "targets": [{"index": "SV", "order": 1}],
        "relative_budgets": [1],
        "game_seeds": [0],
        "seeds": [0],
        "min_players": 4,
        "min_signal_ratio": 1e-6,
        "cell_timeout_policy": {
            "ordinary_seconds": 30,
            "extended_seconds": 120,
            "min_players": 128,
            "min_relative_budget": 32,
        },
        "focused_design": {"recipes": [recipe]},
        "comprehensive_design": {"version": 1},
    }
    task = {
        "case": 0,
        "kind": "family",
        "recipe": "recipe",
        "seed": 0,
        "spec": family,
        "specs": [family],
    }
    config = {
        "pool_directory": str(tmp_path / "pool"),
        "suite": save(tmp_path / "suite.json", suite),
        "source": save(tmp_path / "source.json", source),
        "inventory": save(tmp_path / "inventory.json", [task]),
    }
    script = tmp_path / "launcher"
    script.write_text("frozen launcher")
    config["jobs"] = {
        "1": {
            "registration": save(tmp_path / "registered.json", {"job": "1", "cpus": 1}),
            "intent": save(
                tmp_path / "intent.json",
                {
                    "command": ["sbatch", "--cpus-per-task=1", "--nodelist=node", str(script)],
                    "script_sha256": collector.digest(script),
                },
            ),
        }
    }
    allocation = {
        "job": "1",
        "hostname": "node.cluster",
        "cpus": [2],
        "workers": 1,
        "suite_sha256": config["suite"]["sha256"],
        "source_sha256": config["source"]["sha256"],
    }
    save(tmp_path / "pool/allocations/1/started.json", allocation)
    directory = tmp_path / "pool/tasks/case-000000"
    claim = {k: allocation[k] for k in ("job", "hostname", "suite_sha256", "source_sha256")} | {
        "cpu": 2,
        "task": task,
    }
    save(directory / "claim.json", claim)
    fragment = collector.fragment(suite, task)
    save(directory / "suite.json", fragment)
    game = {
        "id": "recipe-i0-sv-1",
        "family": "local_explanation",
        "stratum": "fixture",
        "n_players": 4,
        "index": "SV",
        "order": 1,
        "oracle": "table",
        "artifact": "payoffs.npz",
        "truth": {
            "coordinates": [[i] for i in range(4)],
            "values": [1.0] * 4,
            "baseline": 0.0,
            "energy": 4.0,
        },
        "metadata": {
            "focused_design": {"application": "local", "subtype": "", "recipe": "recipe"},
            "instance_seed": 0,
            "payoff_std": 1.0,
        },
    }
    snapshot_suite = copy.deepcopy(fragment) | {
        "budgets": [4],
        "budgets_by_game": {game["id"]: [4]},
    }
    artifact = directory / "prepared/payoffs.npz"
    artifact.parent.mkdir()
    artifact.write_bytes(b"authenticated fixture; collector does not calculate payoffs")
    snapshot = {
        "schema_version": 1,
        "suite": snapshot_suite,
        "games": [game],
        "coverage": [],
        "provenance": source,
        "artifacts": {"payoffs.npz": collector.digest(artifact)},
    }
    snapshot["snapshot_id"] = runner.identity(snapshot)
    snapshot_pin = save(directory / "prepared/snapshot.json", snapshot)
    save(
        directory / "prepared.json",
        {
            "snapshot_id": snapshot["snapshot_id"],
            "snapshot_sha256": snapshot_pin["sha256"],
            "targets": 1,
            "source": source,
        },
    )
    worker = {
        "slurm_job_id": "1",
        "hostname": "node.cluster",
        "affinity": [2],
        "cpu_model": "fixture CPU",
        "machine": "x86_64",
        "thread_environment": dict.fromkeys(THREAD_VARIABLES, "1"),
        "thread_pools": [],
    }
    response = {
        "status": "ok",
        "worker": worker,
        "queries": 4,
        "requested_queries": 4,
        "seconds": 0.1,
        "wall_seconds": 0.2,
        "estimate": {"coordinates": [[i] for i in range(4)], "values": [1.0] * 4},
        **runner.score(
            InteractionValues(
                {(i,): 1.0 for i in range(4)}, index="SV", max_order=1, n_players=4, min_order=0
            ),
            game,
        ),
    }
    intent = {
        "game_id": game["id"],
        "method": "PermutationSamplingSV",
        "budget": 4,
        "seed": 0,
        "sequence": 1,
        "slurm_job_id": "1",
        "cpu": 2,
        "expected_snapshot_id": snapshot["snapshot_id"],
        "expected_source_hash": source["source_sha256"],
        "method_parameters": None,
        "timeout": 30,
    }
    envelope = {
        "sequence": 1,
        "cell": [game["id"], "PermutationSamplingSV", 4, 0],
        "response": response,
    }
    lines(directory / "intents.jsonl", [intent])
    lines(directory / "responses.jsonl", [envelope])
    common = {
        "game_id": game["id"],
        "budget": 4,
        "seed": 0,
        "nmse": None,
        "mse": None,
        "error": None,
        "timing_scope": "estimator_with_table_oracle",
        "timing_profile": "diagnostic",
        "official_timing": False,
    }
    rows = [
        {**common, "method": "PermutationSamplingSV", **response},
        {
            **common,
            "method": "PermutationSamplingSTII",
            "status": "unsupported",
            "queries": 0,
            "requested_queries": 0,
            "seconds": None,
            "wall_seconds": 0,
        },
    ]
    raw = {
        "schema_version": 1,
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_provenance": source,
        "suite": snapshot_suite,
        "games": [game],
        "coverage": [],
        "methods": {
            name: {
                "source_sha256": source["source_sha256"],
                "software_sha256": runner.identity(source),
                "private": False,
            }
            for name in suite["methods"]
        },
        "run_provenance": {
            **source,
            "execution": {
                "hardware": {"hostname": "node.cluster", "affinity": [2]},
                "threads": 1,
                "timeout": 30,
                "memory_gb": 8,
                "timing_profile": "diagnostic",
                "cell_timeout_policy": suite["cell_timeout_policy"],
                "game_ids": [game["id"]],
            },
        },
        "records": rows,
        "campaign": {"planned": 2, "completed": 2, "complete": True},
    }
    (directory / "results").mkdir()
    Checkpoint(directory / "results", directory / "prepared", raw)
    save(directory / "outcome.json", {"status": "complete"})
    return config, directory, raw, envelope, intent


def run_collection(campaign, tmp_path):
    with RecordStore(tmp_path / "scratch.sqlite") as store:
        data, audit = collector.collect(campaign[0], store)
        return {**data, "records": list(store)}, audit


def test_complete_run_preserves_identity_and_design(campaign, tmp_path):
    data, audit = run_collection(campaign, tmp_path)
    assert audit["complete"] and audit["score_complete"] and audit["outcome_complete"]
    assert not audit["publication_ready"]
    assert set(data["runs"]) == {runner.identity(campaign[2])}
    assert data["suite"]["comprehensive_design"] == {"version": 1}
    assert data["suite"]["focused_design"]["recipes"][0]["id"] == "recipe"
    assert [r["nmse"] for r in data["records"]] == [0.0, None]


def test_unanswered_intent_is_explicit_unscored_outcome(campaign, tmp_path):
    _, directory, raw, _, _ = campaign
    (directory / "responses.jsonl").unlink()
    (directory / "results/results.json").unlink()
    (directory / "results/records.jsonl").unlink()
    (directory / "outcome.json").unlink()
    data, audit = run_collection(campaign, tmp_path)
    assert audit["complete"] and not audit["score_complete"]
    failed = next(r for r in data["records"] if r["status"] == "failed")
    assert failed["nmse"] is failed["seconds"] is failed["queries"] is None
    assert failed["error_type"] == "OperationalInterruption"
    assert runner.identity(raw) not in data["runs"]


def test_never_attempted_is_not_complete(campaign, tmp_path):
    _, directory, _, _, _ = campaign
    for path in (
        "intents.jsonl",
        "responses.jsonl",
        "results/results.json",
        "results/records.jsonl",
        "outcome.json",
    ):
        (directory / path).unlink()
    data, audit = run_collection(campaign, tmp_path)
    assert not audit["complete"] and audit["cases"][0]["never_attempted_supported"] == 1
    assert [r["status"] for r in data["records"]] == ["unsupported"]


def test_partial_checkpoint_preserves_success_as_derived(campaign, tmp_path):
    _, directory, raw, _, _ = campaign
    partial = copy.deepcopy(raw)
    partial["records"] = []
    partial["campaign"] = {"planned": 2, "completed": 0, "complete": False}
    checkpoint = Checkpoint(directory / "results", directory / "prepared", partial)
    checkpoint.append(raw["records"][0])  # genuine durable row after old checkpoint
    with (directory / "results/records.jsonl").open("ab") as stream:
        stream.write(b'{"partial":')
    (directory / "outcome.json").unlink()
    data, audit = run_collection(campaign, tmp_path)
    assert audit["complete"] and audit["score_complete"]
    assert not audit["cases"][0]["original_checkpoint_closed"]
    assert next(r for r in data["records"] if r["status"] == "ok")["seconds"] == 0.1


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_intent",
        "worker",
        "score",
        "wall_seconds",
        "false_completion",
        "committed_prefix",
        "orphan_journal",
    ],
)
def test_bad_evidence_fails_closed(campaign, tmp_path, mutation):
    _, directory, raw, envelope, intent = campaign
    if mutation == "duplicate_intent":
        lines(directory / "intents.jsonl", [intent, intent])
    elif mutation in {"worker", "score", "wall_seconds"}:
        if mutation == "worker":
            envelope["response"]["worker"]["slurm_job_id"] = "2"
        elif mutation == "score":
            envelope["response"]["nmse"] = 9
        else:
            envelope["response"]["wall_seconds"] = -1
        raw["records"][0].update(envelope["response"])
        lines(directory / "responses.jsonl", [envelope])
        Checkpoint(directory / "results", directory / "prepared", raw)
    elif mutation == "false_completion":
        raw["records"] = []
        raw["campaign"] = {"planned": 2, "completed": 0, "complete": False}
        Checkpoint(directory / "results", directory / "prepared", raw)
    elif mutation == "orphan_journal":
        (directory / "results/results.json").unlink()
        (directory / "responses.jsonl").unlink()
    else:
        path = directory / "results/records.jsonl"
        path.write_bytes(path.read_bytes().replace(b"0.1", b"0.9", 1))
    with pytest.raises(ValueError):
        run_collection(campaign, tmp_path)


def test_terminal_check_happens_before_inputs(campaign, tmp_path, monkeypatch):
    def live(jobs):
        message = "Pool allocation is still live"
        raise ValueError(message)

    monkeypatch.setattr(collector, "terminal_accounting", live)
    campaign[0]["suite"]["path"] = "/does/not/exist"
    with pytest.raises(ValueError, match="still live"):
        run_collection(campaign, tmp_path)


def test_live_scheduler_ids_are_text(monkeypatch):
    def output(command, *, text=False):
        assert command[0] == "squeue"  # live ownership must stop before accounting
        return "1\n" if text else b"1\n"

    monkeypatch.setattr(collector.subprocess, "check_output", output)
    with pytest.raises(ValueError, match="still live"):
        collector.terminal_accounting({"1": {}})


def test_authenticated_preparation_failure_counts_as_outcome(campaign, tmp_path):
    _, directory, _, _, _ = campaign
    (directory / "prepared.json").unlink()
    for name in (
        "intents.jsonl",
        "responses.jsonl",
        "results/results.json",
        "results/records.jsonl",
    ):
        (directory / name).unlink()
    execution = {"returncode": 1, "timed_out": False, "wall_seconds": 1}
    save(directory / "preparation-execution.json", execution)
    save(directory / "outcome.json", {"status": "preparation_failed", "preparation": execution})
    data, audit = run_collection(campaign, tmp_path)
    assert audit["complete"] and not audit["score_complete"]
    assert not data["records"] and data["coverage"][0]["status"] == "not_measured"
