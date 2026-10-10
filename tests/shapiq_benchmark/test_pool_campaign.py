"""Bounded orchestration checks; no dataset or scientific execution."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "benchmark" / "pool_campaign.py"
spec = importlib.util.spec_from_file_location("pool_campaign_tested", PATH)
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)


def test_claim_is_once_even_without_final_outcome(tmp_path):
    task = {"case": 4, "kind": "family", "seed": 0, "spec": {"id": "example"}}
    directory = pool.claim(tmp_path, task, {"job": "1"})
    assert json.loads((directory / "claim.json").read_text())["task"] == task
    assert pool.claim(tmp_path, task, {"job": "2"}) is None
    assert not (directory / "outcome.json").exists()


def test_incomplete_claim_is_never_reassigned(tmp_path):
    (tmp_path / "case-000004").mkdir()
    assert pool.claim(tmp_path, {"case": 4}, {"job": "2"}) is None


def test_evidence_is_create_only_and_progress_replaceable(tmp_path):
    path = tmp_path / "receipt.json"
    pool.write(path, {"phase": 1})
    with pytest.raises(FileExistsError):
        pool.write(path, {"phase": 2})
    pool.write(path, {"phase": 3}, replace=True)
    assert json.loads(path.read_text()) == {"phase": 3}


def test_preparation_timeout_is_explicit(tmp_path):
    result = pool.bounded(
        [sys.executable, "-c", "import time; time.sleep(20)"], tmp_path / "log", 0.03
    )
    assert result["timed_out"]
    assert result["returncode"] != 0
    assert result["wall_seconds"] < 5


def test_source_pins_and_shared_registry_rejected(tmp_path):
    suite, source = tmp_path / "suite.json", tmp_path / "source.json"
    suite.write_text(json.dumps({"duplicate_registry": "/shared/unsafe"}))
    source.write_text("{}")
    args = argparse.Namespace(
        suite=suite,
        source=source,
        suite_sha256=pool.digest(suite),
        source_sha256=pool.digest(source),
    )
    with pytest.raises(ValueError, match="cross-host"):
        pool.authenticate(args)
    args.suite_sha256 = "wrong"
    with pytest.raises(ValueError, match="changed"):
        pool.authenticate(args)


def test_intent_precedes_execution_and_response_is_preserved(tmp_path, monkeypatch):
    request = {
        "game_id": "g",
        "method": "m",
        "budget": 8,
        "seed": 0,
        "expected_snapshot_id": "s",
        "expected_source_hash": "h",
        "method_parameters": None,
    }
    observed = []
    response = {"status": "failed", "seconds": None, "queries": None, "error": "failure"}

    def isolated(req, timeout, memory):
        intent = json.loads((tmp_path / "intents.jsonl").read_text())
        assert intent["game_id"] == req["game_id"]
        observed.append(intent)
        return response

    execution = types.SimpleNamespace(isolated=isolated)

    def run(*args, **kwargs):
        assert execution.isolated(request, 30, 8) == response
        with pytest.raises(ValueError, match="Repeated"):
            execution.isolated(request, 30, 8)
        return {"campaign": {"complete": True, "planned": 1, "completed": 1}}

    runner = types.SimpleNamespace(provenance=lambda: {"source": "pinned"}, run=run)
    package = types.ModuleType("shapiq_benchmark")
    package.execution, package.runner = execution, runner
    monkeypatch.setitem(sys.modules, "shapiq_benchmark", package)
    monkeypatch.setattr(pool, "bind", lambda *args: None)
    monkeypatch.setattr(pool, "authenticate", lambda args: ({}, {"source": "pinned"}))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    args = argparse.Namespace(
        cpu=1, memory_gb=8, phase="evaluate", task_directory=tmp_path, deadline=time.time() + 1000
    )
    pool.scientific(args)
    assert execution.isolated is isolated
    assert len(observed) == 1
    assert json.loads((tmp_path / "responses.jsonl").read_text())["response"] == response
    with pytest.raises(ValueError, match="already exists"):
        pool.scientific(args)


def test_unanswered_intent_survives_operational_interruption(tmp_path, monkeypatch):
    request = {
        "game_id": "g",
        "method": "m",
        "budget": 8,
        "seed": 0,
        "expected_snapshot_id": "s",
        "expected_source_hash": "h",
        "method_parameters": None,
    }

    def interrupted(*args):
        message = "operational interruption"
        raise RuntimeError(message)

    execution = types.SimpleNamespace(isolated=interrupted)

    def run(*args, **kwargs):
        execution.isolated(request, 30, 8)

    package = types.ModuleType("shapiq_benchmark")
    package.execution = execution
    package.runner = types.SimpleNamespace(provenance=dict, run=run)
    monkeypatch.setitem(sys.modules, "shapiq_benchmark", package)
    monkeypatch.setattr(pool, "bind", lambda *args: None)
    monkeypatch.setattr(pool, "authenticate", lambda args: ({}, {}))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    args = argparse.Namespace(
        cpu=1, memory_gb=8, phase="evaluate", task_directory=tmp_path, deadline=time.time() + 1000
    )
    with pytest.raises(RuntimeError, match="interruption"):
        pool.scientific(args)
    assert (tmp_path / "intents.jsonl").exists()
    assert not (tmp_path / "responses.jsonl").exists()
    assert not (tmp_path / "evaluation.json").exists()
    assert execution.isolated is interrupted


def test_native_targets_share_claim_and_fragment_drops_other_recipes(monkeypatch):
    monkeypatch.syspath_prepend(str(PATH.parent))
    suite = {
        "game_seeds": [0, 1, 2, 3],
        "families": [{"id": "feature", "family": "feature_selection", "dataset": "a"}],
        "games": [
            {"id": "tree-sv", "basecase_id": "tree", "oracle": "tree", "dataset": "b"},
            {"id": "tree-sii", "basecase_id": "tree", "oracle": "tree", "dataset": "b"},
        ],
        "focused_design": {
            "recipes": [
                {"id": "feature", "application": "features"},
                {"id": "tree", "application": "local"},
            ]
        },
    }
    inventory = pool.tasks(suite)
    assert len(inventory) == 8
    assert inventory == pool.tasks(suite)
    assert inventory[0]["spec"]["dataset"] != inventory[1]["spec"]["dataset"]
    native = next(row for row in inventory if row["kind"] == "structured")
    selected = pool.fragment(suite, native)
    assert selected["games"] == suite["games"]
    assert "families" not in selected
    assert selected["game_seeds"] == [native["seed"]]
    assert selected["focused_design"]["recipes"] == [{"id": "tree", "application": "local"}]
    assert len(suite["focused_design"]["recipes"]) == 2
