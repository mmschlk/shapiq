"""Tiny real compact records verify durable shard reuse and fail-closed merging."""

from __future__ import annotations

import copy
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from shapiq_benchmark.record_store import RecordStore
from tests.shapiq_benchmark import test_collect_pool as fixtures
from tests.shapiq_benchmark.test_collect_pool import collector, save

campaign = fixtures.campaign
sys.modules["collect_pool"] = collector
spec = importlib.util.spec_from_file_location(
    "pool_checkpoints_tested", Path(__file__).resolve().parents[2] / "benchmark/pool_checkpoints.py"
)
checkpoints = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoints)


def two_cases(config, directory):
    """Add one genuinely unclaimed second seed without changing the first game."""
    suite_path = Path(config["suite"]["path"])
    suite = json.loads(suite_path.read_text())
    suite["game_seeds"] = [0, 1]
    config["suite"] = save(suite_path, suite)
    inventory_path = Path(config["inventory"]["path"])
    tasks = json.loads(inventory_path.read_text())
    tasks.append(copy.deepcopy(tasks[0]) | {"case": 1, "seed": 1})
    config["inventory"] = save(inventory_path, tasks)
    for path in (directory / "claim.json", directory.parents[1] / "allocations/1/started.json"):
        value = json.loads(path.read_text())
        value["suite_sha256"] = config["suite"]["sha256"]
        save(path, value)
    path = directory.parents[2] / "collection.json"
    save(path, config)
    return path


def test_shards_match_serial_and_survive_export_failure(campaign, tmp_path):
    config, directory, *_ = campaign
    path = two_cases(config, directory)
    with RecordStore(tmp_path / "serial.sqlite") as store:
        serial, checked = collector.collect(config, store)
        expected = list(store)
    shards = [tmp_path / "first", tmp_path / "second"]
    for case, shard in enumerate(shards):
        checkpoints.create_shard(path, shard, [case])
    merged = tmp_path / "merged"
    checkpoints.merge_shards(path, list(reversed(shards)), merged)
    before = collector.digest(merged / "records.sqlite")
    store, data, audit = checkpoints.open_checkpoint(merged, tmp_path / "working.sqlite")
    try:
        assert list(store) == expected
        assert data["games"] == serial["games"]
        assert data["runs"] == serial["runs"]
        assert data["coverage"] == serial["coverage"]
        assert data["preparation_coverage"] == serial["preparation_coverage"]
        assert audit["statuses"] == checked["statuses"]
        assert audit["cases"] == checked["cases"]
        # Alias/export operations write a fork in the disposable copy only.
        store.fork().extend(expected)
    finally:
        store.close()
    assert not (tmp_path / "working.sqlite").exists()
    assert collector.digest(merged / "records.sqlite") == before
    with checkpoints.readonly_store(merged / "records.sqlite") as view:
        assert list(view) == expected
        cells = [(r["game_id"], r["budget"], r["seed"]) for r in expected]
        assert list(view.measurements(cells, [expected[0]["method"]]))
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            view.extend(expected)
    assert (merged / "records.sqlite").exists()


def test_missing_and_overlapping_shards_rejected(campaign, tmp_path):
    config, directory, *_ = campaign
    path = two_cases(config, directory)
    shard = tmp_path / "first"
    checkpoints.create_shard(path, shard, [0])
    for directories in ([shard], [shard, shard]):
        with pytest.raises(ValueError, match="overlap or omit"):
            checkpoints.merge_shards(path, directories, tmp_path / "bad")
    with pytest.raises(ValueError, match="complete case inventory"):
        checkpoints.open_checkpoint(shard)


def test_changed_database_and_raw_input_rejected(campaign, tmp_path):
    config, directory, *_ = campaign
    path = tmp_path / "collection.json"
    save(path, config)
    shard = tmp_path / "only"
    checkpoints.create_shard(path, shard, [0])
    claim = directory / "claim.json"
    original = claim.read_bytes()
    claim.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="inputs changed"):
        checkpoints.open_checkpoint(shard)
    claim.write_bytes(original)
    with (shard / "records.sqlite").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Checkpoint bytes changed"):
        checkpoints.open_checkpoint(shard)


def test_foreign_case_selection_fails(campaign, tmp_path):
    config, *_ = campaign
    with (
        RecordStore(tmp_path / "bad.sqlite") as store,
        pytest.raises(ValueError, match="Invalid explicit"),
    ):
        collector.collect(config, store, case_ids=[1])


def test_completed_shard_and_merge_resume_without_collection(campaign, tmp_path, monkeypatch):
    config, directory, *_ = campaign
    path = two_cases(config, directory)
    shards = [tmp_path / "a", tmp_path / "b"]
    receipts = [checkpoints.create_shard(path, shard, [i]) for i, shard in enumerate(shards)]

    def forbidden(*args, **kwargs):
        pytest.fail("Completed checkpoint must not rescore records")

    monkeypatch.setattr(collector, "collect", forbidden)
    for i, shard in enumerate(shards):
        assert checkpoints.create_shard(path, shard, [i]) == receipts[i]
    merged = tmp_path / "merged"
    receipt = checkpoints.merge_shards(path, shards, merged)
    assert checkpoints.merge_shards(path, shards, merged) == receipt
    claim = directory / "claim.json"
    claim.write_text(claim.read_text() + "\n")
    with pytest.raises(ValueError, match="inputs changed"):
        checkpoints.create_shard(path, shards[0], [0])
    with pytest.raises(ValueError, match="inputs changed"):
        checkpoints.merge_shards(path, shards, merged)
