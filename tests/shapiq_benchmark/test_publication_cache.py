"""Caching must preserve authentication, complete normalized values and global joins."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from shapiq_benchmark import campaign
from shapiq_benchmark.publication_cache import normalized_batch
from shapiq_benchmark.results_io import Checkpoint
from shapiq_benchmark.runner import digest, identity
from tests.shapiq_benchmark.test_campaign_export import make_campaign
from tests.shapiq_benchmark.test_campaign_recovery import recovery_pair
from tests.shapiq_benchmark.test_campaign_replacements import reruns

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("compact", [False, True])
def test_campaign_cache_skips_only_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, compact: bool
) -> None:
    """Hits still validate raw shards, but skip merge rescoring and payoff analysis."""
    root, cache = tmp_path / "campaign", tmp_path / "cache"
    make_campaign(root)
    if compact:
        for path in root.glob("*/sweep/shard-*/results.json"):
            Checkpoint(path.parent, path.parents[2] / "prepared", json.loads(path.read_text()))
    original = campaign.assemble_campaign(root, 3)
    assert campaign.assemble_campaign(root, 3, cache_dir=cache) == original
    assert len(list(cache.glob("*.json"))) == 2  # Excluded batch needs no entry.
    for name in ("merge_results", "_historical_quality", "payoff_fingerprint"):
        monkeypatch.setattr(
            campaign, name, Mock(side_effect=AssertionError("normalization repeated"))
        )
    reader = Mock(wraps=campaign.read_results)
    monkeypatch.setattr(campaign, "read_results", reader)
    assert campaign.assemble_campaign(root, 3, cache_dir=cache) == original
    assert reader.call_count == (4 if compact else 0)  # Raw validation remains.
    # Added shards cannot hide behind an existing normalized value.
    extra = root / "batch-0/sweep/shard-099/results.json"
    extra.parent.mkdir()
    extra.write_text("{}")
    with pytest.raises(ValueError, match="unexpected result shards"):
        campaign.assemble_campaign(root, 3, cache_dir=cache)
    extra.unlink()
    shard = root / "batch-0/sweep/shard-000/results.json"
    if compact:
        journal = shard.parent / "records.jsonl"
        journal.write_text(journal.read_text() + "{}\n")
        with pytest.raises(ValueError, match="journal is incomplete or changed"):
            campaign.assemble_campaign(root, 3, cache_dir=cache)
        return
    raw = json.loads(shard.read_text())
    raw["records"].pop()
    shard.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="missing, duplicate, or unexpected cells"):
        campaign.assemble_campaign(root, 3, cache_dir=cache)


def test_cache_authenticates_inputs_policy_and_output(tmp_path: Path) -> None:
    """A hit is bound to exact inputs/runtime policy and detects damaged output."""
    raw = tmp_path / "raw.json"
    raw.write_text("original")
    inputs = {str(raw): digest(raw)}
    policy = {"files": {}, "numpy": "old"}
    cache = tmp_path / "cache"
    value = {"records": [{"run_id": "original", "nmse": 1e-300}], "fingerprints": {}}
    build = Mock(return_value=value)
    assert normalized_batch(cache, inputs, policy, build) == value
    assert normalized_batch(cache, inputs, policy, build) == value
    assert build.call_count == 1
    assert normalized_batch(cache, inputs, {**policy, "numpy": "new"}, build) == value
    assert build.call_count == 2
    raw.write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        normalized_batch(cache, inputs, policy, build)
    raw.write_text("original")
    for path in cache.glob("*.json"):
        entry = json.loads(path.read_text())
        entry["value"]["records"][0]["nmse"] = 99
        path.write_text(json.dumps(entry))
    with pytest.raises(ValueError, match="checksum"):
        normalized_batch(cache, inputs, policy, build)


def test_failed_or_mutating_build_is_never_published(tmp_path: Path) -> None:
    """Failed work and changed input files cannot leave a usable partial entry."""
    raw = tmp_path / "raw"
    raw.write_text("before")
    inputs, policy = {str(raw): digest(raw)}, {"files": {}}
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError):
        normalized_batch(cache, inputs, policy, Mock(side_effect=RuntimeError("interrupted")))
    assert not list(cache.iterdir())

    def mutate() -> dict:
        raw.write_text("after")
        return {"panel": []}

    with pytest.raises(ValueError, match="input changed"):
        normalized_batch(cache, inputs, policy, mutate)
    assert not list(cache.iterdir())


def test_concurrent_entries_must_match_exact_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent publication never overwrites an existing, numerically different entry."""
    cache = tmp_path / "cache"
    policy = {"files": {}}
    binding = {"inputs": {}, "policy": policy}
    other = {"number": 1.0}

    def competing_link(_temporary: Path, destination: Path) -> None:
        destination.write_text(
            json.dumps({"binding": binding, "value": other, "sha256": identity(other)})
        )
        raise FileExistsError

    monkeypatch.setattr("shapiq_benchmark.publication_cache.os.link", competing_link)
    with pytest.raises(ValueError, match="conflicting output"):
        normalized_batch(cache, {}, policy, lambda: {"number": 1})
    assert len(list(cache.iterdir())) == 1
    assert json.loads(next(cache.iterdir()).read_text())["value"] == other


def test_warm_cache_preserves_recovery_replacements_and_worker_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacements and cross-campaign aliases still operate on complete original panels."""
    parent, retry = recovery_pair(tmp_path)
    worker = {
        "cpu_model": "EPYC",
        "cpu_affinity": [2, 3],
        "peak_rss_bytes": 123456,
        "process_cpu_seconds": 0.0123456789012345,
        "thread_environment": {"OMP_NUM_THREADS": "1"},
    }
    for path in parent.glob("*/sweep/shard-*/results.json"):
        raw = json.loads(path.read_text())
        for row in raw["records"]:
            row["worker"] = worker
        path.write_text(json.dumps(raw))
    replacements = reruns(tmp_path, [parent, retry])
    options = {"supplements": (retry,), "replacements": replacements}
    original = campaign.assemble_campaign(parent, 3, **options)
    cache = tmp_path / "cache"
    assert campaign.assemble_campaign(parent, 3, cache_dir=cache, **options) == original
    monkeypatch.setattr(campaign, "merge_results", Mock(side_effect=AssertionError("cache miss")))
    actual = campaign.assemble_campaign(parent, 3, cache_dir=cache, **options)
    assert identity(actual) == identity(original)
    assert actual["duplicate_games"]
    assert all(row["worker"] == worker for row in actual["records"])
    corrected = [row for row in actual["records"] if row["method"] == "KernelSHAP"]
    assert corrected and all(row["nmse"] == 0.125 for row in corrected)
    assert all(actual["runs"][row["run_id"]]["source_sha256"] == "new-source" for row in corrected)
