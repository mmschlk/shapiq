"""Parallel caches must preserve every serial preset and resume without recomputing."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from shapiq_benchmark import partitioned
from shapiq_benchmark.record_store import RecordStore
from shapiq_benchmark.summary import iter_summaries
from tests.shapiq_benchmark.test_partitioned import fixture_data

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
import parallel_summaries


def test_parallel_presets_match_serial_and_resume(tmp_path, monkeypatch):
    data = fixture_data()
    # Exercise the independent diagnostic target alongside real/core/control panels.
    synthetic = copy.deepcopy(data["games"][0])
    synthetic["id"] += "-diagnostic"
    synthetic["metadata"]["synthetic"] = True
    data["games"].append(synthetic)
    data["records"].extend(
        [
            {**r, "game_id": synthetic["id"]}
            for r in data["records"]
            if r["game_id"] == data["games"][0]["id"]
        ]
    )
    if "budgets_by_game" in data["suite"]:
        data["suite"]["budgets_by_game"][synthetic["id"]] = data["suite"]["budgets_by_game"][
            data["games"][0]["id"]
        ]
    original = partitioned.iter_summaries
    with RecordStore(tmp_path / "records.sqlite") as store:
        store.extend(data["records"])
        data["records"] = store
        expected = {
            (degree, controls): list(
                iter_summaries(data, score_order=degree, include_controls=controls)
            )
            for controls in [False, True]
            for degree in [None, 1, 2]
        }
        database_before = parallel_summaries._digest(tmp_path / "records.sqlite")
        with parallel_summaries.cached_summaries(data, tmp_path / "cache", 2) as directory:
            for (degree, controls), serial in expected.items():
                actual = list(
                    partitioned.iter_summaries(data, score_order=degree, include_controls=controls)
                )
                # Full exact comparison includes IDs, ordered presets, errors, ratings and CIs.
                assert actual == serial
            with pytest.raises(ValueError, match="different data/options"):
                list(partitioned.iter_summaries(data, bootstrap_draws=1))
        assert partitioned.iter_summaries is original
        assert parallel_summaries._digest(tmp_path / "records.sqlite") == database_before
        done_files = sorted(directory.glob("*.done.json"))
        before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in done_files}

        def unexpected_executor(**_kwargs):
            message = "Completed summary tasks were recomputed"
            raise AssertionError(message)

        monkeypatch.setattr(parallel_summaries, "ProcessPoolExecutor", unexpected_executor)
        with parallel_summaries.cached_summaries(data, tmp_path / "cache", 2):
            assert list(partitioned.iter_summaries(data)) == expected[None, False]
        assert before == {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in done_files}
        receipt = json.loads(done_files[0].read_text())
        output, _ = parallel_summaries._task_paths(directory, receipt["task"])
        with output.open("a") as stream:
            stream.write("{}\n")
        with (
            pytest.raises(ValueError, match="checkpoint changed"),
            parallel_summaries.cached_summaries(data, tmp_path / "cache", 2),
        ):
            pass
        assert partitioned.iter_summaries is original
