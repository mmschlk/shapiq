"""Lossless opt-in blocks preserve exact fields while bounding writer buffers."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path
from typing import cast

import pytest

from shapiq_benchmark import partitioned
from shapiq_benchmark.partitioned import METRIC_FIELDS, write_partitioned_report
from shapiq_benchmark.published import _restore_worker, _rows
from shapiq_benchmark.record_store import RecordStore
from shapiq_benchmark.runner import digest, identity
from shapiq_benchmark.summary import iter_summaries
from tests.shapiq_benchmark.test_published import panel


def fixture_data() -> dict:
    """Mix worker diagnostics, source execution presence, failures and two targets."""
    data = panel()
    data["games"][0]["metadata"].update(
        dataset="example",
        model_profile="forest",
        case_id="point0",
        synthetic=False,
        evaluation_timing={"backend": "cpu", "thread_environment": {"OMP_NUM_THREADS": "1"}},
        order_scores={"1": {"score_eligible": True, "energy_share": 1.0}},
        game_quality={"role": "core", "large_diagnostic": list(range(50))},
    )
    data["games"][1]["metadata"].update(
        dataset="other",
        model="linear",
        game_quality={"role": "control"},
        order_scores={
            "1": {"score_eligible": True, "energy_share": 0.6},
            "2": {"score_eligible": False, "energy_share": 0.4},
        },
    )
    data["methods"]["SVARM"] = copy.deepcopy(data["methods"]["KernelSHAP"])
    data["records"] = [*data["records"], *[{**row, "method": "SVARM"} for row in data["records"]]]
    data["records"][1].update(
        status="failed", mse=None, nmse=None, failure_reason="timeout", timing_profile="diagnostic"
    )
    data["records"][3].update(status="unsupported", mse=None, nmse=None, queries=0)
    cast("dict", data["records"][0]["worker"]).update(
        thread_pools=[{"num_threads": 1}], machine="x86_64"
    )
    data["runs"]["absent-execution"] = {"source_sha256": "d" * 64}
    data["runs"]["null-execution"] = {"source_sha256": "d" * 64, "execution": None}
    return data


def decoded(root: Path, manifest: dict, kind: str) -> list[dict]:
    """Verify each descriptor and decode only the small fixture under test."""
    rows = []
    for descriptor in manifest["assets"][kind]:
        path = root / descriptor["file"]
        assert path.stat().st_size == descriptor["bytes"] and digest(path) == descriptor["sha256"]
        payload = json.loads(path.read_text())
        for field in ["snapshot_id", "kind", "count", "target", "family", "method"]:
            assert payload.get(field) == descriptor.get(field)
        rows.extend(_rows(payload))
    return rows


def test_lossless_blocks_and_exact_streamed_presets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No raw store iteration is used, and every original measurement/detail restores."""
    data = fixture_data()
    original = copy.deepcopy(data)
    expected = [
        p
        for include in [False, True]
        for degree in [None, 1, 2]
        for p in iter_summaries(data, include_controls=include, score_order=degree)
    ]
    with RecordStore(tmp_path / "rows.sqlite") as store:
        store.extend(data["records"])
        data["records"] = store

        def forbidden(*args, **kwargs):
            message = "The writer must use bounded indexed selections"
            raise AssertionError(message)

        monkeypatch.setattr(RecordStore, "__iter__", forbidden)
        root = tmp_path / "report"
        manifest = write_partitioned_report(data, root, block_rows=2, max_bytes=8192)
        assert manifest == json.loads((root / "data.json").read_text())
        assert manifest["layout"] == "partitioned-v1" and manifest["record_count"] == 8
        assert manifest["evaluated_count"] == 7
        assert not {"records", "presets", "runs", "workers", "games"}.intersection(manifest)
        assert all(
            d["count"] <= 2 and d["bytes"] <= 8192
            for entries in manifest["assets"].values()
            for d in entries
        )
        assert decoded(root, manifest, "summaries") == expected
        profiles = {row["id"]: row["value"] for row in decoded(root, manifest, "profiles")}
        raw = decoded(root, manifest, "raw")
        restored = []
        for row in sorted(raw, key=lambda r: r["sequence"]):
            restored_row = _restore_worker(copy.deepcopy(row), profiles)
            restored_row.pop("sequence")
            restored.append(restored_row)
        assert identity({"rows": restored}) == identity({"rows": original["records"]})
        metrics = decoded(root, manifest, "metrics")
        assert identity({"rows": sorted(metrics, key=lambda r: r["sequence"])}) == identity(
            {
                "rows": [
                    {k: v for k, v in row.items() if k in METRIC_FIELDS}
                    for row in sorted(raw, key=lambda r: r["sequence"])
                ]
            }
        )
        assert not any("worker" in row or "worker_process_cpu_seconds" in row for row in metrics)
        runs = {}
        for row in decoded(root, manifest, "runs"):
            run = copy.deepcopy(profiles[row["source_id"]])
            if "execution" in row:
                run["execution"] = row["execution"]
            runs[row["id"]] = run
        assert identity(runs) == identity(original["runs"])
        details = decoded(root, manifest, "details")
        assert [r["value"] for r in details if r["type"] == "game"] == original["games"]
        metadata = {r["id"]: r["value"] for r in details if r["type"] == "report"}
        assert identity(metadata) == identity(
            {k: v for k, v in original.items() if k not in {"records", "games", "runs", "methods"}}
        )
        games = decoded(root, manifest, "games")
        assert (
            games[0]["metadata"]["evaluation_timing"]
            == original["games"][0]["metadata"]["evaluation_timing"]
        )
        assert (
            games[1]["metadata"]["order_scores"] == original["games"][1]["metadata"]["order_scores"]
        )
        assert "large_diagnostic" not in games[0]["metadata"]["game_quality"]
        assert manifest["method_targets"]["KernelSHAP"]["unsupported"] == ["SII · order 2"]


def test_indexed_selection_preserves_order_and_interleaved_lifetimes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sequence keys retain original ordering through filters and independent cursors."""
    data = fixture_data()
    with RecordStore(tmp_path / "rows.sqlite") as store:
        store.extend(reversed(data["records"]))
        all_rows = list(store.select_indexed())
        assert [row for _, row in all_rows] == list(reversed(data["records"]))
        first = store.select_indexed(game_ids=["SV"])
        second = store.select(game_ids=["SII"])
        assert next(first)[1]["game_id"] == "SV"
        assert next(second)["game_id"] == "SII"
        first.close()
        assert all(row["game_id"] == "SII" for row in second)
        selected = list(store.select_indexed(methods=["KernelSHAP"]))
        assert selected == [(seq, row) for seq, row in all_rows if row["method"] == "KernelSHAP"]
        assert list(store.select_indexed(methods=[])) == []
        held = store.select_indexed(game_ids=["SV"])
        monkeypatch.setattr(store, "select_indexed", lambda **kwargs: held)
        outer = store.select()
        next(outer)
        outer.close()
        with pytest.raises(StopIteration):
            next(held)


@pytest.mark.parametrize(
    "bad",
    [
        "private",
        "raw_error",
        "sequence",
        "unknown_game",
        "unknown_method",
        "unknown_run",
        "duplicate",
        "snapshot",
        "schema_bool",
    ],
)
def test_rejects_invalid_publication_without_partial_manifest(tmp_path: Path, bad: str) -> None:
    """Privacy, catalog and identity failures leave no partially accepted output."""
    data = fixture_data()
    if bad == "private":
        data["methods"]["KernelSHAP"]["private"] = True
    elif bad == "snapshot":
        data["snapshot_id"] = "not-a-hash"
    elif bad == "schema_bool":
        data["schema_version"] = True
    else:
        field, value = {
            "raw_error": ("error", "/private/path"),
            "sequence": ("sequence", 10),
            "unknown_game": ("game_id", "absent"),
            "unknown_method": ("method", "absent"),
            "unknown_run": ("run_id", "absent"),
            "duplicate": ("status", "duplicate"),
        }[bad]
        data["records"][0][field] = value
    with RecordStore(tmp_path / "rows.sqlite") as store:
        store.extend(data["records"])
        data["records"] = store
        root = tmp_path / "report"
        with pytest.raises(ValueError):
            write_partitioned_report(data, root)
        assert not root.exists()
        assert len(store) == 8


def test_late_summary_failure_and_existing_directory_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup owns only the new directory and never overwrites an existing report."""
    data = fixture_data()
    with RecordStore(tmp_path / "rows.sqlite") as store:
        store.extend(data["records"])
        data["records"] = store
        root = tmp_path / "report"
        root.mkdir()
        marker = root / "existing.txt"
        marker.write_text("untouched")
        with pytest.raises(FileExistsError):
            write_partitioned_report(data, root)
        assert marker.read_text() == "untouched"
        root = tmp_path / "new"

        def fail(*args, **kwargs):
            assert list(root.glob("partition-raw-*.json"))
            message = "late summary failure"
            raise ValueError(message)

        monkeypatch.setattr(partitioned, "iter_summaries", fail)
        with pytest.raises(ValueError, match="late summary"):
            write_partitioned_report(data, root)
        assert not root.exists()


def test_byte_bound_applies_after_encoding_and_rejects_oversized_metadata(tmp_path: Path) -> None:
    """Encoding overhead cannot evade a cap; large individual objects fail explicitly."""
    root = tmp_path / "blocks"
    root.mkdir()
    blocks = partitioned._Blocks(root, "a" * 64, 100, 700)
    rows = [{str(i): n for i in range(12)} for n in range(10)]
    blocks.write("metrics", rows)
    assert all(d["bytes"] <= 700 for d in blocks.assets["metrics"])
    assert [
        r
        for d in blocks.assets["metrics"]
        for r in _rows(json.loads((root / d["file"]).read_text()))
    ] == rows
    with pytest.raises(ValueError, match="Encoded publication row"):
        partitioned._Blocks(root, "a" * 64, 100, 100).emit("metrics", [{"a": 1}], {})
    data = fixture_data()
    data["composition"] = {"large": "x" * 20000}
    with RecordStore(tmp_path / "rows.sqlite") as store:
        store.extend(data["records"])
        data["records"] = store
        with pytest.raises(ValueError, match="single publication row"):
            write_partitioned_report(data, tmp_path / "oversize", max_bytes=8192)
        assert not (tmp_path / "oversize").exists()


def test_actual_browser_reader_restores_python_writer(tmp_path: Path) -> None:
    """The shipped JS column accessor decodes real Python blocks, not a parallel codec."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for the browser decoder integration")
    data = fixture_data()
    expected = {
        **data,
        "presets": [
            p
            for include in [False, True]
            for degree in [None, 1, 2]
            for p in iter_summaries(data, include_controls=include, score_order=degree)
        ],
    }
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps(expected, allow_nan=False))
    with RecordStore(tmp_path / "rows.sqlite") as store:
        store.extend(data["records"])
        write_partitioned_report(
            {**data, "records": store}, tmp_path / "report", block_rows=2, max_bytes=8192
        )
    script = tmp_path / "reader.cjs"
    script.write_text(r"""
const fs = require("node:fs"), path = require("node:path"), assert = require("node:assert/strict");
globalThis.crypto = require("node:crypto").webcrypto;
require(process.argv[2]);
(async () => {
  const root = process.argv[3], expected = JSON.parse(fs.readFileSync(process.argv[4], "utf8"));
  const manifest = JSON.parse(fs.readFileSync(path.join(root, "data.json"), "utf8"));
  const rows = {}, files = new Map();
  for (const entries of Object.values(manifest.assets)) for (const d of entries)
    files.set(d.file, new Blob([fs.readFileSync(path.join(root, d.file))]));
  for (const [kind, entries] of Object.entries(manifest.assets)) {
    rows[kind] = [];
    for (const d of entries) {
      const block = await BenchmarkPartitions.read(manifest, d, {files});
      for (let i = 0; i < block.count; i++) rows[kind].push(block.row(i));
    }
  }
  const profiles = Object.fromEntries(rows.profiles.map(r => [r.id, r.value]));
  const records = rows.raw.sort((a,b) => a.sequence - b.sequence).map(row => {
    const result = {...row}; delete result.sequence;
    if (Object.hasOwn(result, "worker_id")) {
      const worker = structuredClone(profiles[result.worker_id]); delete result.worker_id;
      for (const field of ["peak_rss_bytes", "process_cpu_seconds"]) {
        const key = "worker_" + field;
        if (Object.hasOwn(result, key)) {worker[field] = result[key]; delete result[key];}
      }
      result.worker = worker;
    }
    return result;
  });
  assert.deepStrictEqual(records, expected.records);
  const runs = Object.fromEntries(rows.runs.map(row => {
    const run = structuredClone(profiles[row.source_id]);
    if (Object.hasOwn(row, "execution")) run.execution = row.execution;
    return [row.id, run];
  }));
  assert.deepStrictEqual(runs, expected.runs);
  assert.deepStrictEqual(rows.summaries, expected.presets);
  assert.deepStrictEqual(rows.details.filter(r => r.type === "game").map(r => r.value), expected.games);
  assert.deepStrictEqual(manifest.methods, expected.methods);
  assert(Object.is(records[0].nmse, -0));
  assert(Object.is(records[0].worker.process_cpu_seconds, -0));
  assert(Object.hasOwn(records[2], "worker") && records[2].worker === null);
  assert(!Object.hasOwn(records[3], "worker"));
  console.log("Python writer/browser decoder: exact records, provenance, presets and details PASS");
})().catch(error => {console.error(error); process.exit(1);});
""")
    reader = Path(__file__).resolve().parents[2] / "benchmark/site/partitions.js"
    result = subprocess.run(
        [node, str(script), str(reader), str(tmp_path / "report"), str(expected_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_grouped_selection_uses_cell_index_not_whole_partition(tmp_path: Path) -> None:
    """The repeated writer query must narrow by game/method before sorting by sequence."""
    with RecordStore(tmp_path / "rows.sqlite") as store:
        store.extend(fixture_data()["records"])
        connection = store._connection
        queries = []
        connection.set_trace_callback(queries.append)
        selected = list(store.select_indexed(game_ids=["SV"], methods=["KernelSHAP"]))
        connection.set_trace_callback(None)
        query = next(q for q in queries if q.startswith("SELECT sequence,value"))
        plan = "\n".join(row[3] for row in connection.execute("EXPLAIN QUERY PLAN " + query))
        assert "record_cells (partition=? AND game_id=? AND method=?)" in plan
        assert "record_order" not in plan
        assert len(selected) == 2 and selected == sorted(selected, key=lambda row: row[0])
        indexes = list(connection.execute("PRAGMA index_list(records)"))
        assert [row[1] for row in indexes if row[2]] == ["record_cells"]
