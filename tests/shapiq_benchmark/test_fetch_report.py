"""The Pages downloader authenticates both public layouts without importing shapiq."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from benchmark.fetch_report import fetch_report

from shapiq_benchmark.partitioned import write_partitioned_report
from shapiq_benchmark.record_store import RecordStore
from shapiq_benchmark.report import encode_records
from shapiq_benchmark.runner import digest
from tests.shapiq_benchmark.test_partitioned import fixture_data


def fixture(tmp_path: Path) -> tuple[dict, Path, dict]:
    data = fixture_data()
    with RecordStore(tmp_path / "records.sqlite") as store:
        store.extend(data["records"])
        source = tmp_path / "source"
        manifest = write_partitioned_report({**data, "records": store}, source, block_rows=2)
    return (
        {"url": (source / "data.json").as_uri(), "sha256": digest(source / "data.json")},
        source,
        manifest,
    )


def seal(source: Path, manifest: dict) -> dict:
    (source / "data.json").write_text(json.dumps(manifest))
    return {"url": (source / "data.json").as_uri(), "sha256": digest(source / "data.json")}


def test_partitioned_closure_and_fresh_staging(tmp_path: Path) -> None:
    pin, source, manifest = fixture(tmp_path)
    out = tmp_path / "out"
    result = fetch_report(pin, out)
    assert result["layout"] == "partitioned-v1"
    assert (out / "about.json").read_bytes() == (out / "data.json").read_bytes()
    for descriptors in manifest["assets"].values():
        for descriptor in descriptors:
            assert (out / descriptor["file"]).read_bytes() == (
                source / descriptor["file"]
            ).read_bytes()
    with pytest.raises(ValueError, match="fresh"):
        fetch_report(pin, out)


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "path",
        "private",
        "count",
        "snapshot",
        "dictionary",
        "missing",
        "nonfinite",
        "private_detail",
    ],
)
def test_reject_invalid_partition(tmp_path: Path, fault: str) -> None:
    _, source, manifest = fixture(tmp_path)
    d = manifest["assets"]["details" if fault == "private_detail" else "raw"][0]
    if fault == "hash":
        d["sha256"] = "0" * 64
    elif fault == "path":
        d["file"] = "../bad.json"
    elif fault == "private":
        next(iter(manifest["methods"].values()))["private"] = True
    elif fault == "count":
        manifest["record_count"] += 1
    elif fault == "snapshot":
        d["snapshot_id"] = "0" * 64
    else:
        p = source / d["file"]
        part = json.loads(p.read_text())
        if fault == "dictionary":
            part["columns"]["method"] = {"values": [999] * part["count"], "dictionary": ["x"]}
        elif fault == "missing":
            part["columns"]["method"]["missing"] = [0, 0]
        elif fault == "nonfinite":
            part["columns"]["nmse"]["values"][0] = float("inf")
        else:
            part["columns"]["type"] = {"values": ["game"] * part["count"]}
            part["columns"]["value"] = {"values": [{"truth": [1]}] * part["count"]}
        p.write_text(json.dumps(part))
        d.update(sha256=digest(p), bytes=p.stat().st_size)
    with pytest.raises(ValueError):
        fetch_report(seal(source, manifest), tmp_path / "out")
    assert not (tmp_path / "out/data.json").exists()


def test_legacy_inline_report(tmp_path: Path) -> None:
    data = copy.deepcopy(fixture_data())
    source = tmp_path / "source"
    source.mkdir()
    result = fetch_report(seal(source, data), tmp_path / "out")
    assert result["layout"] == "legacy"
    about = json.loads((tmp_path / "out/about.json").read_text())
    assert about["games"] == data["games"]
    assert "records" not in about


def test_legacy_target_shards(tmp_path: Path) -> None:
    data = fixture_data()
    source = tmp_path / "source"
    source.mkdir()
    part = {
        **encode_records(data["records"]),
        "snapshot_id": data["snapshot_id"],
        "target": "SV · order 1",
    }
    path = source / "records-sv-1.json"
    path.write_text(json.dumps(part))
    data["record_shards"] = [
        {
            "file": path.name,
            "sha256": digest(path),
            "target": part["target"],
            "count": len(data["records"]),
        }
    ]
    data["records"] = []
    result = fetch_report(seal(source, data), tmp_path / "out")
    assert result["files"] == ["about.json", "data.json", path.name]
    assert (tmp_path / "out" / path.name).read_bytes() == path.read_bytes()


@pytest.mark.parametrize("field,value", [("count", True), ("count", 2.0), ("bytes", 200.0)])
def test_reject_coerced_descriptor_numbers(tmp_path: Path, field: str, value: object) -> None:
    _, source, manifest = fixture(tmp_path)
    manifest["assets"]["raw"][0][field] = value
    with pytest.raises(ValueError, match="descriptor"):
        fetch_report(seal(source, manifest), tmp_path / "out")
