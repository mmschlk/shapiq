"""Historical publication import preserves identities and fails atomically."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

import pytest

from shapiq_benchmark import published
from shapiq_benchmark.published import merge_method_catalogs, read_published
from shapiq_benchmark.record_store import RecordStore
from shapiq_benchmark.report import compact_workers, encode_records, write_report
from shapiq_benchmark.runner import digest, identity

if TYPE_CHECKING:
    from pathlib import Path


def panel() -> dict:
    """Provide two targets, distinct worker field presence and real source identities."""
    source = {"source_sha256": "a" * 64, "git_commit": "b" * 40, "source_dirty": False}
    run_id = identity({"original_unsanitized_result": "not its published projection"})
    games = [
        {
            "id": index,
            "family": "local",
            "stratum": "data",
            "n_players": 11,
            "index": index,
            "order": order,
            "metadata": {"instance_seed": 0},
        }
        for index, order in [("SV", 1), ("SII", 2)]
    ]
    rows: list[dict] = [
        {
            "game_id": game["id"],
            "method": "KernelSHAP",
            "budget": budget,
            "seed": 0,
            "status": "ok",
            "mse": 0.0,
            "nmse": -0.0,
            "queries": budget,
            "seconds": 0.5,
            "run_id": run_id,
            "order_scores": {"1": {"mse": 0, "nmse": -0.0}},
        }
        for game in games
        for budget in [11, 22]
    ]
    rows[0]["worker"] = {
        "cpu_model": "cpu",
        "affinity": [3],
        "peak_rss_bytes": 10,
        "process_cpu_seconds": -0.0,
    }
    rows[1]["worker"] = {"cpu_model": "cpu", "affinity": [3], "peak_rss_bytes": None}
    rows[2]["worker"] = None
    return {
        "schema_version": 1,
        "snapshot_provenance": copy.deepcopy(source),
        "snapshot_id": identity({"historical": True}),
        "games": games,
        "methods": {
            "KernelSHAP": {
                "private": False,
                "parameters": {},
                "factory": "shapiq.KernelSHAP",
                "software_sha256": identity(source),
                "source_sha256": source["source_sha256"],
            }
        },
        "runs": {run_id: {**source, "execution": {"original_worker_protocol": "retained"}}},
        "suite": {"budgets": [11, 22], "seeds": [0]},
        "records": rows,
    }


def save(path: Path, value: dict) -> None:
    """Write fixture JSON without changing numeric types or field presence."""
    path.write_text(json.dumps(value, allow_nan=False))


def fixture(tmp_path: Path, *, nested: bool = False) -> tuple[dict, Path]:
    """Use the actual worker/column codecs with small deterministic public panels."""
    data = panel()
    root = tmp_path / "site"
    root.mkdir()
    encoded = compact_workers(data)
    descriptors = []
    for target, order in [("SV", 1), ("SII", 2)]:
        rows = [row for row in encoded["records"] if row["game_id"] == target]
        name = f"records-{target.lower()}-{order}.json"
        payload = {
            **encode_records(rows, nested=nested),
            "snapshot_id": data["snapshot_id"],
            "target": f"{target} · order {order}",
            "presets": [],
        }
        save(root / name, payload)
        descriptors.append(
            {
                "file": name,
                "sha256": digest(root / name),
                "count": len(rows),
                "target": payload["target"],
            }
        )
    save(
        root / "data.json",
        {
            **encoded,
            "records": [],
            "presets": [],
            "record_shards": descriptors,
            "record_count": 4,
            "evaluated_count": 4,
        },
    )
    return data, root / "data.json"


def pins(index_path: Path) -> dict:
    """Take explicit reviewed-driver style pins; the reader never infers approval."""
    index = json.loads(index_path.read_text())
    return {
        "index_sha256": digest(index_path),
        "shard_sha256": {item["file"]: item["sha256"] for item in index["record_shards"]},
    }


def repin(index_path: Path, index: dict) -> None:
    """Authenticate malformed fixture bytes to exercise schema rather than hashes."""
    for item in index["record_shards"]:
        item["sha256"] = digest(index_path.parent / item["file"])
    save(index_path, index)


@pytest.mark.parametrize("nested", [False, True])
def test_exact_roundtrip_and_atomic_destination(tmp_path: Path, *, nested: bool) -> None:
    """Original runs, exact rows and non-wire metadata survive both column codecs."""
    data, path = fixture(tmp_path, nested=nested)
    with RecordStore(tmp_path / "rows.sqlite") as store:
        actual = read_published(path, record_store=store, **pins(path))
        assert identity({"rows": list(store)}) == identity({"rows": data["records"]})
        assert identity({k: v for k, v in actual.items() if k != "records"}) == identity(
            {k: v for k, v in data.items() if k != "records"}
        )
        with pytest.raises(ValueError, match="empty"):
            read_published(path, record_store=store, **pins(path))
        assert len(store) == 4


def test_current_public_writer_compatibility(tmp_path: Path) -> None:
    """Current published output, including derived preset encoding, imports unchanged."""
    data = panel()
    root = tmp_path / "written"
    write_report(data, root, public=True, compact=True)
    with RecordStore(tmp_path / "rows.sqlite") as store:
        read_published(root / "data.json", record_store=store, **pins(root / "data.json"))
        assert identity({"rows": list(store)}) == identity({"rows": data["records"]})


@pytest.mark.parametrize(
    "mutation",
    [
        "snapshot",
        "target",
        "count",
        "unknown_game",
        "unknown_method",
        "unknown_run",
        "private_field",
        "negative_queries",
        "boolean_score",
        "string_score",
        "order_score",
        "duplicate",
        "worker",
        "ambiguous_worker",
    ],
)
def test_late_shard_failure_leaves_destination_empty(tmp_path: Path, mutation: str) -> None:
    """Even a fully authenticated invalid final shard cannot partially import history."""
    _data, path = fixture(tmp_path)
    index = json.loads(path.read_text())
    shard_path = path.parent / index["record_shards"][-1]["file"]
    shard = json.loads(shard_path.read_text())
    if mutation in {"snapshot", "target", "count"}:
        shard[{"snapshot": "snapshot_id"}.get(mutation, mutation)] = "wrong"
    else:
        rows = list(published._rows(shard))
        field, value = {
            "unknown_game": ("game_id", "unknown"),
            "unknown_method": ("method", "unknown"),
            "unknown_run": ("run_id", "unknown"),
            "private_field": ("error", "/private/path"),
            "negative_queries": ("queries", -1),
            "boolean_score": ("nmse", True),
            "string_score": ("nmse", "oops"),
            "order_score": ("order_scores", {"1": {"nmse": -1, "mse": 0}}),
            "worker": ("worker_id", "unknown"),
            "ambiguous_worker": ("worker_id", "w0"),
            "duplicate": ("budget", 11),
        }[mutation]
        rows[-1 if mutation == "duplicate" else 0][field] = value
        shard = {**shard, **encode_records(rows)}
    save(shard_path, shard)
    repin(path, index)
    with RecordStore(tmp_path / "rows.sqlite") as store, pytest.raises(ValueError):
        try:
            read_published(path, record_store=store, **pins(path))
        finally:
            assert len(store) == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "private",
        "schema_bool",
        "players_bool",
        "game_private",
        "unexpected_shard",
        "missing_pin",
        "symlink",
        "changed_hash",
    ],
)
def test_index_authentication_and_path_confinement(tmp_path: Path, mutation: str) -> None:
    """Pins, source membership and confined exact shard inventories are mandatory."""
    _data, path = fixture(tmp_path)
    index = json.loads(path.read_text())
    if mutation == "source":
        next(iter(index["runs"].values())).pop("source_sha256")
    elif mutation == "private":
        index["methods"]["KernelSHAP"]["private"] = True
    elif mutation == "schema_bool":
        index["schema_version"] = True
    elif mutation == "players_bool":
        index["games"][0]["n_players"] = True
    elif mutation == "game_private":
        index["games"][0]["artifact"] = "/private/file"
    elif mutation == "unexpected_shard":
        save(path.parent / "records-extra-1.json", {})
    elif mutation == "symlink":
        shard = path.parent / index["record_shards"][0]["file"]
        outside = tmp_path / "outside.json"
        shard.rename(outside)
        shard.symlink_to(outside)
    save(path, index)
    options = pins(path)
    if mutation == "missing_pin":
        options["shard_sha256"].pop(index["record_shards"][0]["file"])
    elif mutation == "changed_hash":
        path.write_text(path.read_text() + " ")
    with RecordStore(tmp_path / "rows.sqlite") as store, pytest.raises(ValueError):
        read_published(path, record_store=store, **options)


@pytest.mark.parametrize("text", ['{"x":1e999}', '{"x":{"y":NaN}}', '{"x":1,"x":2}'])
def test_invalid_json_numbers_and_duplicate_keys(tmp_path: Path, text: str) -> None:
    """Authentication never makes nonfinite or ambiguous JSON acceptable."""
    path = tmp_path / "invalid.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        published._read(path, digest(path))


def test_final_rehash_rejects_changed_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late input change aborts after staging but before caller-visible insertion."""
    _data, path = fixture(tmp_path)
    options = pins(path)
    original = published._read

    def changed(file: Path, expected: str) -> dict:
        value = original(file, expected)
        if file.name == "records-sii-2.json":
            path.write_text(path.read_text() + " ")
        return value

    monkeypatch.setattr(published, "_read", changed)
    with RecordStore(tmp_path / "rows.sqlite") as store, pytest.raises(ValueError, match="changed"):
        try:
            read_published(path, record_store=store, **options)
        finally:
            assert len(store) == 0


def test_catalog_union_preserves_singletons_and_arbitrary_versions() -> None:
    """Three sources retain exact definitions and conflicts cannot overwrite history."""
    first = panel()["methods"]
    catalogs = [first]
    for digit in ["c", "d"]:
        method = copy.deepcopy(first["KernelSHAP"])
        method.update(software_sha256=digit * 64, source_sha256=digit * 64)
        catalogs.append({"KernelSHAP": method})
    before = copy.deepcopy(catalogs)
    assert merge_method_catalogs(first, first) == first
    combined = merge_method_catalogs(*catalogs)
    assert len(combined["KernelSHAP"]["source_versions"]) == 3
    assert merge_method_catalogs(combined, first) == combined
    assert catalogs == before
    catalogs[-1]["KernelSHAP"]["software_sha256"] = catalogs[0]["KernelSHAP"]["software_sha256"]
    with pytest.raises(ValueError, match="same software"):
        merge_method_catalogs(*catalogs)


@pytest.mark.parametrize("value", [0, -0.0, True])
def test_catalog_parameter_identity_uses_json_types(value: object) -> None:
    """Numerically equal Python values do not erase recorded configuration identity."""
    first = panel()["methods"]
    first["KernelSHAP"]["parameters"] = {"option": 0.0}
    second = copy.deepcopy(first)
    second["KernelSHAP"]["parameters"]["option"] = value
    with pytest.raises(ValueError, match="Conflicting method"):
        merge_method_catalogs(first, second)


@pytest.mark.parametrize("versions", [None, {}, {"short": "a" * 64}])
def test_catalog_rejects_ambiguous_or_invalid_versions(versions: object) -> None:
    """A source_versions field is never silently treated as an absent field."""
    first = panel()["methods"]
    first["KernelSHAP"]["source_versions"] = versions
    with pytest.raises(ValueError):
        merge_method_catalogs(first)
