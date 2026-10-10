"""Read authenticated historical public rows without inventing new run identities.

Pins come from the caller's reviewed publication plan. Decoding is not scientific
review or publication approval; records enter a caller-owned disposable store.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import TYPE_CHECKING, cast

from shapiq_benchmark.report import GAME_FIELDS, ROW_FIELDS
from shapiq_benchmark.runner import digest, identity

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from shapiq_benchmark.record_store import RecordStore


HASH = re.compile(r"[0-9a-f]{64}")
DIAGNOSTICS = ("peak_rss_bytes", "process_cpu_seconds")


def _require(condition: bool, message: str) -> None:  # noqa: FBT001
    if not condition:
        raise ValueError(message)


def _sources(method: dict) -> dict[str, str]:
    versions = method.get("source_versions")
    if "source_versions" not in method:
        versions = {method.get("software_sha256"): method.get("source_sha256")}
    else:
        _require(
            "software_sha256" not in method and "source_sha256" not in method,
            "Ambiguous method source catalog",
        )
    _require(
        isinstance(versions, dict)
        and bool(versions)
        and all(
            isinstance(key, str)
            and HASH.fullmatch(key)
            and isinstance(value, str)
            and HASH.fullmatch(value)
            for key, value in versions.items()
        ),
        "Invalid method software/source hashes",
    )
    return dict(cast("dict[str, str]", versions))


def merge_method_catalogs(*catalogs: dict[str, dict]) -> dict[str, dict]:
    """Union any number of source versions without relabeling unchanged methods."""
    result = {}
    version_fields = {"software_sha256", "source_sha256", "source_versions"}
    for catalog in catalogs:
        for name, method in catalog.items():
            incoming = _sources(method)
            if name not in result:
                result[name] = copy.deepcopy(method)
                continue
            previous = result[name]
            common = {key: value for key, value in previous.items() if key not in version_fields}
            _require(
                identity(common)
                == identity(
                    {key: value for key, value in method.items() if key not in version_fields}
                ),
                "Conflicting method parameters, factory, or public status",
            )
            versions = _sources(previous)
            for software, source in incoming.items():
                _require(
                    software not in versions or versions[software] == source,
                    "Conflicting source for the same software identity",
                )
                versions[software] = source
            if versions != _sources(previous):
                result[name] = {**common, "source_versions": versions}
    return result


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, "Repeated JSON object key")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    message = f"Nonfinite published JSON number: {value}"
    raise ValueError(message)


def _float(value: str) -> float:
    number = float(value)
    _require(math.isfinite(number), "Nonfinite published JSON number")
    return number


def _read(path: Path, expected: str) -> dict:
    _require(
        isinstance(expected, str) and HASH.fullmatch(expected) is not None,
        "Invalid publication hash",
    )
    content = path.read_bytes()
    _require(hashlib.sha256(content).hexdigest() == expected, "Published input hash changed")
    value = json.loads(
        content, object_pairs_hook=_object, parse_constant=_nonfinite, parse_float=_float
    )
    _require(isinstance(value, dict), "Published file must contain an object")
    return value


def _rows(payload: dict) -> Iterator[dict]:
    """Decode one row at a time, preserving missing fields and exact JSON values."""
    count = payload.get("count")
    _require(
        payload.get("codec") in {"columns-v1", "columns-v2"}
        and type(count) is int
        and count >= 0
        and isinstance(payload.get("columns"), dict),
        "Invalid published columns encoding",
    )
    columns = []
    for field, column in payload["columns"].items():
        _require(
            isinstance(column, dict)
            and isinstance(column.get("values"), list)
            and len(column["values"]) == count,
            "Invalid published column length",
        )
        missing = column.get("missing", [])
        _require(
            isinstance(missing, list)
            and all(type(i) is int and 0 <= i < count for i in missing)
            and len(set(missing)) == len(missing),
            "Invalid missing-value positions",
        )
        dictionary = column.get("dictionary")
        if "dictionary" in column:
            _require(
                isinstance(dictionary, list)
                and (
                    payload["codec"] == "columns-v2" or all(isinstance(v, str) for v in dictionary)
                ),
                "Invalid published dictionary",
            )
        columns.append((field, column["values"], set(missing), dictionary))
    for position in range(payload["count"]):
        row = {}
        for field, values, missing, dictionary in columns:
            if position in missing:
                continue
            value = values[position]
            if dictionary is not None and value is not None:
                _require(
                    type(value) is int and 0 <= value < len(dictionary),
                    "Invalid published dictionary reference",
                )
                value = dictionary[value]
            row[field] = value
        yield row


def _restore_worker(row: dict, workers: dict) -> dict:
    wire = {f"worker_{field}" for field in DIAGNOSTICS}
    if "worker_id" not in row:
        _require(not wire.intersection(row), "Worker diagnostics lack a profile")
        return row
    _require("worker" not in row, "Ambiguous raw and compact worker metadata")
    key = row["worker_id"]
    _require(
        isinstance(key, str) and key in workers and isinstance(workers[key], dict),
        "Unknown worker profile",
    )
    worker = copy.deepcopy(workers[key])
    restored = {key: value for key, value in row.items() if key not in wire | {"worker_id"}}
    for field in DIAGNOSTICS:
        if f"worker_{field}" in row:
            worker[field] = row[f"worker_{field}"]
    restored["worker"] = worker
    return restored


def read_published(
    index_path: Path,
    *,
    index_sha256: str,
    shard_sha256: dict[str, str],
    record_store: RecordStore,
) -> dict:
    """Import target shards sequentially and atomically into an empty record store.

    Preserve original run IDs, provenance, games and method definitions. Worker
    wire IDs/scalar columns are restored to their original nested fields. Old
    presets and other derived wire containers are omitted for recomputation.
    """
    _require(len(record_store) == 0, "Historical import requires an empty record store")
    index_path = index_path.resolve()
    root = index_path.parent
    index = _read(index_path, index_sha256)
    _require(
        type(index.get("schema_version")) is int
        and index["schema_version"] == 1
        and isinstance(index.get("snapshot_id"), str)
        and index.get("records") == []
        and index.get("presets") == [],
        "Historical import requires a target-sharded public report",
    )
    games = {game["id"]: game for game in index["games"]}
    _require(len(games) == len(index["games"]) and bool(games), "Repeated or empty game inventory")
    for game in games.values():
        _require(
            set(GAME_FIELDS) <= game.keys() and not set(game) - {*GAME_FIELDS, "metadata"},
            "Private, missing, or unknown game fields",
        )
        _require(
            all(
                isinstance(game[key], str) and game[key]
                for key in ("id", "family", "stratum", "index")
            )
            and re.fullmatch(r"[A-Za-z-]+", game["index"]) is not None
            and type(game["order"]) is int
            and game["order"] >= 1
            and type(game["n_players"]) is int
            and game["n_players"] >= 1,
            "Invalid public game identity or dimensions",
        )
    methods = index["methods"]
    allowed_sources = {name: _sources(method) for name, method in methods.items()}
    _require(
        bool(methods) and all(method.get("private") is False for method in methods.values()),
        "Historical report contains private methods",
    )
    runs = index["runs"]
    run_sources = {
        key: (
            identity({k: v for k, v in run.items() if k != "execution"}),
            run.get("source_sha256"),
        )
        for key, run in runs.items()
    }
    descriptors = index["record_shards"]
    names = [item["file"] for item in descriptors]
    targets = {f"{game['index']} · order {game['order']}" for game in games.values()}
    _require(
        len(names) == len(set(names)) == len(targets)
        and {item["target"] for item in descriptors} == targets
        and set(names) == set(shard_sha256)
        and {path.name for path in root.glob("records-*.json")} == set(names),
        "Missing, repeated, or unexpected published shard paths or targets",
    )
    checked = {index_path: index_sha256}
    staged = record_store.fork()
    evaluated = 0

    def decoded_rows(payload: dict, target: str) -> Iterator[dict]:
        nonlocal evaluated
        for decoded in _rows(payload):
            row = _restore_worker(decoded, index.get("workers", {}))
            _require(not set(row) - {*ROW_FIELDS, "run_id"}, "Private or unknown record fields")
            game = games.get(row.get("game_id"))
            _require(
                game is not None and row.get("method") in methods and row.get("run_id") in runs,
                "Unknown published game, method, or run",
            )
            game = games[row["game_id"]]
            _require(
                f"{game['index']} · order {game['order']}" == target,
                "Record appears in the wrong target shard",
            )
            software, source = run_sources[row["run_id"]]
            _require(
                software in allowed_sources[row["method"]]
                and allowed_sources[row["method"]][software] == source,
                "Record run source differs from its method catalog",
            )
            suite = index["suite"]
            grid = suite.get("budgets_by_game", {}).get(game["id"], suite["budgets"])
            _require(
                type(row.get("budget")) is int
                and row["budget"] in grid
                and type(row.get("seed")) is int
                and row["seed"] in suite["seeds"]
                and row.get("status") in {"ok", "failed", "unsupported"},
                "Published record is outside its declared experiment",
            )
            for field in (
                "nmse",
                "mse",
                "seconds",
                "wall_seconds",
                "queries",
                "requested_queries",
                "cache_lookup_seconds",
                "estimated_oracle_seconds",
                "estimated_uncached_seconds",
            ):
                value = row.get(field)
                _require(
                    value is None or (type(value) in (int, float) and value >= 0),
                    f"Invalid numeric published result: {field}",
                )
            _require(
                row["status"] != "ok" or row.get("mse") is not None,
                "Successful published records require MSE",
            )
            for degree, scores in row.get("order_scores", {}).items():
                _require(
                    degree in {str(i) for i in range(1, game["order"] + 1)}
                    and isinstance(scores, dict)
                    and set(scores) == {"nmse", "mse"}
                    and all(
                        v is None or (type(v) in (int, float) and v >= 0) for v in scores.values()
                    ),
                    "Invalid published order score",
                )
            evaluated += row["status"] != "unsupported"
            yield row

    for descriptor in descriptors:
        name = descriptor["file"]
        _require(
            re.fullmatch(r"records-[a-z-]+-[1-9][0-9]*\.json", name) is not None,
            "Invalid published shard filename",
        )
        path = root / name
        _require(path.resolve().parent == root, "Published shard escapes its directory")
        expected = shard_sha256[name]
        _require(descriptor["sha256"] == expected, "Shard pin differs from authenticated index")
        payload = _read(path, expected)
        checked[path] = expected
        _require(
            type(descriptor["count"]) is int
            and descriptor["count"] >= 0
            and payload.get("snapshot_id") == index["snapshot_id"]
            and payload.get("target") == descriptor["target"]
            and payload.get("count") == descriptor["count"],
            "Published snapshot, target, or count mismatch",
        )
        staged.extend(decoded_rows(payload, descriptor["target"]))
        del payload
    _require(
        type(index.get("record_count")) is int
        and len(staged) == index["record_count"]
        and type(index.get("evaluated_count")) is int
        and evaluated == index["evaluated_count"],
        "Published record totals changed",
    )
    _require(
        all(digest(path) == sha for path, sha in checked.items()),
        "Published inputs changed during import",
    )
    _require(len(record_store) == 0, "Historical import destination changed")
    record_store.extend(staged)
    wire = {
        "records",
        "presets",
        "workers",
        "record_shards",
        "record_count",
        "evaluated_count",
        "method_targets",
    }
    return {
        **{key: value for key, value in index.items() if key not in wire},
        "records": record_store,
    }
