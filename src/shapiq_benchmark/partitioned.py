"""Write lossless, bounded data blocks for the opt-in partitioned-v1 layout.

This writes data only. Browser adapters, independent publication audits and
hosting size checks must pass before enabling the layout on a website.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import shutil
from typing import TYPE_CHECKING

from shapiq_benchmark.published import HASH, _require
from shapiq_benchmark.record_store import RecordStore
from shapiq_benchmark.report import GAME_FIELDS, ROW_FIELDS, compact_workers, encode_records
from shapiq_benchmark.runner import identity
from shapiq_benchmark.summary import iter_summaries

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

KINDS = ("raw", "metrics", "games", "details", "runs", "profiles", "summaries")
FILTER_METADATA = (
    "dataset",
    "model_profile",
    "model",
    "case_id",
    "synthetic",
    "zero_truth_energy",
    "score_eligible",
    "order_scores",
    "evaluation_timing",
    "instance_seed",
    "cluster_id",
    "data_sha256",
)
METRIC_FIELDS = (*(key for key in ROW_FIELDS if key != "worker"), "run_id", "worker_id", "sequence")
MAX_BYTES = 64 * 1024 * 1024


def _json(value: dict) -> bytes:
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _chunks(rows: Iterable[dict], limit: int, max_bytes: int) -> Iterator[list[dict]]:
    """Bound both row count and uncompressed input size before column encoding."""
    chunk, size = [], 0
    for row in rows:
        length = len(_json(row))
        _require(length <= max_bytes, "A single publication row exceeds the block byte limit")
        if chunk and (len(chunk) == limit or size + length > max_bytes):
            yield chunk
            chunk, size = [], 0
        chunk.append(row)
        size += length
    if chunk:
        yield chunk


class _Blocks:
    """One output directory, with independently authenticated column blocks."""

    def __init__(self, output: Path, snapshot: str, rows: int, max_bytes: int) -> None:
        self.output, self.snapshot, self.rows, self.max_bytes = output, snapshot, rows, max_bytes
        self.assets: dict[str, list[dict]] = {kind: [] for kind in KINDS}

    def write(self, kind: str, rows: Iterable[dict], **group: str) -> None:
        for chunk in _chunks(rows, self.rows, self.max_bytes):
            self.emit(kind, chunk, group)

    def emit(self, kind: str, rows: list[dict], group: dict) -> None:
        payload = {
            "snapshot_id": self.snapshot,
            "kind": kind,
            **group,
            **encode_records(rows, nested=True),
        }
        content = _json(payload)
        if len(content) > self.max_bytes:
            _require(len(rows) > 1, "Encoded publication row exceeds the block byte limit")
            midpoint = len(rows) // 2
            self.emit(kind, rows[:midpoint], group)
            self.emit(kind, rows[midpoint:], group)
            return
        filename = f"partition-{kind}-{len(self.assets[kind])}.json"
        (self.output / filename).write_bytes(content)
        self.assets[kind].append(
            {
                "file": filename,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "count": len(rows),
                "kind": kind,
                "snapshot_id": self.snapshot,
                **group,
            }
        )


def _filter_game(game: dict, position: int, budgets: list[int]) -> dict:
    metadata = game.get("metadata", {})
    selected = {key: metadata[key] for key in FILTER_METADATA if key in metadata}
    if "game_quality" in metadata:
        selected["game_quality"] = {"role": metadata["game_quality"].get("role")}
    return {
        **{key: game[key] for key in GAME_FIELDS},
        "metadata": selected,
        "sequence": position,
        "budgets": budgets,
    }


def write_partitioned_report(
    data: dict,
    output: Path,
    *,
    block_rows: int = 32768,
    max_bytes: int = 16 * 1024 * 1024,
) -> dict:
    """Write a new data directory, committing its manifest only after all blocks.

    Rows must already be authenticated and sanitized. Each raw block reconstructs
    original records through its shared worker profile and diagnostic columns;
    ``sequence`` orders records across groups and is not a scientific field.
    Encoded byte limits do not assert a browser or Python heap limit.
    """
    _require(isinstance(data["records"], RecordStore), "Partitioned export requires a RecordStore")
    _require(
        type(block_rows) is int
        and block_rows > 0
        and type(max_bytes) is int
        and 0 < max_bytes <= MAX_BYTES,
        "Invalid block limits",
    )
    _require(
        type(data.get("schema_version")) is int
        and data["schema_version"] == 1
        and isinstance(data.get("snapshot_id"), str)
        and HASH.fullmatch(data["snapshot_id"]) is not None,
        "Invalid report identity",
    )
    _require(
        all(method.get("private") is False for method in data["methods"].values()),
        "Private methods cannot be published",
    )
    games = {game["id"]: game for game in data["games"]}
    _require(len(games) == len(data["games"]), "Repeated publication game IDs")
    for game in games.values():
        _require(not set(game) - {*GAME_FIELDS, "metadata"}, "Unsanitized publication game")
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _write(data, output, games, block_rows, max_bytes)
    except BaseException:
        shutil.rmtree(output)
        raise


def _write(data: dict, output: Path, games: dict, block_rows: int, max_bytes: int) -> dict:
    blocks = _Blocks(output, data["snapshot_id"], block_rows, max_bytes)
    store: RecordStore = data["records"]
    seen_profiles, profiles = set(), []
    profile_bytes = 0

    def profile(kind: str, value: dict) -> str:
        nonlocal profile_bytes
        key = f"{kind}-{identity(value)}"
        if key not in seen_profiles:
            row = {"id": key, "type": kind, "value": value}
            size = len(_json(row))
            _require(size <= max_bytes, "A profile exceeds the block byte limit")
            if profiles and (len(profiles) == block_rows or profile_bytes + size > max_bytes):
                flush_profiles()
            seen_profiles.add(key)
            profiles.append(row)
            profile_bytes += size
        return key

    def flush_profiles() -> None:
        nonlocal profile_bytes
        blocks.write("profiles", profiles)
        profiles.clear()
        profile_bytes = 0

    groups = {}
    for game in games.values():
        key = (f"{game['index']} · order {game['order']}", game["family"])
        groups.setdefault(key, []).append(game["id"])
    count = evaluated = 0
    capabilities: dict = {}
    for (target, family), ids in groups.items():
        for method in data["methods"]:
            group = {"target": target, "family": family, "method": method}

            def rows(ids: list[str], method: str, target: str) -> Iterator[dict]:
                nonlocal count, evaluated
                for sequence, row in store.select_indexed(game_ids=ids, methods=[method]):
                    _require(
                        not set(row) - {*ROW_FIELDS, "run_id"},
                        "Unsanitized publication record or reserved sequence",
                    )
                    _require(
                        row.get("run_id") in data["runs"]
                        and row["status"] in {"ok", "failed", "unsupported"},
                        "Unknown run or unresolved publication record",
                    )
                    count += 1
                    evaluated += row["status"] != "unsupported"
                    status = "unsupported" if row["status"] == "unsupported" else "supported"
                    capabilities.setdefault(method, {}).setdefault(status, set()).add(target)
                    yield {**row, "sequence": sequence}

            for chunk in _chunks(rows(ids, method, target), block_rows, max_bytes):
                compact = compact_workers({"records": chunk})
                worker_ids = {
                    key: profile("worker", value) for key, value in compact["workers"].items()
                }
                for row in compact["records"]:
                    if "worker_id" in row:
                        row["worker_id"] = worker_ids[row["worker_id"]]
                blocks.write("raw", compact["records"], **group)
                blocks.write(
                    "metrics",
                    (
                        {k: v for k, v in row.items() if k in METRIC_FIELDS}
                        for row in compact["records"]
                    ),
                    **group,
                )
                flush_profiles()
    _require(count == len(store), "Records include a game or method outside the public catalog")
    suite = data["suite"]
    blocks.write(
        "games",
        (
            _filter_game(
                game, i, suite.get("budgets_by_game", {}).get(game["id"], suite["budgets"])
            )
            for i, game in enumerate(data["games"])
        ),
    )
    blocks.write(
        "details", ({"type": "game", "id": game["id"], "value": game} for game in data["games"])
    )
    blocks.write(
        "details",
        (
            {"type": "report", "id": key, "value": value}
            for key, value in data.items()
            if key not in {"records", "games", "runs", "methods", "presets"}
        ),
    )

    def runs() -> Iterator[dict]:
        for key, run in data["runs"].items():
            row = {
                "id": key,
                "source_id": profile("source", {k: v for k, v in run.items() if k != "execution"}),
            }
            if "execution" in run:
                row["execution"] = run["execution"]
            yield row

    for chunk in _chunks(runs(), block_rows, max_bytes):
        blocks.write("runs", chunk)
        flush_profiles()
    degrees = sorted(
        {
            int(degree)
            for game in data["games"]
            if game["order"] > 1
            for degree in game.get("metadata", {}).get("order_scores", {})
        }
    )
    controls = any(
        game.get("metadata", {}).get("game_quality", {}).get("role") == "control"
        for game in data["games"]
    )
    for included in [False, True] if controls else [False]:
        for degree in [None, *degrees]:
            presets = iter_summaries(data, score_order=degree, include_controls=included)
            for (index, order, family), group in itertools.groupby(
                presets, key=lambda p: (p["index"], p["order"], p["family"])
            ):
                scope = {"target": f"{index} · order {order}"}
                if family is not None:
                    scope["family"] = family
                blocks.write("summaries", group, **scope)
    manifest = {
        "schema_version": 1,
        "layout": "partitioned-v1",
        "snapshot_id": data["snapshot_id"],
        "methods": copy.deepcopy(data["methods"]),
        "suite": {
            k: v
            for k, v in suite.items()
            if k
            in {
                "name",
                "protocol",
                "relative_budgets",
                "seeds",
                "game_seeds",
                "min_players",
                "min_signal_ratio",
                "methods",
                "method_parameters",
            }
        },
        "record_count": count,
        "evaluated_count": evaluated,
        "game_count": len(games),
        "assets": blocks.assets,
        "method_targets": {
            m: {s: sorted(t) for s, t in entries.items()} for m, entries in capabilities.items()
        },
    }
    content = _json(manifest)
    _require(len(content) <= MAX_BYTES, "Partition directory exceeds the manifest byte limit")
    (output / "data.json").write_bytes(content)
    return manifest
