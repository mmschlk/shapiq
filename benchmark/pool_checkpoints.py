"""Durable, authenticated collection shards; operational only, never estimator execution.

Receipts appear last, after SQLite is closed. Exporters use an isolated working copy:
RecordStore forks used by alias removal must never mutate the reusable checkpoint.
"""

from __future__ import annotations

# This operational adapter deliberately reuses the frozen RecordStore schema.
# ruff: noqa: SLF001
import copy
import itertools
import json
import os
import shutil
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import collect_pool as collector
from collect_pool import Inputs, digest, require

from shapiq_benchmark.record_store import RecordStore
from shapiq_benchmark.runner import identity


class _DurableStore(RecordStore):
    def close(self) -> None:
        """Keep completed SQLite bytes even if later metadata/export work fails."""
        self._owner._connection.close()


def _write(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _code_pins() -> dict[str, str]:
    return {str(Path(p).absolute()): digest(p) for p in (__file__, collector.__file__)}


def _publish(
    directory: Path,
    config_path: Path,
    data: dict,
    audit: dict,
    cases: list[int],
    inventory_size: int,
) -> dict:
    """Finish a checkpoint only after its database and metadata are durable."""
    data = {k: v for k, v in data.items() if k != "records"}
    audit["input_hashes"].update(_code_pins())
    audit["input_hashes"][str(config_path)] = digest(config_path)
    _write(directory / "data.json", data)
    _write(directory / "audit.json", audit)
    with (directory / "records.sqlite").open("rb") as stream:
        os.fsync(stream.fileno())
    receipt = {
        "format": "pool-collection-checkpoint-v1",
        "config_path": str(config_path),
        "config_sha256": digest(config_path),
        "code": _code_pins(),
        "cases": cases,
        "inventory_size": inventory_size,
        "rows": audit["rows"],
        "files": {
            name: digest(directory / name) for name in ("data.json", "audit.json", "records.sqlite")
        },
    }
    _write(directory / "checkpoint.json", receipt)
    return receipt


def create_shard(config_path: Path, directory: Path, case_ids: list[int]) -> dict:
    """Collect one contiguous range, preserving every ordinary ownership check."""
    config_path, directory = Path(config_path).absolute(), Path(directory).absolute()
    require(
        case_ids and case_ids == list(range(case_ids[0], case_ids[-1] + 1)),
        "Shard cases must be a nonempty contiguous ordered range",
    )
    if directory.exists():
        return _reuse(directory, config_path, case_ids)
    config_sha, code = digest(config_path), _code_pins()
    config = json.loads(config_path.read_text())
    directory.mkdir(parents=True, exist_ok=False)
    with _DurableStore(directory / "records.sqlite") as store:
        data, audit = collector.collect(config, store, case_ids=case_ids)
    require(digest(config_path) == config_sha and _code_pins() == code, "Collection inputs changed")
    return _publish(directory, config_path, data, audit, case_ids, audit["inventory_size"])


def _read(directory: Path) -> tuple[dict, dict, dict]:
    directory = Path(directory).absolute()
    receipt = json.loads((directory / "checkpoint.json").read_text())
    require(receipt["format"] == "pool-collection-checkpoint-v1", "Unknown checkpoint")
    require(receipt["code"] == _code_pins(), "Checkpoint collection code differs")
    require(
        set(receipt["files"]) == {"data.json", "audit.json", "records.sqlite"},
        "Incomplete checkpoint files",
    )
    for name, expected in receipt["files"].items():
        require(digest(directory / name) == expected, "Checkpoint bytes changed: " + name)
    for suffix in ("-wal", "-journal"):
        require(
            not Path(str(directory / "records.sqlite") + suffix).exists(),
            "Checkpoint has an uncommitted SQLite companion",
        )
    data = json.loads((directory / "data.json").read_text())
    audit = json.loads((directory / "audit.json").read_text())
    require(data["snapshot_id"] == audit["report_id"], "Checkpoint report identity differs")
    cases = receipt["cases"]
    require(cases and cases == list(range(cases[0], cases[-1] + 1)), "Invalid checkpoint cases")
    require([case["case"] for case in audit["cases"]] == cases, "Checkpoint case ownership differs")
    require(audit["rows"] == receipt["rows"], "Checkpoint row count differs")
    require(
        audit["input_hashes"][receipt["config_path"]] == receipt["config_sha256"],
        "Checkpoint config provenance differs",
    )
    return receipt, data, audit


def _reuse(directory: Path, config_path: Path, cases: list[int] | None) -> dict:
    """Resume only a completed authenticated checkpoint; partial work stays preserved."""
    require(
        (directory / "checkpoint.json").is_file(),
        "Incomplete checkpoint directory; preserve it and use a new attempt directory",
    )
    receipt, _, audit = _read(directory)
    expected = list(range(receipt["inventory_size"])) if cases is None else cases
    require(
        receipt["cases"] == expected
        and receipt["config_path"] == str(config_path)
        and receipt["config_sha256"] == digest(config_path),
        "Reusable checkpoint selection/config differs",
    )
    config = json.loads(config_path.read_text())
    _stable(config, audit, receipt["inventory_size"])
    return receipt


def _stable(config: dict, audit: dict, inventory_size: int) -> None:
    """One unioned raw-input hash pass, not one repeated pass per shard."""
    require(
        collector.terminal_accounting(config["jobs"]) == audit["slurm_accounting"],
        "Allocation state differs from collected checkpoint",
    )
    inputs = Inputs()
    inputs.hashes.update(audit["input_hashes"])
    inputs.absent.update(audit["absent_files"])
    inputs.stable()
    root = Path(config["pool_directory"])
    require(
        {p.name for p in (root / "allocations").iterdir()} == set(config["jobs"]),
        "Allocation directory inventory changed",
    )
    require(
        {p.name for p in (root / "tasks").iterdir()}
        <= {f"case-{i:06d}" for i in range(inventory_size)},
        "Foreign case directory",
    )
    for case in audit["cases"]:
        if case["status"] in {"unclaimed", "interrupted_claim"}:
            collector.unstarted_case(
                root / "tasks" / f"case-{case['case']:06d}",
                {"claim.json"} if "job" in case else set(),
                inputs,
            )


def _same_map(target: dict, additions: dict, label: str) -> None:
    for key, value in additions.items():
        require(key not in target or target[key] == value, "Conflicting " + label)
        target[key] = value


def merge_shards(config_path: Path, shard_dirs: list[Path], directory: Path) -> dict:
    """Authenticate complete case coverage and merge SQL without decoding result rows."""
    config_path, directory = Path(config_path).absolute(), Path(directory).absolute()
    if directory.exists():
        return _reuse(directory, config_path, None)
    config_sha = digest(config_path)
    config = json.loads(config_path.read_text())
    parts = [(Path(path).absolute(), *_read(path)) for path in shard_dirs]
    require(parts, "No collection shards")
    parts.sort(key=lambda part: part[1]["cases"][0])
    size = parts[0][1]["inventory_size"]
    cases = [case for _, receipt, _, _ in parts for case in receipt["cases"]]
    require(cases == list(range(size)), "Shard cases overlap or omit planned cases")
    first = parts[0][2]
    data = copy.deepcopy(first)
    for key in ("games", "coverage", "preparation_coverage"):
        data[key] = []
    for key in ("methods", "runs"):
        data[key] = {}
    data["suite"]["budgets_by_game"] = {}
    audit = copy.deepcopy(parts[0][3])
    audit.update(cases=[], input_hashes={}, absent_files=[], statuses={})
    counts, absent, seen_games = Counter(), set(), set()
    base_suite = {
        k: v for k, v in first["suite"].items() if k not in {"budgets", "budgets_by_game"}
    }
    for _, receipt, item, checked in parts:
        require(
            receipt["config_path"] == str(config_path)
            and receipt["config_sha256"] == config_sha
            and receipt["inventory_size"] == size,
            "Shard configuration differs",
        )
        require(
            item["snapshot_provenance"] == first["snapshot_provenance"]
            and item["schema_version"] == first["schema_version"]
            and item["coverage_methods"] == first["coverage_methods"]
            and {k: v for k, v in item["suite"].items() if k not in {"budgets", "budgets_by_game"}}
            == base_suite
            and checked["slurm_accounting"] == audit["slurm_accounting"],
            "Shard scientific/accounting metadata differs",
        )
        ids = [game["id"] for game in item["games"]]
        require(
            len(ids) == len(set(ids)) and not seen_games.intersection(ids),
            "Repeated game across collection shards",
        )
        seen_games.update(ids)
        for key in ("games", "coverage", "preparation_coverage"):
            data[key].extend(item[key])
        for key in ("methods", "runs"):
            _same_map(data[key], item[key], key)
        _same_map(data["suite"]["budgets_by_game"], item["suite"]["budgets_by_game"], "budget grid")
        _same_map(audit["input_hashes"], checked["input_hashes"], "input pin")
        absent.update(checked["absent_files"])
        audit["cases"].extend(checked["cases"])
        counts.update(checked["statuses"])
    require(not absent.intersection(audit["input_hashes"]), "Input both present and absent")
    audit["absent_files"] = sorted(absent)
    _stable(config, audit, size)
    checkpoint_pins = {}
    for path, receipt, _, _ in parts:
        checkpoint_pins[str(path / "checkpoint.json")] = digest(path / "checkpoint.json")
        checkpoint_pins.update({str(path / name): sha for name, sha in receipt["files"].items()})
    _same_map(audit["input_hashes"], checkpoint_pins, "source checkpoint pin")
    directory.mkdir(parents=True, exist_ok=False)
    with _DurableStore(directory / "records.sqlite") as store:
        for path, receipt, _, _ in parts:
            connection = store._connection
            connection.execute("ATTACH DATABASE ? AS shard", (str(path / "records.sqlite"),))
            try:
                row_count = connection.execute("SELECT count(*) FROM shard.records").fetchone()[0]
                require(row_count == receipt["rows"], "Stored shard row count differs")
                require(
                    connection.execute(
                        "SELECT count(*) FROM shard.records WHERE partition!=0"
                    ).fetchone()[0]
                    == 0,
                    "Shard contains unexpected SQL partitions",
                )
                # Original global case order, then exact encounter order within each shard.
                with store._transaction():
                    connection.execute(
                        "INSERT INTO records(partition,game_id,method,budget,seed,value,scores,zero_truth) "
                        "SELECT partition,game_id,method,budget,seed,value,scores,zero_truth "
                        "FROM shard.records ORDER BY sequence"
                    )
            finally:
                connection.execute("DETACH DATABASE shard")
        audit["rows"] = len(store)
    require(audit["rows"] == sum(part[1]["rows"] for part in parts), "Merged row count differs")
    require(
        all(digest(path) == sha for path, sha in checkpoint_pins.items()),
        "Source checkpoint changed during SQL merge",
    )
    data["suite"]["budgets"] = sorted(
        {b for grid in data["suite"]["budgets_by_game"].values() for b in grid}
    )
    data["snapshot_id"] = identity(
        {
            "pool_suite": config["suite"]["sha256"],
            "source": data["snapshot_provenance"],
            "inputs": audit["input_hashes"],
            "absent_files": audit["absent_files"],
            "runs": sorted(data["runs"]),
        }
    )
    complete = all(case["complete"] for case in audit["cases"])
    audit.update(
        status="PASS" if complete else "INCOMPLETE",
        complete=complete,
        outcome_complete=complete,
        accounting_complete=all(c["accounting_complete"] for c in audit["cases"]),
        score_complete=all(c.get("score_complete", False) for c in audit["cases"]),
        report_id=data["snapshot_id"],
        collection_cases=cases,
        inventory_size=size,
        intended_instances=size,
        prepared_targets=len(seen_games),
        collected_targets=len(data["games"]),
        case_statuses=dict(Counter(c["status"] for c in audit["cases"])),
        statuses=dict(counts),
    )
    require(digest(config_path) == config_sha, "Merge config changed")
    return _publish(directory, config_path, data, audit, cases, size)


def readonly_store(path: Path, partition: int = 0) -> RecordStore:
    """Open immutable public rows; TEMP query tables remain connection-local."""
    path = Path(path).absolute()
    store = object.__new__(_DurableStore)
    store._path, store._owner, store._partition = path, store, partition
    store._numbers = itertools.count(1)
    store._connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, isolation_level=None)
    store._connection.execute("PRAGMA cache_size=-8192")
    store._connection.execute("PRAGMA temp_store=FILE")
    store._connection.execute(
        "CREATE TEMP TABLE panel_cells (selection INTEGER, position INTEGER, game_id TEXT, budget INTEGER, seed INTEGER, PRIMARY KEY(selection,position))"
    )
    store._connection.execute(
        "CREATE TEMP TABLE selections (selection INTEGER, game_id TEXT, PRIMARY KEY(selection,game_id))"
    )
    return store


def open_checkpoint(
    directory: Path, database: Path | None = None
) -> tuple[RecordStore, dict, dict]:
    """Return (store,data,audit) with a disposable working copy, never the saved DB."""
    directory = Path(directory).absolute()
    receipt, data, audit = _read(directory)
    require(
        receipt["cases"] == list(range(receipt["inventory_size"])),
        "Final export requires the complete case inventory",
    )
    config = json.loads(Path(receipt["config_path"]).read_text())
    _stable(config, audit, receipt["inventory_size"])
    if database is None:
        fd, name = tempfile.mkstemp(prefix="shapiq-export-", suffix=".sqlite")
        os.close(fd)
        database = Path(name)
    else:
        database = Path(database)
        with database.open("xb"):
            pass
    try:
        shutil.copyfile(directory / "records.sqlite", database)
        require(digest(database) == receipt["files"]["records.sqlite"], "Working DB copy differs")
        store = object.__new__(RecordStore)
        store._path, store._owner, store._partition = database, store, 0
        store._numbers = itertools.count(1)
        store._connection = sqlite3.connect(database, isolation_level=None)
        store._connection.execute("PRAGMA cache_size=-8192")
        store._connection.execute("PRAGMA temp_store=FILE")
        store._connection.execute(
            "CREATE TEMP TABLE panel_cells (selection INTEGER, position INTEGER, game_id TEXT, budget INTEGER, seed INTEGER, PRIMARY KEY(selection,position))"
        )
        store._connection.execute(
            "CREATE TEMP TABLE selections (selection INTEGER, game_id TEXT, PRIMARY KEY(selection,game_id))"
        )
        data["records"] = store
        audit["input_hashes"].update(
            {str(directory / name): sha for name, sha in receipt["files"].items()}
        )
        audit["input_hashes"][str(directory / "checkpoint.json")] = digest(
            directory / "checkpoint.json"
        )
        require(len(store) == receipt["rows"], "Reopened checkpoint row count differs")
    except BaseException:
        if "store" in locals() and hasattr(store, "_connection"):
            store._connection.close()
        database.unlink(missing_ok=True)
        raise
    return store, data, audit
