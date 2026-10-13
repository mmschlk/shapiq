"""Cache unchanged summary calculations by target using read-only record workers."""

from __future__ import annotations

# Operational worker imports follow bootstrap; the adapter reopens RecordStore's
# existing partition without changing the frozen storage implementation.
# ruff: noqa: PLC0415, PLW0603, SLF001
import atexit
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_DATA = None
_RECORD_CONTEXT = None


def _digest(path: Path | str) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _write(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        stream.write(_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _tasks(data: dict) -> list[list]:
    """Match the writer's option loops and the summary generator's target order."""
    degrees = sorted(
        {
            int(k)
            for g in data["games"]
            if g["order"] > 1
            for k in g.get("metadata", {}).get("order_scores", {})
        }
    )
    controls = any(
        g.get("metadata", {}).get("game_quality", {}).get("role") == "control"
        for g in data["games"]
    )
    tasks = []
    for included in [False, True] if controls else [False]:
        for degree in [None, *degrees]:
            targets = sorted(
                {
                    (g["index"], g["order"], bool(g.get("metadata", {}).get("synthetic")))
                    for g in data["games"]
                    if (
                        included
                        or g.get("metadata", {}).get("game_quality", {}).get("role") != "control"
                    )
                    and (degree is None or g["order"] >= degree)
                }
            )
            tasks.extend(
                [index, order, synthetic, degree, included] for index, order, synthetic in targets
            )
    return tasks


def _initialize(metadata: str, database: str, partition: int) -> None:
    global _DATA, _RECORD_CONTEXT
    import pool_checkpoints

    _DATA = json.loads(Path(metadata).read_text())
    _RECORD_CONTEXT = pool_checkpoints.readonly_store(Path(database), partition)
    _DATA["records"] = _RECORD_CONTEXT.__enter__()
    atexit.register(_RECORD_CONTEXT.__exit__, None, None, None)


def _task_paths(directory: Path, task: list) -> tuple[Path, Path]:
    key = hashlib.sha256(_json(task).encode()).hexdigest()
    return directory / f"{key}.jsonl", directory / f"{key}.done.json"


def _verified(directory: Path, task: list, identity: str) -> dict | None:
    output, receipt = _task_paths(directory, task)
    if not receipt.exists():
        return None
    done = json.loads(receipt.read_text())
    if done["identity"] != identity or done["task"] != task or done["sha256"] != _digest(output):
        message = "Summary checkpoint changed"
        raise ValueError(message)
    return done


def _calculate(directory: str, task: list, identity: str) -> dict:
    from shapiq_benchmark.summary import iter_summaries

    directory = Path(directory)
    done = _verified(directory, task, identity)
    if done is not None:
        return done
    index, order, synthetic, degree, included = task
    data = {
        **_DATA,
        "games": [
            g
            for g in _DATA["games"]
            if (g["index"], g["order"], bool(g.get("metadata", {}).get("synthetic")))
            == (index, order, synthetic)
        ],
    }
    output, receipt = _task_paths(directory, task)
    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    count = 0
    with temporary.open("w") as stream:
        for row in iter_summaries(data, score_order=degree, include_controls=included):
            stream.write(_json(row) + "\n")
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    done = {"identity": identity, "task": task, "rows": count, "sha256": _digest(output)}
    _write(receipt, done)
    return done


@contextmanager
def cached_summaries(data: dict, cache_dir: Path, workers: int) -> Iterator[Path]:
    """Build/resume exact target caches, then temporarily serve the existing writer.

    Call after all record mutations and report identity changes. The owner must
    keep its RecordStore open and unchanged throughout this context. Only the
    writer's iter_summaries binding is replaced; the scientific function is not.
    """
    import pool_checkpoints

    from shapiq_benchmark import partitioned, record_store, summary

    if type(workers) is not int or workers < 1:
        message = "Need a positive worker count"
        raise ValueError(message)
    store = data["records"]
    database = store._owner._path
    database_stat = database.stat()
    if store._owner._connection.in_transaction:
        message = "Commit record writes before caching summaries"
        raise ValueError(message)
    metadata = {k: v for k, v in data.items() if k != "records"}
    encoded = _json(metadata).encode()
    pins = {
        "metadata_sha256": hashlib.sha256(encoded).hexdigest(),
        "database_sha256": _digest(database),
        "partition": store._partition,
        "code": {
            str(Path(p).resolve()): _digest(p)
            for p in [__file__, summary.__file__, record_store.__file__, pool_checkpoints.__file__]
        },
    }
    identity = hashlib.sha256(_json(pins).encode()).hexdigest()
    directory = Path(cache_dir) / identity
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        if _digest(metadata_path) != hashlib.sha256(encoded + b"\n").hexdigest():
            message = "Summary metadata checkpoint changed"
            raise ValueError(message)
    else:
        _write(metadata_path, metadata)
    tasks = _tasks(data)
    pending = [task for task in tasks if _verified(directory, task, identity) is None]
    if pending:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(pending)),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
            initargs=(str(metadata_path), str(database), store._partition),
        ) as pool:
            futures = [pool.submit(_calculate, str(directory), task, identity) for task in pending]
            for future in futures:
                future.result()
    completed = [_verified(directory, task, identity) for task in tasks]
    if any(done is None for done in completed):
        message = "Incomplete summary checkpoints"
        raise ValueError(message)
    final_stat = database.stat()
    if (database_stat.st_ino, database_stat.st_size, database_stat.st_mtime_ns) != (
        final_stat.st_ino,
        final_stat.st_size,
        final_stat.st_mtime_ns,
    ):
        message = "Record database changed while computing summaries"
        raise ValueError(message)
    _write(directory / "complete.json", {"identity": identity, "inputs": pins, "tasks": completed})

    def iterate(
        current: dict,
        *,
        bootstrap_draws: int = 200,
        score_order: int | None = None,
        include_controls: bool = False,
    ) -> Iterator[dict]:
        if current is not data or bootstrap_draws != 200:
            message = "Summary cache called with different data/options"
            raise ValueError(message)
        for task, done in zip(tasks, completed, strict=True):
            if task[3:] != [score_order, include_controls]:
                continue
            output, _ = _task_paths(directory, task)
            count = 0
            checksum = hashlib.sha256()
            with output.open("rb") as stream:
                for line in stream:
                    checksum.update(line)
                    count += 1
                    yield json.loads(line)
            if count != done["rows"] or checksum.hexdigest() != done["sha256"]:
                message = "Summary checkpoint count changed"
                raise ValueError(message)

    original = partitioned.iter_summaries
    partitioned.iter_summaries = iterate
    try:
        yield directory
    finally:
        partitioned.iter_summaries = original
