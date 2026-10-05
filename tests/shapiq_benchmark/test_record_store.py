"""Disposable row storage preserves authenticated assembly and JSON semantics."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

import pytest

from shapiq_benchmark.campaign import assemble_campaign
from shapiq_benchmark.record_store import RecordStore
from shapiq_benchmark.report import write_report
from shapiq_benchmark.runner import identity
from shapiq_benchmark.summary import summarize
from tests.shapiq_benchmark.test_campaign_backend import recovery
from tests.shapiq_benchmark.test_campaign_export import make_campaign
from tests.shapiq_benchmark.test_campaign_recovery import recovery_pair
from tests.shapiq_benchmark.test_campaign_replacements import reruns

if TYPE_CHECKING:
    from pathlib import Path


def row(game: str, method: str = "a", budget: int = 11) -> dict:
    """Include distinctions SQL scalar conversion must never erase."""
    return {
        "game_id": game,
        "method": method,
        "budget": budget,
        "seed": 0,
        "nmse": -0.0,
        "status": "ok",
        "run_id": "unchanged",
        "optional": None,
        "worker": {"peak_rss_bytes": 100, "process_cpu_seconds": 1e-300},
        "order_scores": {"1": {"nmse": 1.2345678901234567}},
    }


def test_exact_rows_selection_order_and_lifetime(tmp_path: Path) -> None:
    """Filtering uses bound IDs and preserves null, signed zero and detailed fields."""
    path = tmp_path / "records.sqlite"
    rows = [row("c"), row("a", "b"), row("b"), row("a")]
    with RecordStore(path) as store:
        store.extend(rows)
        assert len(store) == 4
        assert identity({"rows": list(store)}) == identity({"rows": rows})
        assert list(store.select(game_ids=["a", "c"], methods=["a"])) == [rows[0], rows[3]]
        assert list(store.select(game_ids=[])) == []
        assert list(store.select(methods=[])) == []
        assert list(store.select(game_ids=["'); DROP TABLE records; --"])) == []
        values = list(store)
        values[0]["worker"]["peak_rss_bytes"] = 0
        assert identity({"rows": list(store)}) == identity({"rows": rows})
        fork = store.fork()
        fork.extend(store.select(game_ids=["a"]))
        assert list(fork) == [rows[1], rows[3]]
        store.discard_games(["a"])
        assert list(store) == [rows[0], rows[2]] and len(fork) == 2
    assert not path.exists()
    with pytest.raises(sqlite3.ProgrammingError):
        list(fork)


def test_atomic_append_conflicts_and_exception_cleanup(tmp_path: Path) -> None:
    """Duplicate or unserializable batches roll back fully; owners clean up on error."""
    path = tmp_path / "records.sqlite"
    with pytest.raises(RuntimeError, match="abort"), RecordStore(path) as store:
        store.extend([row("old")])
        with pytest.raises(ValueError, match="Duplicate"):
            store.extend([row("new"), row("old")])
        assert list(store) == [row("old")]
        with pytest.raises(ValueError):
            store.extend([row("new"), {**row("bad"), "nmse": float("nan")}])
        assert list(store) == [row("old")]
        with pytest.raises(ValueError, match="itself"):
            store.extend(store)
        message = "abort"
        raise RuntimeError(message)
    assert not path.exists()
    path.write_text("existing unauthenticated file")
    with pytest.raises(FileExistsError):
        RecordStore(path)
    assert path.read_text() == "existing unauthenticated file"


def test_replacement_key_coverage_and_original_order(tmp_path: Path) -> None:
    """SQL key comparisons reject both missing and unexpected corrected cells."""
    with RecordStore(tmp_path / "rows.sqlite") as original:
        rows = [row("g2"), row("g1", "b"), row("g1")]
        original.extend(rows)
        corrected = original.fork()
        corrected.extend([{**rows[2], "nmse": 0.25}])
        with pytest.raises(ValueError, match="complete public panel"):
            original.replaced(["a"], corrected)
        corrected.extend([{**rows[0], "nmse": 0.5}])
        result = original.replaced(["a"], corrected)
        assert identity({"rows": list(result)}) == identity({"rows": [rows[1], *list(corrected)]})
        assert identity({"rows": list(original)}) == identity({"rows": rows})
        corrected.extend([row("unexpected")])
        with pytest.raises(ValueError, match="complete public panel"):
            original.replaced(["a"], corrected)


@pytest.mark.parametrize("kind", ["original", "native", "correction", "supplement", "backend"])
def test_disk_and_legacy_assembly_are_identical(tmp_path: Path, kind: str) -> None:
    """Use actual authentication, cross-campaign aliases and corrected source overlays."""
    options = {}
    if kind == "backend":
        root, _retry, manifest = recovery(tmp_path)
        options["backend_supersession"] = manifest
        options["replacements"] = reruns(tmp_path, [root])
    elif kind == "supplement":
        root, retry = recovery_pair(tmp_path)
        options.update(supplements=(retry,), replacements=reruns(tmp_path, [root, retry]))
    else:
        root = tmp_path / "campaign"
        make_campaign(root, structured=kind == "native")
        if kind == "correction":
            options["replacements"] = reruns(tmp_path, [root])
    expected = assemble_campaign(root, 3, **options)
    cache = tmp_path / "cache"
    for attempt in range(2):  # Cold and warm normalization cache both preserve rows.
        with RecordStore(tmp_path / f"rows-{attempt}.sqlite") as store:
            result = assemble_campaign(root, 3, cache_dir=cache, record_store=store, **options)
            assert isinstance(result["records"], RecordStore)
            actual = {**result, "records": list(result["records"])}
            assert identity(actual) == identity(expected)
            assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def test_prepopulated_store_cannot_inject_unauthenticated_rows(tmp_path: Path) -> None:
    """Only freshly collected rows may enter an authenticated composition."""
    root = tmp_path / "campaign"
    make_campaign(root)
    with RecordStore(tmp_path / "rows.sqlite") as store:
        store.extend([row("injected")])
        with pytest.raises(ValueError, match="must be empty"):
            assemble_campaign(root, 3, record_store=store)


def test_interleaved_selections_do_not_interfere(tmp_path: Path) -> None:
    """Two active selections keep independent filters and can close in either order."""
    with RecordStore(tmp_path / "rows.sqlite") as store:
        values = [row("a"), row("b"), row("c")]
        store.extend(values)
        first = store.select(game_ids=["a", "c"])
        second = store.select(game_ids=["b", "c"])
        assert next(first) == values[0]
        assert next(second) == values[1]
        first.close()
        assert list(second) == [values[2]]
        assert list(store) == values


def test_constructor_failure_removes_only_its_new_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed connect cannot strand an unusable spill file or overwrite old data."""
    path = tmp_path / "rows.sqlite"

    def failed_connect(*_args, **_kwargs):
        message = "cannot create database"
        raise sqlite3.OperationalError(message)

    monkeypatch.setattr("shapiq_benchmark.record_store.sqlite3.connect", failed_connect)
    with pytest.raises(sqlite3.OperationalError):
        RecordStore(path)
    assert not path.exists()
    path.write_text("old")
    with pytest.raises(FileExistsError):
        RecordStore(path)
    assert path.read_text() == "old"


@pytest.mark.parametrize("operation", ["summary", "report"])
def test_legacy_writers_reject_store_before_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """A spill buffer must never silently become a full in-memory publication."""

    def forbidden_iteration(_self):
        message = "record iteration must not happen"
        raise AssertionError(message)

    monkeypatch.setattr(RecordStore, "__iter__", forbidden_iteration)
    output = tmp_path / "report"
    with (
        RecordStore(tmp_path / "rows.sqlite") as store,
        pytest.raises(TypeError, match="bounded"),
    ):
        if operation == "summary":
            summarize({"records": store})
        else:
            write_report({"records": store}, output)
    assert not output.exists()
