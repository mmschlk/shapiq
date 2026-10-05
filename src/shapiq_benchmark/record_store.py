"""Disposable SQLite storage for authenticated public rows during large exports.

This is a spill buffer, not a persistent cache or an authentication mechanism.
The owner must remain open while its forks or row iterators are in use.
"""

from __future__ import annotations

# Internal partitions share one owner. SQL identifiers are generated integers;
# all caller-supplied values use bound parameters.
# ruff: noqa: SLF001, S608
import itertools
import json
import os
import sqlite3
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator
    from pathlib import Path
    from types import TracebackType
    from typing import Self


class RecordStore:
    """Store exact JSON rows in insertion order with indexed, unique cell identities.

    Keep the owner context open while consuming any fork or selection. Stream
    bounded selections or ``iter_summaries``; ``summarize`` collects all presets
    into a list, while ``write_report`` still requires in-memory record lists.
    """

    def __init__(self, path: Path) -> None:
        """Create new disposable storage; never reuse an unauthenticated database."""
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.close(descriptor)
        self._path = path
        self._owner = self
        self._partition = 0
        self._numbers = itertools.count(1)
        try:
            self._connection = sqlite3.connect(path, isolation_level=None)
            self._connection.execute("PRAGMA cache_size=-8192")
            self._connection.execute("PRAGMA temp_store=FILE")
            self._connection.execute(
                "CREATE TABLE records (sequence INTEGER PRIMARY KEY, partition INTEGER NOT NULL, "
                "game_id TEXT NOT NULL, method TEXT NOT NULL, budget INTEGER NOT NULL, "
                "seed INTEGER NOT NULL, value TEXT NOT NULL, scores TEXT NOT NULL, zero_truth INTEGER NOT NULL, "
                "UNIQUE(partition, game_id, method, budget, seed))"
            )
            self._connection.execute("CREATE INDEX record_order ON records(partition, sequence)")
            self._connection.execute(
                "CREATE INDEX zero_games ON records(partition,zero_truth,game_id)"
            )
            self._connection.execute(
                "CREATE TEMP TABLE panel_cells (selection INTEGER, position INTEGER, "
                "game_id TEXT, budget INTEGER, seed INTEGER, PRIMARY KEY(selection,position))"
            )
            self._connection.execute(
                "CREATE TEMP TABLE selections (selection INTEGER, game_id TEXT, "
                "PRIMARY KEY(selection, game_id))"
            )
        except BaseException:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            path.unlink(missing_ok=True)
            raise

    def __enter__(self) -> Self:
        """Keep the owner alive throughout authenticated assembly and publication."""
        return self

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Release disposable storage on success or failure."""
        self.close()

    def close(self) -> None:
        """Close all forks and remove the disposable database, including on failure."""
        self._owner._connection.close()
        self._owner._path.unlink(missing_ok=True)

    def fork(self) -> RecordStore:
        """Create an empty partition sharing the owner's connection and lifetime."""
        child = object.__new__(type(self))
        child._owner = self._owner
        child._connection = self._owner._connection
        child._partition = next(self._owner._numbers)
        return child

    def __len__(self) -> int:
        """Count rows without materializing them."""
        return self._connection.execute(
            "SELECT count(*) FROM records WHERE partition=?", (self._partition,)
        ).fetchone()[0]

    def __iter__(self) -> Iterator[dict]:
        """Stream all rows in original order."""
        return self.select()

    def extend(self, rows: Iterable[dict]) -> None:
        """Append a batch atomically, rejecting duplicate result cells."""
        if rows is self:
            msg = "Cannot append a record store to itself"
            raise ValueError(msg)
        try:
            with self._transaction():
                self._connection.executemany(
                    "INSERT INTO records(partition,game_id,method,budget,seed,value,scores,zero_truth) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        (
                            self._partition,
                            row["game_id"],
                            row["method"],
                            row["budget"],
                            row["seed"],
                            json.dumps(row, allow_nan=False, separators=(",", ":")),
                            json.dumps(
                                {
                                    key: row[key]
                                    for key in (
                                        "game_id",
                                        "status",
                                        "nmse",
                                        "order_scores",
                                        "zero_truth_energy",
                                    )
                                    if key in row
                                },
                                allow_nan=False,
                                separators=(",", ":"),
                            ),
                            bool(row.get("zero_truth_energy")),
                        )
                        for row in rows
                    ),
                )
        except sqlite3.IntegrityError as error:
            msg = "Duplicate or invalid result cell in record store"
            raise ValueError(msg) from error

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Nest iterator setup/cleanup safely inside atomic appends."""
        name = f"operation_{next(self._owner._numbers)}"
        self._connection.execute(f"SAVEPOINT {name}")
        try:
            yield
        except BaseException:
            self._connection.execute(f"ROLLBACK TO {name}")
            raise
        finally:
            self._connection.execute(f"RELEASE {name}")

    @contextmanager
    def _game_filter(self, game_ids: Iterable[str] | None) -> Iterator[tuple[str, list]]:
        """Keep independent indexed filters without DDL while cursors are active."""
        if game_ids is None:
            yield "", []
            return
        selection = next(self._owner._numbers)
        try:
            with self._transaction():
                self._connection.executemany(
                    "INSERT OR IGNORE INTO selections VALUES (?,?)",
                    ((selection, game_id) for game_id in game_ids),
                )
            yield " AND game_id IN (SELECT game_id FROM selections WHERE selection=?)", [selection]
        finally:
            self._connection.execute("DELETE FROM selections WHERE selection=?", (selection,))

    def select(
        self, *, game_ids: Iterable[str] | None = None, methods: Iterable[str] | None = None
    ) -> Generator[dict, None, None]:
        """Stream an indexed subset in its original order, without mutating rows."""
        with self._game_filter(game_ids) as (game_filter, filter_parameters):
            parameters: list = [self._partition, *filter_parameters]
            method_filter = ""
            if methods is not None:
                names = list(methods)
                if not names:
                    return
                method_filter = f" AND method IN ({','.join('?' for _ in names)})"
                parameters.extend(names)
            cursor = self._connection.execute(
                "SELECT value FROM records WHERE partition=?"
                + game_filter
                + method_filter
                + " ORDER BY sequence",
                parameters,
            )
            try:
                for (value,) in cursor:
                    yield json.loads(value)
            finally:
                cursor.close()

    def zero_games(self) -> set[str]:
        """Preserve global row-derived truth exclusions, including filtered-out rows."""
        return {
            row[0]
            for row in self._connection.execute(
                "SELECT DISTINCT game_id FROM records WHERE partition=? AND zero_truth=1",
                (self._partition,),
            )
        }

    def measurements(
        self, cells: Iterable[tuple[str, int, int]], methods: Iterable[str]
    ) -> Generator[tuple[str, list[dict | None]], None, None]:
        """Yield exact compact measurements in requested cell order, one method at a time.

        A single indexed selection is reused across methods. Missing cells remain
        explicit ``None``; scores stay JSON text rather than SQLite floating point.
        """
        selection = next(self._owner._numbers)
        try:
            with self._transaction():
                self._connection.executemany(
                    "INSERT INTO panel_cells VALUES (?,?,?,?,?)",
                    (
                        (selection, position, game, budget, seed)
                        for position, (game, budget, seed) in enumerate(cells)
                    ),
                )
            for method in methods:
                cursor = self._connection.execute(
                    "SELECT r.scores FROM panel_cells c LEFT JOIN records r "
                    "ON r.partition=? AND r.game_id=c.game_id AND r.method=? "
                    "AND r.budget=c.budget AND r.seed=c.seed "
                    "WHERE c.selection=? ORDER BY c.position",
                    (self._partition, method, selection),
                )
                try:
                    measured = [
                        json.loads(value) if value is not None else None for (value,) in cursor
                    ]
                finally:
                    cursor.close()
                yield method, measured
        finally:
            self._connection.execute("DELETE FROM panel_cells WHERE selection=?", (selection,))

    def discard_games(self, game_ids: Iterable[str]) -> None:
        """Remove global aliases only after their complete union was authenticated."""
        with self._transaction():
            self._connection.executemany(
                "DELETE FROM records WHERE partition=? AND game_id=?",
                ((self._partition, game_id) for game_id in game_ids),
            )

    def replaced(self, methods: Iterable[str], replacements: RecordStore) -> RecordStore:
        """Return unchanged rows followed by an exact complete corrected-method panel."""
        if replacements._owner is not self._owner:
            msg = "Replacement records must share the assembly database"
            raise ValueError(msg)
        names = list(methods)
        placeholders = ",".join("?" for _ in names)
        baseline = (
            "SELECT game_id,method,budget,seed FROM records "
            f"WHERE partition=? AND method IN ({placeholders})"
        )
        corrected = "SELECT game_id,method,budget,seed FROM records WHERE partition=?"
        for query, parameters in (
            (baseline + " EXCEPT " + corrected, [self._partition, *names, replacements._partition]),
            (corrected + " EXCEPT " + baseline, [replacements._partition, self._partition, *names]),
        ):
            if self._connection.execute(query + " LIMIT 1", parameters).fetchone() is not None:
                msg = "Replacement does not cover the complete public panel"
                raise ValueError(msg)
        result = self.fork()
        with self._transaction():
            self._connection.execute(
                "INSERT INTO records(partition,game_id,method,budget,seed,value,scores,zero_truth) "
                "SELECT ?,game_id,method,budget,seed,value,scores,zero_truth FROM records "
                f"WHERE partition=? AND method NOT IN ({placeholders}) ORDER BY sequence",
                [result._partition, self._partition, *names],
            )
        result.extend(replacements)
        return result
