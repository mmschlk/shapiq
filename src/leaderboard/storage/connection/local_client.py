"""This module defines the LocalClient class.

Provides functionality for:
- Storing run records as JSONL files on the local filesystem.
- Inserting, querying, and deleting run records without any network dependency.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Self

import numpy as np
from dotenv import load_dotenv

from leaderboard.storage.data_classes import RunConfig

from .client import DatabaseClient


def _json_default(value: object) -> object:
    """Convert non-standard numeric values to JSON-compatible types."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _matches_config(document: dict[str, Any], config: RunConfig) -> bool:
    """Return True if *document* contains all key/value pairs in *config*."""

    return (
        document.get("game_name") == config.game_name and
        document.get("n_players") == config.n_players and
        document.get("approximator_name") == config.approximator_name and
        document.get("index") == config.index and
        document.get("max_order") == config.max_order and
        document.get("budget") == config.budget and
        document.get("ground_truth_method") == config.ground_truth_method
    )


class LocalClient(DatabaseClient):
    """File-system database client that stores run records as JSONL.

    Each document is written as a single JSON line. All read operations
    load the file into memory and filter in Python — appropriate for the
    typical leaderboard dataset sizes.

    Parameters
    ----------
    path:
        Path to the ``.jsonl`` file used as the backing store.
        Parent directories are created automatically on first write.
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize the LocalClient with the given file path.

        Parameters
        ----------
        path:
            Path to the JSONL file used as the backing store. Parent directories are created automatically on first write.
        """
        self._path = Path(path)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, args: dict) -> Self:
        """Create a LocalClient from the ``LOCAL_DB_PATH`` environment variable.

        Falls back to ``"data/runs.jsonl"`` when the variable is unset.
        """
        # load env
        load_dotenv()

        path = (
            args["LOCAL_DB_PATH"]
            if "LOCAL_DB_PATH" in args
            else os.getenv("LOCAL_DB_PATH", "data/runs.jsonl")
        )

        return cls(path=path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> list[dict[str, Any]]:
        """Load and return all documents from the JSONL file."""
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def _save(self, documents: list[dict[str, Any]]) -> None:
        """Overwrite the JSONL file with *documents*."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as fh:
            for doc in documents:
                fh.write(json.dumps(doc, default=_json_default))
                fh.write("\n")

    def _append(self, document: dict[str, Any]) -> None:
        """Append a single document to the JSONL file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(document, default=_json_default))
            fh.write("\n")

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """If the path exists, return True; otherwise, return False."""

        return True if self._path.exists() else False

    def close(self) -> None:
        """No-op: file handles are opened and closed per operation."""

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert_one(self, document: dict[str, Any]) -> None:
        """Append a single run document to the JSONL file."""
        self._append(document)

    def insert_many(self, documents: list[dict[str, Any]]) -> None:
        """Append multiple run documents (no-op for an empty list)."""
        if not documents:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            for doc in documents:
                fh.write(json.dumps(doc, default=_json_default))
                fh.write("\n")

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_all(self) -> int:
        """Delete every document. Returns the number of deleted documents."""
        documents = self._load()
        count = len(documents)
        self._save([])
        return count

    def delete_by_config(self, config: RunConfig) -> int:
        """Delete all documents matching *config*. Returns deleted count."""
        documents = self._load()
        kept = [d for d in documents if not _matches_config(d, config)]
        deleted = len(documents) - len(kept)
        if deleted:
            self._save(kept)
        return deleted

    # ------------------------------------------------------------------
    # Read — generic
    # ------------------------------------------------------------------

    def get_all(self) -> list[dict[str, Any]]:
        """Return every document."""
        return self._load()

    def get_by_config(self, config: RunConfig) -> list[dict[str, Any]]:
        """Return all documents whose fields match *config*."""
        return [d for d in self._load() if _matches_config(d, config)]

    # ------------------------------------------------------------------
    # Read — domain helpers
    # ------------------------------------------------------------------

    def get_unique_configs(self) -> list[RunConfig]:
        """Return one ``RunConfig`` per unique configuration in the store."""
        seen: set[tuple[Any, ...]] = set()
        result: list[RunConfig] = []
        for doc in self._load():
            config = RunConfig.from_dict(doc)
            key = tuple(sorted(config.to_dict().items()))
            if key not in seen:
                seen.add(key)
                result.append(config)
        return result

    def get_games(self) -> list[str]:
        """Return sorted distinct game names."""
        return sorted({d["game_name"] for d in self._load() if "game_name" in d})

    def get_by_game(self, game_name: str) -> list[dict[str, Any]]:
        """Return all runs for a given game name."""
        return [d for d in self._load() if d.get("game_name") == game_name]

    def get_approximators(self) -> list[str]:
        """Return sorted distinct approximator names."""
        return sorted({d["approximator_name"] for d in self._load() if "approximator_name" in d})

    def get_by_approximator(self, approximator_name: str) -> list[dict[str, Any]]:
        """Return all runs that used a given approximator."""
        return [d for d in self._load() if d.get("approximator_name") == approximator_name]

    def count_by_config(self, config: RunConfig) -> int:
        """Return the number of runs stored for *config*."""
        return sum(1 for d in self._load() if _matches_config(d, config))
