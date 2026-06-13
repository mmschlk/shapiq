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
from .utilities import _matches_config, _matches_config_with_seed


def _json_default(value: object) -> object:
    """Convert non-standard numeric values to JSON-compatible types."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


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
        return self._path.exists()

    def close(self) -> None:
        """No-op: file handles are opened and closed per operation."""

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert_one(self, document: dict[str, Any]) -> None:
        """Append a single run document to the JSONL file."""
        self._append(document)

    def safe_insert_one(self, document: dict[str, Any], mode: str = "merge") -> bool:
        """Append a single run document only if no existing document matches its config and seed.

        Args:
            document: the run document to insert
            mode: merge / replace / skip (default: merge)
                - merge: if a matching document exists, override metrics that are different and keep the rest; update timestamp to newest document
                - replace: if a matching document exists, replace it entirely with the new document
                - skip: if a matching document exists, do not modify it

        Returns True if the document was inserted, or False if a matching document already exists.
        """
        # Extract config from the new document
        new_doc_config = RunConfig.from_dict(document)

        # Load documents by config
        existing_docs = self.get_by_config(new_doc_config)

        if not existing_docs:
            # No existing document matches the config, safe to insert
            self.insert_one(document)
            return True

        # Check for matching seed
        for existing_doc in existing_docs:
            if _matches_config_with_seed(document, existing_doc):
                if mode == "merge":
                    # Merge metrics and update timestamp
                    merged_doc = existing_doc.copy()
                    merged_doc.update(document)  # New document's fields override existing ones

                    # delete only if duplicate
                    self.delete_by_id(
                        existing_doc.get("run_id")
                    )  # Remove old document(s) by unique identifier

                    self.insert_one(merged_doc)  # Insert merged document
                elif mode == "replace":
                    self.delete_by_config(new_doc_config)  # Remove old document(s)
                    self.insert_one(document)  # Insert new document
                elif mode == "skip":
                    return False  # Do not insert, as a matching document already exists

        return True

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

    def delete_by_id(self, doc_id: str) -> int:
        """Delete a document by its unique identifier. Returns 1 if deleted, 0 if not found."""
        documents = self._load()
        kept = [d for d in documents if d.get("run_id") != doc_id]
        deleted = len(documents) - len(kept)
        if deleted:
            self._save(kept)
        return deleted

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
