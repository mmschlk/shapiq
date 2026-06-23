"""Active storage registry.

Tracks every open DatabaseClient and assigns human-readable IDs like `local1`, `mongodb2`, `huggingface1`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leaderboard.storage.connection.client import DatabaseClient


class StorageRegistry:
    """Keeps track of all open database connections.

    IDs are assigned as ``{backend_type}{n}`` where *n* starts at 1 and
    increments per backend type independently:
        local1, local2, mongodb1, huggingface1, …
    """

    def __init__(self) -> None:
        """Initialize a new StorageRegistry."""
        self._clients: dict[str, DatabaseClient] = {}  # id -> client
        self._meta: dict[str, dict] = {}  # id -> {backend, args}
        self._counters: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def add(self, backend: str, client: DatabaseClient, args: dict) -> str:
        """Register *client* and return its assigned ID."""
        self._counters[backend] += 1
        sid = f"{backend}{self._counters[backend]}"
        self._clients[sid] = client
        self._meta[sid] = {"backend": backend, "args": args}
        return sid

    def close(self, sid: str) -> None:
        """Close *sid* and remove it from the registry."""
        client = self._clients.pop(sid, None)
        self._meta.pop(sid, None)
        if client is not None:
            client.close()

    def close_all(self) -> None:
        """Close all clients and clear the registry."""
        for sid in list(self._clients):
            self.close(sid)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, sid: str) -> DatabaseClient | None:
        """Return the DatabaseClient for *sid*, or None if not found."""
        return self._clients.get(sid)

    def ids(self) -> list[str]:
        """Return a sorted list of all active storage IDs."""
        return sorted(self._clients)

    def meta(self, sid: str) -> dict:
        """Return the metadata for *sid*, or an empty dict if not found."""
        return self._meta.get(sid, {})

    def __len__(self) -> int:
        """Return the number of active storage connections."""
        return len(self._clients)

    def __contains__(self, sid: str) -> bool:
        """Return True if *sid* is an active storage ID."""
        return sid in self._clients
