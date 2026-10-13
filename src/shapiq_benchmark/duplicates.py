"""Exclude exact payoff-table aliases without discarding their provenance."""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np

from shapiq_benchmark.record_store import RecordStore


def payoff_fingerprint(game: dict, root: Path) -> str | None:
    """Hash the player-labelled game, excluding timing and attribution conventions."""
    if game.get("oracle", "table") != "table":
        return None  # Matching model names do not establish matching live games.
    with np.load(root / game["artifact"], allow_pickle=False) as archive:
        values = np.asarray(archive["values"], dtype="<f8").copy()
    if values.shape != (2 ** game["n_players"],) or not np.isfinite(values).all():
        message = "Duplicate detection requires a finite complete payoff table."
        raise ValueError(message)
    values[values == 0] = 0  # Signed zero has the same payoff semantics.
    header = json.dumps(["payoff-v1", game["n_players"], game["index"], game["order"]])
    return hashlib.sha256(header.encode() + values.tobytes()).hexdigest()


def claim_games(snapshot: dict, root: Path, registry_path: Path) -> dict[str, str]:
    """Atomically claim each exact game once across batches and resumable shards.

    The first registered game is canonical. All targets remain separate. The
    registry belongs to one campaign; aliases keep their original snapshot IDs.
    Call only after authenticating the snapshot and its artifacts.
    """
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprints = [(game, payoff_fingerprint(game, root)) for game in snapshot["games"]]
    with registry_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
        aliases = {}
        for game, fingerprint in fingerprints:
            if fingerprint is None:
                continue
            # Equal payoffs can have different predictor/oracle qualifications.
            # Never skip a new core recipe in favor of an earlier control run.
            role = game.get("metadata", {}).get("game_quality", {}).get("role", "unqualified")
            registry_key = f"{fingerprint}:{role}"
            canonical = registry.setdefault(
                registry_key, {"game_id": game["id"], "snapshot_id": snapshot["snapshot_id"]}
            )
            if canonical["game_id"] != game["id"]:
                aliases[game["id"]] = canonical["game_id"]
            elif canonical["snapshot_id"] != snapshot["snapshot_id"]:
                message = "Game ID reused across different duplicate-registry snapshots."
                raise ValueError(message)
        temporary = registry_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(registry, sort_keys=True) + "\n")
        temporary.replace(registry_path)
    return aliases


def remove_aliases(data: dict, aliases: dict[str, str]) -> None:
    """Remove duplicate measurements from the public panel, retaining explicit aliases."""
    data["duplicate_games"] = [
        {"game_id": key, "duplicate_of": value} for key, value in sorted(aliases.items())
    ]
    data["games"] = [game for game in data["games"] if game["id"] not in aliases]
    if isinstance(data["records"], RecordStore):
        data["records"].discard_games(aliases)
    else:
        data["records"] = [row for row in data["records"] if row["game_id"] not in aliases]
    for key in aliases:
        data["suite"].get("budgets_by_game", {}).pop(key, None)
    for entry in data.get("coverage", []):
        entry["game_ids"] = [key for key in entry.get("game_ids", []) if key not in aliases]
