"""Private candidates reuse immutable snapshots without production-only state."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from shapiq_benchmark.duplicates import claim_games
from shapiq_benchmark.runner import digest, identity, run

if TYPE_CHECKING:
    from pathlib import Path


def snapshot(root: Path, registry: Path, *, historical: bool = False) -> Path:
    np.savez(root / "game.npz", values=np.arange(8, dtype=float))
    suite: dict = {
        "methods": ["KernelSHAP"],
        "budgets": [8],
        "seeds": [0],
        "duplicate_registry": str(registry),
    }
    if historical:
        suite["method_parameters"] = {"KernelSHAP": {"historical_regularizer": 0.001}}
    data: dict = {
        "schema_version": 1,
        "provenance": {"source_sha256": "historical-baseline-source"},
        "suite": suite,
        "games": [
            {
                "id": "tiny",
                "n_players": 3,
                "index": "SV",
                "order": 1,
                "artifact": "game.npz",
                "truth": {"coordinates": [[0], [1], [2]], "values": [1.0, 2.0, 4.0]},
            }
        ],
        "artifacts": {"game.npz": digest(root / "game.npz")},
    }
    data["snapshot_id"] = identity(data)
    path = root / "snapshot.json"
    path.write_text(json.dumps(data))
    return path


def test_isolated_candidate_ignores_inaccessible_registry_and_baseline_signature(
    tmp_path: Path,
) -> None:
    """A downloaded snapshot works off-cluster without altering its identity or files."""
    blocked = tmp_path / "inaccessible-cluster-directory"
    blocked.write_text("This regular file cannot contain a registry.")
    path = snapshot(tmp_path, blocked / "registry.json", historical=True)
    original = path.read_bytes()
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        "from shapiq.approximator import KernelSHAP\n"
        "def factory(n, index, order, seed):\n"
        "    return KernelSHAP(n=n, random_state=seed)\n"
    )
    spec = f"{candidate}:factory"
    result = run(path, tmp_path / "private", spec)
    assert result["campaign"] == {"planned": 1, "completed": 1, "complete": True}
    row = result["records"][0]
    assert row["status"] == "ok" and row["nmse"] < 1e-12
    assert 0 < row["queries"] <= 8
    assert result["snapshot_id"] == json.loads(original)["snapshot_id"]
    assert result["snapshot_provenance"] == {"source_sha256": "historical-baseline-source"}
    assert result["methods"][row["method"]]["private"]
    assert run(path, tmp_path / "private", spec, resume=True)["records"] == [row]
    assert path.read_bytes() == original
    assert digest(tmp_path / "game.npz") == json.loads(original)["artifacts"]["game.npz"]
    assert blocked.read_text() == "This regular file cannot contain a registry."
    with pytest.raises(ValueError, match="constructor parameters"):
        run(path, tmp_path / "builtin")
    (tmp_path / "game.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        run(path, tmp_path / "tampered", spec)


def test_builtin_campaign_still_claims_and_skips_production_aliases(tmp_path: Path) -> None:
    """Candidate portability must not disable existing production deduplication."""
    registry = tmp_path / "registry.json"
    path = snapshot(tmp_path, registry)
    data = json.loads(path.read_text())
    previous = {"snapshot_id": "previous", "games": [{**data["games"][0], "id": "earlier"}]}
    assert claim_games(previous, tmp_path, registry) == {}
    result = run(path, tmp_path / "builtin")
    assert len(result["records"]) == 1
    assert result["records"][0]["status"] == "duplicate"
    assert result["records"][0]["duplicate_of"] == "earlier"
    assert result["records"][0]["queries"] == 0
