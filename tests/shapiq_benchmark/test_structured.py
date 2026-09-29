"""Structured truth matches the same callable game, without exponential large tables."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from shapiq_benchmark.games import load_game, prepare_structured
from shapiq_benchmark.prepare import prepare
from shapiq_benchmark.runner import run

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def structured(tmp_path: Path) -> tuple[dict, Path]:
    """Prepare both natural feature count and training-example player definitions."""
    specs = [
        {"id": "tree-sv", "oracle": "tree", "index": "SV", "order": 1},
        {"id": "tree-pairs", "oracle": "tree", "index": "k-SII", "order": 2},
        {"id": "knn", "oracle": "knn", "index": "SV", "order": 1, "n_players": 128},
    ]
    suite = {
        "name": "test",
        "games": specs,
        "methods": ["KernelSHAP"],
        "budgets": [16],
        "seeds": [0],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    output = tmp_path / "snapshot"
    return prepare(path, output), output


def test_structured_exact_truth_and_live_reconstruction(structured: tuple[dict, Path]) -> None:
    """Small enumeration checks pass and reconstructed live games preserve endpoints."""
    snapshot, root = structured
    assert [game["n_players"] for game in snapshot["games"]] == [30, 30, 128]
    for game in snapshot["games"]:
        assert game["metadata"]["small_validation_max_error"] < 1e-10
        assert game["metadata"].get("active_players", 0) <= game["n_players"]
        with np.load(root / game["artifact"]) as artifact:
            assert "values" not in artifact.files
        oracle = load_game(game, root)
        endpoints = oracle(
            np.array([np.zeros(game["n_players"]), np.ones(game["n_players"])], dtype=bool)
        )
        assert endpoints[0] == pytest.approx(game["truth"]["baseline"])
        assert endpoints[1] - endpoints[0] == pytest.approx(sum(game["truth"]["values"]))


def test_reconstructed_model_and_version_are_checked(structured: tuple[dict, Path]) -> None:
    """A changed model or dependency cannot masquerade as the frozen oracle."""
    snapshot, root = structured
    game = copy.deepcopy(snapshot["games"][0])
    game["metadata"]["model_sha256"] = "wrong"
    with pytest.raises(ValueError, match="differs from the frozen"):
        load_game(game, root)
    game["metadata"]["sklearn_version"] = "unknown"
    with pytest.raises(ValueError, match="scikit-learn version"):
        load_game(game, root)


def test_incompatible_method_is_explicit(structured: tuple[dict, Path], tmp_path: Path) -> None:
    """SV-only methods cannot silently compete in the interaction panel."""
    _, root = structured
    result = run(root, tmp_path / "results")
    row = next(row for row in result["records"] if row["game_id"] == "tree-pairs")
    assert row["status"] == "unsupported"
    assert row["queries"] == 0
    assert row["seconds"] is None
    assert all(row["timing_scope"] != "estimator_with_table_oracle" for row in result["records"])


@pytest.mark.parametrize("ids", [["same", "same"], ["../outside"], []])
def test_invalid_game_names_rejected_before_writes(tmp_path: Path, ids: list[str]) -> None:
    """IDs cannot overwrite other games or escape the artifact directory."""
    output = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="filename-safe"):
        prepare_structured([{"id": name} for name in ids], output)
    assert not output.exists()


def test_knn_truth_rewards_the_heldout_label(structured: tuple[dict, Path]) -> None:
    """Data valuation utility rewards the true label, even when it is class zero."""
    snapshot, root = structured
    game = snapshot["games"][2]
    oracle = load_game(game, root)
    assert oracle.model.classes_[oracle.class_index] == game["metadata"]["point_label"]
    assert game["metadata"]["point_label"] == 0
    expected = oracle.model.predict_proba(oracle.x[None])[0, oracle.class_index]
    assert oracle(np.ones((1, game["n_players"]), dtype=bool))[0] == pytest.approx(expected)
