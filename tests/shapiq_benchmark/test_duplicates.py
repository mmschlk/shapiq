"""Exact aliases cannot consume new estimator work or headline weight."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq_benchmark.duplicates import claim_games, payoff_fingerprint, remove_aliases


def game(tmp_path, name, values, index="SV"):
    np.savez(
        tmp_path / (name + ".npz"),
        values=np.array(values, dtype=float),
        evaluation_seconds=np.arange(4),
    )
    return {"id": name, "artifact": name + ".npz", "n_players": 2, "index": index, "order": 1}


def test_cross_batch_alias_resume_and_targets(tmp_path):
    first = game(tmp_path, "first", [0, 1, 2, 3])
    second = game(tmp_path, "second", [-0.0, 1, 2, 3])
    other_target = game(tmp_path, "banzhaf", [0, 1, 2, 3], "BV")
    distinct = game(tmp_path, "permuted", [0, 2, 1, 3])
    registry = tmp_path / "registry.json"
    one = {"snapshot_id": "one", "games": [first]}
    two = {"snapshot_id": "two", "games": [second, other_target, distinct]}
    assert claim_games(one, tmp_path, registry) == {}
    expected = {"second": "first"}
    assert claim_games(two, tmp_path, registry) == expected
    assert claim_games(two, tmp_path, registry) == expected
    assert claim_games(one, tmp_path, registry) == {}
    with pytest.raises(ValueError, match="reused"):
        claim_games({**one, "snapshot_id": "changed"}, tmp_path, registry)


def test_invalid_table_and_public_removal(tmp_path):
    invalid = game(tmp_path, "invalid", [0, 1, np.nan, 3])
    with pytest.raises(ValueError, match="finite complete"):
        payoff_fingerprint(invalid, tmp_path)
    panel = {
        "games": [{"id": "a"}, {"id": "b"}],
        "records": [{"game_id": "a"}, {"game_id": "b"}],
        "suite": {"budgets_by_game": {"a": [2], "b": [2]}},
        "coverage": [{"game_ids": ["a", "b"]}],
    }
    remove_aliases(panel, {"b": "a"})
    assert panel["games"] == [{"id": "a"}]
    assert panel["records"] == [{"game_id": "a"}]
    assert panel["duplicate_games"] == [{"game_id": "b", "duplicate_of": "a"}]


def test_control_registration_cannot_skip_equivalent_core_game(tmp_path):
    """Cross-qualification copies may run once each; within-class copies still skip."""
    registry = tmp_path / "registry.json"
    for name, role, expected in [
        ("control", "control", {}),
        ("core", "core", {}),
        ("core-copy", "core", {"core-copy": "core"}),
        ("control-copy", "control", {"control-copy": "control"}),
    ]:
        item = game(tmp_path, name, [0, 1, 2, 3])
        item["metadata"] = {"game_quality": {"role": role}}
        snapshot = {"snapshot_id": name, "games": [item]}
        assert claim_games(snapshot, tmp_path, registry) == expected
