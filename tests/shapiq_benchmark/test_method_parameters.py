"""Explicit estimator variants preserve target/seed control and execution provenance."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from shapiq_benchmark import execution, runner

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("option", ["n", "index", "max_order", "random_state", "unknown", "kwargs"])
def test_target_seed_and_implicit_kwargs_cannot_be_overridden(option: str) -> None:
    """A permissive **kwargs constructor does not make undeclared settings valid."""
    suite = {
        "methods": ["KernelSHAP"],
        "budgets": [4],
        "seeds": [0],
        "method_parameters": {"KernelSHAP": {option: 1}},
    }
    with pytest.raises(ValueError, match="constructor parameters"):
        runner.validate_suite(suite)


def test_explicit_ridge_reaches_constructor_without_changing_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generic hook accepts the optional ridge constructor once its source is present."""

    class Estimator:
        def __init__(
            self, n: int, *, ridge: float = 0, random_state: int = 0, **kwargs: object
        ) -> None:
            self.arguments = n, ridge, random_state, kwargs

    monkeypatch.setitem(runner.METHODS, "OddSHAP", Estimator)
    game = {"n_players": 11, "index": "SV", "order": 1}
    configured = cast("Estimator", runner.builtin_factory("OddSHAP", game, 3, {"ridge": 0.001}))
    default = cast("Estimator", runner.builtin_factory("OddSHAP", game, 3))
    assert configured.arguments == (11, 0.001, 3, {})
    assert default.arguments == (11, 0, 3, {})


def test_suite_override_reaches_worker_and_method_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run metadata cannot confuse a configured estimator with its upstream default."""
    np.savez(tmp_path / "game.npz", values=np.array([0.0, 1.0, 2.0, 3.0]))
    suite = {
        "methods": ["KernelSHAP"],
        "budgets": [4],
        "seeds": [0],
        "method_parameters": {"KernelSHAP": {"pairing_trick": True}},
    }
    snapshot: dict = {
        "schema_version": 1,
        "provenance": {},
        "suite": suite,
        "games": [
            {
                "id": "tiny",
                "n_players": 2,
                "index": "SV",
                "order": 1,
                "artifact": "game.npz",
                "truth": {"coordinates": [[0], [1]], "values": [1.0, 2.0]},
            }
        ],
        "artifacts": {"game.npz": runner.digest(tmp_path / "game.npz")},
    }
    snapshot["snapshot_id"] = runner.identity(snapshot)
    (tmp_path / "snapshot.json").write_text(json.dumps(snapshot))
    monkeypatch.setattr(runner, "provenance", lambda: {"source_sha256": "fixture"})
    requests = []

    def isolated(request: dict, *limits: object) -> dict:
        requests.append(request)
        return {"status": "ok", "nmse": 0.0, "mse": 0.0}

    monkeypatch.setattr(execution, "isolated", isolated)
    result = runner.run(tmp_path, tmp_path / "results")
    assert requests[0]["method_parameters"] == {"pairing_trick": True}
    assert result["methods"]["KernelSHAP"]["parameters"] == {"pairing_trick": True}
    assert runner.run(tmp_path, tmp_path / "results", resume=True)["records"] == result["records"]
    assert len(requests) == 1
