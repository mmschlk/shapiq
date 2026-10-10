"""Export accepts recorded signatures without weakening execution or integrity checks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from shapiq_benchmark.campaign import assemble_campaign
from shapiq_benchmark.runner import (
    builtin_factory,
    identity,
    load_snapshot,
    validate_method_parameters,
    validate_suite,
)

from .test_campaign_export import make_campaign, write

if TYPE_CHECKING:
    from pathlib import Path


def test_historical_parameter_exports_intact_but_execution_stays_strict(tmp_path: Path) -> None:
    parameters = {"KernelSHAP": {"historical_regularizer": 0.001}}
    make_campaign(tmp_path, method_parameters=parameters)
    directory = tmp_path / "batch-0/prepared"
    snapshot, _ = load_snapshot(directory, historical=True)
    assert snapshot["suite"]["method_parameters"] == parameters
    data = assemble_campaign(tmp_path, 3)
    assert data["suite"]["method_parameters"] == parameters
    assert data["methods"]["KernelSHAP"]["parameters"] == parameters["KernelSHAP"]
    with pytest.raises(ValueError, match="constructor parameters"):
        load_snapshot(directory)
    with pytest.raises(ValueError, match="constructor parameters"):
        validate_suite(snapshot["suite"])
    with pytest.raises(ValueError, match="constructor parameters"):
        builtin_factory("KernelSHAP", snapshot["games"][0], 0, parameters["KernelSHAP"])


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize(
    "parameters",
    [
        [],
        None,
        {1: 2},
        {"n": 11},
        {"index": "SII"},
        {"max_order": 2},
        {"random_state": 5},
        {"nested": [float("nan")]},
        {"nested": {"value": float("inf")}},
        {"value": {1, 2}},
    ],
)
def test_invalid_parameters_rejected_in_both_modes(parameters: object, *, strict: bool) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_method_parameters("KernelSHAP", parameters, check_constructor=strict)


@pytest.mark.parametrize("change", ["identity", "artifact", "reserved", "unknown_method"])
def test_historical_snapshot_keeps_integrity_and_structural_guards(
    tmp_path: Path, change: str
) -> None:
    make_campaign(tmp_path)
    directory = tmp_path / "batch-0/prepared"
    path = directory / "snapshot.json"
    snapshot = json.loads(path.read_text())
    if change == "artifact":
        (directory / "game-0.npz").write_bytes(b"changed")
    elif change == "identity":
        snapshot["snapshot_id"] = "changed"
        write(path, snapshot)
    else:
        if change == "reserved":
            snapshot["suite"]["method_parameters"] = {"KernelSHAP": {"random_state": 42}}
        else:
            snapshot["suite"]["methods"] = ["NotAnEstimator"]
        snapshot["snapshot_id"] = identity(
            {k: v for k, v in snapshot.items() if k != "snapshot_id"}
        )
        write(path, snapshot)
    with pytest.raises(ValueError):
        load_snapshot(directory, historical=True)


def test_historical_export_rejects_changed_measured_parameters(tmp_path: Path) -> None:
    make_campaign(tmp_path, method_parameters={"KernelSHAP": {"historical_regularizer": 0.001}})
    path = tmp_path / "batch-0/sweep/shard-000/results.json"
    result = json.loads(path.read_text())
    result["methods"]["KernelSHAP"]["parameters"]["historical_regularizer"] = 1.0
    result["resume_key"] = identity(
        {k: v for k, v in result.items() if k not in ("records", "campaign", "resume_key")}
    )
    write(path, result)
    with pytest.raises(ValueError, match="method source differs"):
        assemble_campaign(tmp_path, 3)
