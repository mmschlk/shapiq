"""Checks for safe, provenance-preserving static report construction."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

import pytest

from shapiq_benchmark.report import merge_results, report

if TYPE_CHECKING:
    from pathlib import Path


def result_fixture() -> dict:
    """A complete small result with deliberately private raw fields."""
    return {
        "schema_version": 1,
        "snapshot_id": "frozen",
        "snapshot_provenance": {"version": "1"},
        "run_provenance": {"version": "1"},
        "suite": {"name": "pilot", "methods": ["baseline"], "budgets": [8], "seeds": [0]},
        "games": [
            {
                "id": "game",
                "family": "tree",
                "stratum": "tiny",
                "n_players": 3,
                "index": "SV",
                "order": 1,
                "artifact": "/private/model.npz",
                "truth": {"values": [123.4]},
                "metadata": {"dataset": "test", "path": "/private/data"},
            }
        ],
        "methods": {"baseline": {"source_sha256": "abc", "private": False}},
        "records": [
            {
                "game_id": "game",
                "method": "baseline",
                "budget": 8,
                "seed": 0,
                "status": "ok",
                "nmse": 0.1,
                "mse": 0.2,
                "seconds": 0.01,
                "queries": 8,
                "estimate": {"values": [43.2]},
            }
        ],
    }


def write(tmp_path: Path, data: dict, name: str = "results.json") -> Path:
    """Write a result fixture."""
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_export_strips_private_artifacts(tmp_path: Path) -> None:
    """Keep scientific metadata while excluding models, truth, and raw estimates."""
    path = write(tmp_path, result_fixture())
    output = tmp_path / "site"
    data = report([path, path], output, public=True)
    assert len(data["records"]) == 1
    text = json.dumps(data)
    assert "/private" not in text
    assert "123.4" not in text
    assert "43.2" not in text
    assert data["games"][0]["metadata"] == {"dataset": "test", "zero_truth_energy": False}
    assert (output / "index.html").exists()
    assert (output / "charts.js").exists()
    assert data["records"][0]["run_id"] in data["runs"]


def test_estimated_runtime_export(tmp_path: Path) -> None:
    """Estimated oracle costs survive export without replacing measured elapsed time."""
    data = result_fixture()
    timing = {
        "cache_lookup_seconds": 0.001,
        "estimated_oracle_seconds": 10.0,
        "estimated_uncached_seconds": 10.009,
    }
    data["records"][0].update(timing)
    data["games"][0]["metadata"]["oracle_cost_protocol"] = "batch-amortized-wall-seconds-v1"
    exported = merge_results([write(tmp_path, data)])
    assert exported["records"][0]["seconds"] == 0.01
    assert all(exported["records"][0][key] == value for key, value in timing.items())
    assert (
        exported["games"][0]["metadata"]["oracle_cost_protocol"]
        == "batch-amortized-wall-seconds-v1"
    )
    data["records"][0]["estimated_uncached_seconds"] = -1
    with pytest.raises(ValueError, match="Invalid numeric"):
        merge_results([write(tmp_path, data)])


def test_frozen_signal_policy_export(tmp_path: Path) -> None:
    """Publish a frozen exclusion policy without rewriting the measured error."""
    data = result_fixture()
    data["suite"]["min_signal_ratio"] = 1e-6
    fields = {
        "score_eligible": False,
        "signal_ratio": 1e-10,
        "score_exclusion_reason": "low_signal",
    }
    data["games"][0]["metadata"].update(fields)
    exported = merge_results([write(tmp_path, data)])
    assert exported["suite"]["min_signal_ratio"] == 1e-6
    assert all(exported["games"][0]["metadata"][key] == value for key, value in fields.items())
    assert exported["records"][0]["nmse"] == 0.1
    assert exported["records"][0]["status"] == "ok"


@pytest.mark.parametrize("change", ["snapshot", "suite", "game", "method", "cell", "provenance"])
def test_conflicting_panels_rejected(tmp_path: Path, change: str) -> None:
    """Sharing a filename or label must not silently join incompatible measurements."""
    first = result_fixture()
    second = copy.deepcopy(first)
    if change == "snapshot":
        second["snapshot_id"] = "another"
    elif change == "suite":
        second["suite"]["seeds"] = [0, 1]
    elif change == "game":
        second["games"][0]["truth"]["values"] = [9.0]
    elif change == "method":
        second["methods"]["baseline"]["source_sha256"] = "different"
    elif change == "cell":
        second["records"][0]["nmse"] = 0.9
    else:
        second["run_provenance"] = {"version": "2"}
    with pytest.raises(ValueError, match=r"same snapshot|Conflicting"):
        merge_results([write(tmp_path, first), write(tmp_path, second, "second.json")])


def test_candidate_is_local_only_and_provenance_retained(tmp_path: Path) -> None:
    """Private comparison works but public export is explicit and rejects it."""
    baseline = result_fixture()
    candidate = copy.deepcopy(baseline)
    candidate["methods"] = {"candidate": {"source_sha256": "xyz", "private": True}}
    candidate["records"][0]["method"] = "candidate"
    candidate["run_provenance"] = {"version": "2"}
    paths = [write(tmp_path, baseline), write(tmp_path, candidate, "candidate.json")]
    data = merge_results(paths)
    assert len(data["runs"]) == 2
    with pytest.raises(ValueError, match="private candidate"):
        report(paths, tmp_path / "public", public=True)
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize("status", ["failed", "unsupported"])
def test_failed_record_has_safe_error_type(tmp_path: Path, status: str) -> None:
    """Exception diagnostics must not leak local filenames into public reports."""
    result = result_fixture()
    result["records"][0].update(
        status=status, error="ValueError: /private/model.py", nmse=None, mse=None
    )
    data = merge_results([write(tmp_path, result)])
    assert data["records"][0]["error_type"] == "ValueError"
    assert "error" not in data["records"][0]


@pytest.mark.parametrize(
    ("method", "error", "minimum"),
    [
        (
            "OddSHAP",
            "ValueError: The budget is too small for OddSHAP. Received budget=8, "
            "but at least 10 evaluations are required. Please increase the budget.",
            10,
        ),
        ("ShaplEIG", "ValueError: Budget (8) must exceed the initial design size (9).", 10),
        (
            "SPEX",
            "ValueError: Insufficient budget to compute the transform. "
            "Increase the budget or use a different approximator.",
            None,
        ),
        *[
            (
                "ProxySPEX",
                "ValueError: Cannot have number of splits n_splits=5 greater than "
                f"the number of samples: n_samples={samples}.",
                None,
            )
            for samples in (2, 4)
        ],
    ],
)
def test_known_budget_failures_are_classified_without_changing_scores(
    tmp_path: Path, method: str, error: str, minimum: int | None
) -> None:
    """Specific guards become actionable metadata, retaining unsuccessful coverage."""
    result = result_fixture()
    result["suite"]["methods"] = [method]
    result["methods"] = {method: result["methods"]["baseline"]}
    result["records"][0].update(method=method, status="failed", error=error, nmse=None, mse=None)
    row = merge_results([write(tmp_path, result)])["records"][0]
    assert row["failure_reason"] == "insufficient_budget"
    assert row.get("minimum_budget") == minimum
    assert row["status"] == "failed" and row["nmse"] is None and row["mse"] is None
    assert row["error_type"] == "ValueError" and "error" not in row


@pytest.mark.parametrize(
    ("method", "error"),
    [
        ("OddSHAP", "ValueError: /private/model.py has invalid parameters"),
        ("ShaplEIG", "TimeoutError: worker wall-time limit exceeded."),
        (
            "KernelSHAP",
            "ValueError: Insufficient budget to compute the transform. "
            "Increase the budget or use a different approximator.",
        ),
        (
            "ProxySPEX",
            "ValueError: Cannot have number of splits n_splits=2 greater than "
            "the number of samples: n_samples=1.",
        ),
        (
            "OddSHAP",
            "ValueError: The budget is too small for OddSHAP. Received budget=8, "
            "but at least 10 evaluations are required. Please increase the budget. /private/path",
        ),
    ],
)
def test_unrelated_failures_are_not_budget_classified(
    tmp_path: Path, method: str, error: str
) -> None:
    """Neither generic exceptions nor similar messages from other methods qualify."""
    result = result_fixture()
    result["methods"] = {method: result["methods"]["baseline"]}
    result["records"][0].update(
        method=method,
        status="failed",
        error=error,
        nmse=None,
        mse=None,
        failure_reason="/private/reason",
        minimum_budget="/private/path",
    )
    row = merge_results([write(tmp_path, result)])["records"][0]
    assert "failure_reason" not in row and "minimum_budget" not in row
    assert "/private" not in json.dumps(row)


@pytest.mark.parametrize("minimum", ["/private/path", True, -1, 0, 1, 8, 1.5])
def test_sanitized_budget_metadata_rejects_unsafe_minimum(tmp_path: Path, minimum: object) -> None:
    """Bundle re-exports retain the category but only valid integer lower bounds."""
    result = result_fixture()
    result["methods"] = {"OddSHAP": result["methods"]["baseline"]}
    result["records"][0].update(
        method="OddSHAP",
        status="failed",
        nmse=None,
        mse=None,
        failure_reason="insufficient_budget",
        minimum_budget=minimum,
    )
    row = merge_results([write(tmp_path, result)])["records"][0]
    assert row["failure_reason"] == "insufficient_budget"
    assert "minimum_budget" not in row


def test_nonfinite_metrics_rejected(tmp_path: Path) -> None:
    """A malformed results file must not create invalid chart data."""
    result = result_fixture()
    result["records"][0]["nmse"] = float("inf")
    with pytest.raises(ValueError, match=r"Invalid numeric|not JSON compliant"):
        merge_results([write(tmp_path, result)])


def test_canonical_site_rejects_private_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tracked publication directory cannot accidentally receive local candidates."""
    canonical = tmp_path / "canonical"
    monkeypatch.setattr("shapiq_benchmark.report.SITE_DIR", canonical)
    result = result_fixture()
    result["methods"]["baseline"].pop("private")
    with pytest.raises(ValueError, match="private candidate"):
        report([write(tmp_path, result)], canonical)
    assert not canonical.exists()


def test_pending_zero_energy_game_is_excluded_from_all_methods(tmp_path: Path) -> None:
    """Frozen truth identifies undefined nMSE even before any worker finishes."""
    result = result_fixture()
    result["games"][0]["truth"]["energy"] = 0.0
    result["records"] = []
    data = report([write(tmp_path, result)], tmp_path / "report")
    assert data["games"][0]["metadata"]["zero_truth_energy"]
    assert data["presets"][0]["excluded_zero_energy_games"] == ["game"]
    assert data["presets"][0]["rows"][0]["planned"] == 0


def test_compact_export_preserves_hardware_and_instance_identity(tmp_path: Path) -> None:
    """Repeated worker details are losslessly shared without losing replicate provenance."""
    result = result_fixture()
    result["suite"]["game_seeds"] = [0, 1, 2, 3]
    result["games"][0]["metadata"].update(
        case_id="recipe", instance_seed=2, replicate_unit="fitted model", input_id="point2"
    )
    worker = {"cpu_model": "test CPU", "affinity": [3], "thread_pools": [{"num_threads": 1}]}
    result["records"][0]["worker"] = worker
    result["suite"]["budgets"].append(16)
    result["records"].append({**result["records"][0], "budget": 16, "wall_seconds": None})
    output = tmp_path / "site"
    data = report([write(tmp_path, result)], output)
    exported = json.loads((output / "data.json").read_text())
    assert len(exported["workers"]) == 1
    for original, compact in zip(data["records"], exported["records"], strict=True):
        restored = dict(compact)
        restored["worker"] = exported["workers"][restored.pop("worker_id")]
        assert restored == {key: value for key, value in original.items() if value is not None}
    assert exported["suite"]["game_seeds"] == [0, 1, 2, 3]
    assert exported["games"][0]["metadata"]["case_id"] == "recipe"
    assert exported["games"][0]["metadata"]["instance_seed"] == 2


def test_public_report_retains_player_floor_and_classifier_output(tmp_path: Path) -> None:
    """Publication must retain the selection constraint and the explained output scale."""
    result = result_fixture()
    result["suite"]["min_players"] = 11
    result["games"][0]["n_players"] = 11
    result["games"][0]["metadata"].update(class_index=1, output_scale="class probability")
    exported = report([write(tmp_path, result)], tmp_path / "site", public=True)
    assert exported["suite"]["min_players"] == 11
    metadata = exported["games"][0]["metadata"]
    assert metadata["class_index"] == 1 and metadata["output_scale"] == "class probability"
    assert "/private" not in json.dumps(exported)


def test_public_report_retains_matrix_exclusions(tmp_path: Path) -> None:
    """A reader must distinguish planned exact-table cases from excluded large games."""
    result = result_fixture()
    result["suite"]["matrix_definition"] = {"datasets": ["digits"], "player_counts": [12, 64]}
    result["suite"]["matrix_coverage"] = {
        "candidates": [
            {
                "recipe": "feature_selection",
                "dataset": "digits",
                "n_players": 12,
                "status": "selected",
                "reason": "exhaustive_table",
            },
            {
                "recipe": "feature_selection",
                "dataset": "digits",
                "n_players": 64,
                "status": "excluded",
                "reason": "unqualified_large_adapter",
            },
        ]
    }
    exported = report([write(tmp_path, result)], tmp_path / "site", public=True)
    for key in ("matrix_definition", "matrix_coverage"):
        assert exported["suite"][key] == result["suite"][key]
    assert "/private" not in json.dumps(exported)
