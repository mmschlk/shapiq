"""Campaign isolation, resumption, and resource-profile regression tests."""

from __future__ import annotations

import fcntl
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from shapiq_benchmark import execution
from shapiq_benchmark.execution import PROFILE, THREAD_VARIABLES, verify_profile
from shapiq_benchmark.runner import builtin_factory, digest, identity, method_catalog, run

if TYPE_CHECKING:
    from pathlib import Path


def snapshot(tmp_path: Path) -> Path:
    """A tiny deterministic complete table keeps process tests independent of datasets."""
    values = np.array([0, 1, 2, 3, 3, 4, 5, 6], dtype=float)
    np.savez(tmp_path / "game.npz", values=values)
    data = {
        "schema_version": 1,
        "provenance": {},
        "suite": {"methods": ["KernelSHAP"], "budgets": [8], "seeds": [0, 1]},
        "artifacts": {"game.npz": digest(tmp_path / "game.npz")},
        "games": [
            {
                "id": "tiny",
                "n_players": 3,
                "index": "SV",
                "order": 1,
                "artifact": "game.npz",
                "truth": {"coordinates": [[0], [1], [2]], "values": [1.0, 2.0, 3.0]},
            }
        ],
    }
    data["snapshot_id"] = identity(data)
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(data))
    return path


def test_campaign_resume_matches_full_run(tmp_path: Path) -> None:
    """Resumption skips recorded cells without changing their estimates or provenance."""
    path = snapshot(tmp_path)
    partial = run(path, tmp_path / "partial", max_runs=1)
    assert partial["campaign"] == {"planned": 2, "completed": 1, "complete": False}
    resumed = run(path, tmp_path / "partial", resume=True)
    full = run(path, tmp_path / "full")
    assert resumed["records"][0] == partial["records"][0]
    assert [row["estimate"] for row in resumed["records"]] == [
        row["estimate"] for row in full["records"]
    ]
    assert resumed["campaign"]["complete"]
    assert all(row["status"] == "ok" for row in resumed["records"])
    assert all(pool["num_threads"] == 1 for pool in resumed["records"][0]["worker"]["thread_pools"])
    with pytest.raises(ValueError, match="requires --resume"):
        run(path, tmp_path / "partial")
    with pytest.raises(ValueError, match="identical snapshot"):
        run(path, tmp_path / "partial", resume=True, timeout=30)


def test_timeout_preserves_next_seed_and_changed_candidate_refuses_resume(tmp_path: Path) -> None:
    """A hanging factory cannot prevent later cells from running or forge zero queries."""
    path = snapshot(tmp_path)
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        "import time\nfrom shapiq.approximator import KernelSHAP\n"
        "def factory(n,index,order,seed):\n"
        "    if seed == 0: time.sleep(60)\n"
        "    return KernelSHAP(n=n,random_state=seed)\n"
    )
    spec = f"{candidate}:factory"
    result = run(path, tmp_path / "results", spec, timeout=4)
    assert result["records"][0]["status"] == "failed"
    assert "TimeoutError" in result["records"][0]["error"]
    assert result["records"][0]["queries"] is None
    assert result["records"][1]["status"] == "ok"
    candidate.write_text(candidate.read_text() + "# changed source\n")
    with pytest.raises(ValueError, match="identical snapshot"):
        run(path, tmp_path / "results", spec, timeout=4, resume=True)


def test_catalog_has_all_public_estimators() -> None:
    """The catalog does not omit optional algorithms or claim placeholder target support."""
    catalog = method_catalog()
    assert len(catalog) == 22
    assert catalog["ShaplEIG"]["indices"] == ["SV"]
    assert catalog["RegressionMSR"]["indices"] == ["SV", "BV"]
    assert "SPEX" in catalog
    assert "k-SII" not in catalog["KernelSHAP"]["indices"]


def test_profile_requires_full_exclusive_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """No-CPU-sharing alone does not prove exclusive node ownership."""
    monkeypatch.setattr(
        "shapiq_benchmark.execution.hardware",
        lambda: {"cpu_model": "AMD EPYC 9754", "affinity": [0]},
    )
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("SLURM_JOB_ID", "123")

    def control(command: list[str], **kwargs: object) -> str:
        del kwargs
        if "job" in command:
            return "OverSubscribe=NO NumNodes=1 NodeList=himem02 NumCPUs=1"
        return "CPUTot=128 ThreadsPerCore=1"

    monkeypatch.setattr("shapiq_benchmark.execution.subprocess.check_output", control)
    with pytest.raises(ValueError, match="complete physical node"):
        verify_profile(PROFILE)


def test_resume_header_and_cell_matrix_are_validated(tmp_path: Path) -> None:
    """A copied resume token cannot authenticate edited experiment metadata."""
    path = snapshot(tmp_path)
    output = tmp_path / "results"
    run(path, output, max_runs=1)
    checkpoint = output / "results.json"
    previous = json.loads(checkpoint.read_text())
    previous["suite"]["budgets"] = [999]
    checkpoint.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="identical snapshot"):
        run(path, output, resume=True)


@pytest.mark.parametrize(
    "limits", [{"timeout": float("nan")}, {"memory_gb": float("inf")}, {"max_seconds": 1}]
)
def test_invalid_resource_limits_rejected(tmp_path: Path, limits: dict) -> None:
    """Every executed cell receives the full finite declared resource allowance."""
    with pytest.raises(ValueError, match="finite and positive"):
        run(snapshot(tmp_path), tmp_path / "results", **limits)


def test_concurrent_campaign_cannot_overwrite_checkpoint(tmp_path: Path) -> None:
    """A second process must not race the first campaign's checkpoint writer."""
    path = snapshot(tmp_path)
    output = tmp_path / "results"
    output.mkdir()
    with (output / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            run(path, output)
    assert not (output / "results.json").exists()


def test_candidate_changed_before_worker_import_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Source edits between scheduling and execution cannot be mislabeled as the old method."""
    path = snapshot(tmp_path)
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def factory(n,index,order,seed): return None\n")
    isolate = execution.isolated

    def change_source(request: dict, timeout: float, memory_gb: float | None) -> dict:
        candidate.write_text("raise RuntimeError('this source must not be imported')\n")
        return isolate(request, timeout, memory_gb)

    monkeypatch.setattr(execution, "isolated", change_source)
    result = run(path, tmp_path / "results", f"{candidate}:factory", max_runs=1)
    assert result["records"][0]["status"] == "failed"
    assert "Candidate source changed" in result["records"][0]["error"]


def test_missing_optional_backend_has_useful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An optional placeholder reports its dependency instead of an abstract-class error."""

    class Missing:
        _import_error = ImportError("Install the optional backend")

    monkeypatch.setitem(
        __import__("shapiq_benchmark.runner", fromlist=["METHODS"]).METHODS, "SPEX", Missing
    )
    with pytest.raises(ImportError, match="optional backend"):
        builtin_factory("SPEX", {"n_players": 3, "index": "SV", "order": 1}, 0)
