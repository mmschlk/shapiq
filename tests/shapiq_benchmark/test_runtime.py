"""Portable final execution policy and authenticated child-only source patches."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from shapiq_benchmark import (
    campaign_runner as runner,
    execution,
)
from shapiq_benchmark.runtime import ExecutionSettings, load_overlays, recovery_memory
from tests.shapiq_benchmark.test_campaign import TIMEOUT_POLICY, snapshot


def test_explicit_settings_override_frozen_policy_and_bind_resume(tmp_path, monkeypatch):
    path = snapshot(tmp_path)
    data = json.loads(path.read_text())
    data["suite"]["cell_timeout_policy"] = TIMEOUT_POLICY
    data["snapshot_id"] = runner.identity({k: v for k, v in data.items() if k != "snapshot_id"})
    path.write_text(json.dumps(data))
    frozen = path.read_bytes()
    settings = ExecutionSettings(approximate_seconds=10, startup_seconds=20, output_seconds=30)
    calls = []

    def worker(request, timeout, memory, **kwargs):
        calls.append((request["budget"], request["seed"], timeout, memory, kwargs))
        return {"status": "failed", "error": "fixture", "wall_seconds": 0}

    monkeypatch.setattr(execution, "isolated", worker)
    result = runner.run(path, tmp_path / "results", settings=settings, max_runs=1)
    assert result["run_provenance"]["execution"]["settings"] == asdict(settings)
    assert calls == [(8, 0, 10, 8, {"startup_timeout": 20, "output_timeout": 30})]
    result = runner.run(path, tmp_path / "results", settings=settings, resume=True)
    assert result["campaign"]["complete"]
    assert calls[1][1] == 1
    assert path.read_bytes() == frozen
    with pytest.raises(ValueError, match="identical snapshot"):
        runner.run(
            path, tmp_path / "results", settings=replace(settings, output_seconds=31), resume=True
        )


def test_settings_reserve_all_phases(tmp_path, monkeypatch):
    path = snapshot(tmp_path)
    settings = ExecutionSettings(approximate_seconds=10, startup_seconds=20, output_seconds=30)
    with pytest.raises(ValueError, match="startup"):
        runner.run(path, tmp_path / "short", settings=settings, max_seconds=60)
    times = iter([0, 50])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        execution, "isolated", lambda *a, **k: pytest.fail("insufficient wall allowance")
    )
    result = runner.run(path, tmp_path / "partial", settings=settings, max_seconds=100)
    assert result["records"] == []
    assert not result["campaign"]["complete"]


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_invalid_settings(value):
    with pytest.raises(ValueError, match="positive finite"):
        ExecutionSettings(output_seconds=value)


def test_settings_file_and_recovery_memory(tmp_path):
    assert ExecutionSettings.load(Path("benchmark/settings.json")) == ExecutionSettings()
    path = tmp_path / "bad.json"
    path.write_text('{"typo": 1}')
    with pytest.raises(TypeError):
        ExecutionSettings.load(path)
    assert recovery_memory("RegressionFBII", 2) == 512
    assert recovery_memory("RegressionFBII", 1) == 128
    assert recovery_memory("OddSHAP", 1, previous_gib=256) == 256


def overlay_manifest(tmp_path):
    import shapiq.approximator.regression.kernelshap as base

    source = Path(base.__file__)
    patch = tmp_path / "kernelshap.py"
    # Same algorithm, with a marker proving that the fresh child executed these bytes.
    patch.write_bytes(source.read_bytes() + b'\nraise RuntimeError("overlay executed")\n')
    manifest = tmp_path / "overlays.json"
    manifest.write_text(
        json.dumps(
            {
                "KernelSHAP": {
                    "module": base.__name__,
                    "path": patch.name,
                    "sha256": runner.digest(patch),
                    "base_sha256": runner.digest(source),
                }
            }
        )
    )
    return manifest, patch, source


def test_overlay_child_import_and_base_authentication(tmp_path):
    path = snapshot(tmp_path)
    manifest, patch, base = overlay_manifest(tmp_path)
    original = base.read_bytes()
    result = runner.run(path, tmp_path / "patched", overlays=manifest, max_runs=1)
    assert "overlay executed" in result["records"][0]["error"]
    descriptor = result["methods"]["KernelSHAP"]["source_overlay"]
    assert descriptor["sha256"] == runner.digest(patch)
    assert "path" not in descriptor
    assert base.read_bytes() == original
    data = json.loads(manifest.read_text())
    data["KernelSHAP"]["base_sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))
    result = runner.run(path, tmp_path / "wrong-base", overlays=manifest, max_runs=1)
    assert "underlying estimator source" in result["records"][0]["error"]
    patch.write_text("# tampered")
    with pytest.raises(ValueError, match="source changed"):
        load_overlays(manifest)


def test_overlay_success_retains_science_and_resume_identity(tmp_path):
    path = snapshot(tmp_path)
    manifest, patch, base = overlay_manifest(tmp_path)
    patch.write_bytes(base.read_bytes() + b"\n# qualified no-op overlay\n")
    data = json.loads(manifest.read_text())
    data["KernelSHAP"]["sha256"] = runner.digest(patch)
    manifest.write_text(json.dumps(data))
    settings = ExecutionSettings(approximate_seconds=10, startup_seconds=60, output_seconds=60)
    patched = runner.run(
        path, tmp_path / "patched", overlays=manifest, settings=settings, max_runs=1
    )
    plain = runner.run(path, tmp_path / "plain", settings=settings, max_runs=1)
    assert patched["records"][0]["status"] == plain["records"][0]["status"] == "ok"
    for key in ("estimate", "nmse", "queries", "budget", "seed"):
        assert patched["records"][0][key] == plain["records"][0][key]
    assert (
        patched["methods"]["KernelSHAP"]["software_sha256"]
        != plain["methods"]["KernelSHAP"]["software_sha256"]
    )
    with pytest.raises(ValueError, match="identical snapshot"):
        runner.run(path, tmp_path / "patched", settings=settings, resume=True)
