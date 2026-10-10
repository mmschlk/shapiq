"""Corrected implementations replace full panels without relabeling old runs."""

from __future__ import annotations

import copy
import json
import os
import runpy
import sys
from pathlib import Path

import pytest

from shapiq_benchmark import campaign_replacements
from shapiq_benchmark.campaign import assemble_campaign
from shapiq_benchmark.campaign_replacements import replace_methods
from shapiq_benchmark.results_io import Checkpoint
from shapiq_benchmark.runner import identity

from .test_campaign_export import make_campaign, write
from .test_campaign_recovery import recovery_pair


def reruns(tmp_path: Path, roots: list[Path]) -> Path:
    """Keep original snapshots, run only KernelSHAP using explicitly different code."""
    output = tmp_path / "corrected"
    output.mkdir()
    source = json.loads((roots[0] / "campaign.json").read_text())["source"]
    source = {**source, "git_commit": "corrected", "source_sha256": "new-source"}
    manifest = {"version": 1, "source": source, "methods": ["KernelSHAP"], "snapshots": []}
    for root in roots:
        campaign = json.loads((root / "campaign.json").read_text())
        for batch in campaign["batches"]:
            directory = root / batch["id"]
            snapshot_path = directory / "prepared/snapshot.json"
            if not snapshot_path.exists():
                continue
            snapshot = json.loads(snapshot_path.read_text())
            entry = {
                "snapshot_id": snapshot["snapshot_id"],
                "snapshot": os.path.relpath(snapshot_path, output),
                "results": [],
            }
            for index, path in enumerate(sorted((directory / "sweep").glob("*/results.json"))):
                result = json.loads(path.read_text())
                result["methods"] = {
                    "KernelSHAP": {
                        **result["methods"]["KernelSHAP"],
                        "source_sha256": source["source_sha256"],
                        "software_sha256": identity(source),
                    }
                }
                result["run_provenance"] = {
                    **source,
                    "execution": result["run_provenance"]["execution"],
                }
                result["records"] = [r for r in result["records"] if r["method"] == "KernelSHAP"]
                for row in result["records"]:
                    if row["status"] == "ok":
                        row.update(nmse=0.125, mse=0.25)
                count = len(result["records"])
                result["campaign"] = {"planned": count, "completed": count, "complete": True}
                result["resume_key"] = identity(
                    {
                        k: v
                        for k, v in result.items()
                        if k not in ("records", "campaign", "resume_key")
                    }
                )
                destination = (
                    output
                    / f"batch-{len(manifest['snapshots'])}"
                    / f"shard-{index}"
                    / "results.json"
                )
                write(destination, result)
                entry["results"].append(os.path.relpath(destination, output))
            manifest["snapshots"].append(entry)
    path = output / "manifest.json"
    write(path, manifest)
    return path


@pytest.fixture
def panel(tmp_path: Path) -> tuple[dict, Path]:
    root = tmp_path / "original"
    make_campaign(root, method_parameters={"KernelSHAP": {"pairing_trick": False}})
    return assemble_campaign(root, 3), reruns(tmp_path, [root])


def first_result(manifest_path: Path) -> tuple[Path, dict]:
    manifest = json.loads(manifest_path.read_text())
    path = manifest_path.parent / manifest["snapshots"][0]["results"][0]
    return path, json.loads(path.read_text())


def test_complete_replacements_preserve_games_and_other_methods(panel):
    original, manifest = panel
    before = copy.deepcopy(original)
    updated = replace_methods(original, manifest)
    assert original == before
    assert updated["games"] == original["games"] and updated["suite"] == original["suite"]
    assert updated["snapshot_provenance"] == original["snapshot_provenance"]
    old_other = [r for r in original["records"] if r["method"] != "KernelSHAP"]
    assert [r for r in updated["records"] if r["method"] != "KernelSHAP"] == old_other
    new = [r for r in updated["records"] if r["method"] == "KernelSHAP"]
    assert len(new) == 8 and all(r["nmse"] == 0.125 for r in new)
    assert all(updated["runs"][r["run_id"]]["source_sha256"] == "new-source" for r in new)
    assert updated["methods"]["KernelSHAP"]["parameters"] == {"pairing_trick": False}
    assert (
        updated["methods"]["PermutationSamplingSV"] == original["methods"]["PermutationSamplingSV"]
    )
    assert set(updated["runs"]) == {r["run_id"] for r in updated["records"]}
    assert updated["snapshot_id"] == identity(updated["composition"]) != original["snapshot_id"]
    assert "/private" not in json.dumps(updated) and str(manifest.parent) not in json.dumps(updated)


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "private",
        "parameters",
        "suite",
        "snapshot",
        "resume",
        "missing",
        "repeated",
        "incomplete",
        "unsupported",
        "duplicate",
        "selection",
    ],
)
def test_untrusted_or_partial_runs_rejected(panel, change):
    data, manifest = panel
    path, result = first_result(manifest)
    if change == "source":
        result["run_provenance"]["source_sha256"] = "wrong"
    elif change == "private":
        result["methods"]["KernelSHAP"]["private"] = True
    elif change == "parameters":
        result["methods"]["KernelSHAP"]["parameters"] = {"pairing_trick": True}
    elif change == "suite":
        result["suite"]["seeds"] = [1]
    elif change == "snapshot":
        result["snapshot_id"] = "unrelated"
    elif change == "resume":
        result["resume_key"] = "changed"
    elif change == "missing":
        result["records"].pop()
    elif change == "repeated":
        result["records"].append(result["records"][0])
    elif change == "incomplete":
        result["campaign"]["complete"] = False
    elif change in {"unsupported", "duplicate"}:
        result["records"][0]["status"] = change
    else:
        result["run_provenance"]["execution"]["game_ids"] = ["unknown"]
    if change != "resume":
        result["resume_key"] = identity(
            {k: v for k, v in result.items() if k not in ("records", "campaign", "resume_key")}
        )
    write(path, result)
    with pytest.raises(ValueError):
        replace_methods(data, manifest)


def test_missing_future_component_or_repeated_snapshot_rejected(panel):
    data, path = panel
    manifest = json.loads(path.read_text())
    manifest["snapshots"].pop()
    write(path, manifest)
    with pytest.raises(ValueError, match="every original component"):
        replace_methods(data, path)
    manifest["snapshots"].append(manifest["snapshots"][0])
    write(path, manifest)
    with pytest.raises(ValueError, match="every original component"):
        replace_methods(data, path)


@pytest.mark.parametrize("field,value", [("source_dirty", True), ("git_commit", "")])
def test_consistently_declared_unfrozen_source_is_rejected(panel, field, value):
    data, path = panel
    manifest = json.loads(path.read_text())
    manifest["source"][field] = value
    for entry in manifest["snapshots"]:
        for relative in entry["results"]:
            result_path = path.parent / relative
            result = json.loads(result_path.read_text())
            result["run_provenance"][field] = value
            result["methods"]["KernelSHAP"]["software_sha256"] = identity(manifest["source"])
            result["resume_key"] = identity(
                {k: v for k, v in result.items() if k not in ("records", "campaign", "resume_key")}
            )
            write(result_path, result)
    write(path, manifest)
    with pytest.raises(ValueError, match="clean frozen revision"):
        replace_methods(data, path)


def test_original_files_still_match_authenticated_composition(panel):
    data, manifest_path = panel
    manifest = json.loads(manifest_path.read_text())
    snapshot_path = (manifest_path.parent / manifest["snapshots"][0]["snapshot"]).resolve()
    baseline = next(snapshot_path.parent.parent.glob("sweep/*/results.json"))
    baseline.write_text(baseline.read_text() + "\n")
    with pytest.raises(ValueError, match="Original campaign input changed"):
        replace_methods(data, manifest_path)


def test_inputs_changing_during_validation_are_rejected(panel, monkeypatch):
    data, manifest = panel
    original = copy.deepcopy(data)
    merge = campaign_replacements.merge_results

    def mutate_after_read(paths):
        merged = merge(paths)
        manifest.write_text(manifest.read_text() + "\n")
        return merged

    monkeypatch.setattr(campaign_replacements, "merge_results", mutate_after_read)
    with pytest.raises(ValueError, match="Replacement input changed"):
        replace_methods(data, manifest)
    assert data == original


def test_compact_replacements_authenticate_journal(panel):
    data, manifest = panel
    path, result = first_result(manifest)
    spec = json.loads(manifest.read_text())["snapshots"][0]
    Checkpoint(path.parent, manifest.parent / spec["snapshot"], result)
    assert replace_methods(data, manifest)["records"]
    with (path.parent / "records.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="journal"):
        replace_methods(data, manifest)


def test_supplement_and_duplicate_markers_are_fully_replaced(tmp_path):
    parent, retry = recovery_pair(tmp_path)
    data = assemble_campaign(parent, 3, supplements=(retry,))
    manifest = reruns(tmp_path, [parent, retry])
    updated = replace_methods(data, manifest)
    assert updated["duplicate_games"] == data["duplicate_games"]
    assert len(updated["records"]) == len(data["records"])
    assert all(r["status"] != "duplicate" for r in updated["records"])
    assert all(r["nmse"] == 0.125 for r in updated["records"] if r["method"] == "KernelSHAP")


def test_export_cli_applies_optional_replacements(panel, tmp_path, monkeypatch):
    data, manifest = panel
    output = tmp_path / "site"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_phase.py",
            str(tmp_path / "original"),
            str(output),
            "--through-phase",
            "3",
            "--replacements",
            str(manifest),
        ],
    )
    monkeypatch.setattr("shapiq_benchmark.report.summarize", lambda *a, **k: [])
    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "benchmark/export_phase.py"), run_name="__main__"
    )
    exported = json.loads((output / "data.json").read_text())
    assert exported["snapshot_id"] != data["snapshot_id"]
    assert exported["methods"]["KernelSHAP"]["source_sha256"] == "new-source"
