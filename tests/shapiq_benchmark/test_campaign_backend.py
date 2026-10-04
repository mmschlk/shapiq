"""GPU recovery needs explicit evidence and a complete separately frozen CPU panel."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from shapiq_benchmark.campaign import _collect_campaign, assemble_campaign
from shapiq_benchmark.campaign_backend import load_backend, merge_backend
from shapiq_benchmark.report import write_report
from shapiq_benchmark.runner import digest, identity

from .test_campaign_export import make_campaign, write
from .test_campaign_replacements import reruns


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def pin(path: Path) -> dict:
    return {"path": str(path), "sha256": digest(path), "identity": identity(read(path))}


def recovery(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Preserve one original CPU batch; explicitly supersede another failed GPU batch."""
    root, retry = tmp_path / "original", tmp_path / "retry"
    make_campaign(root)
    make_campaign(retry)
    original_campaign = read(root / "campaign.json")
    original_campaign["scripts"] = {"prepare.py": "unchanged"}
    old_dir = root / "batch-1"
    suite = read(old_dir / "suite.json")
    suite["families"][0]["device"] = "cuda"
    write(old_dir / "suite.json", suite)
    original_campaign["batches"][1]["suite_sha256"] = identity(suite)
    write(root / "campaign.json", original_campaign)
    write(root / "jobs.json", {"plan_sha256": identity(original_campaign)})
    # A failed preparation has no fabricated qualification, snapshot or estimates.
    for path in old_dir.iterdir():
        if path.name != "suite.json":
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    source = {
        **original_campaign["source"],
        "git_commit": "corrected",
        "source_sha256": "new-source",
    }
    directory = retry / "batch-1"
    original = copy.deepcopy(suite)
    original["families"][0].update(id="recipe-1-cpu", device="cpu")
    qualified = {
        **original,
        "preparation_preflight": {"version": 1, "families": []},
        "preparation_exclusions": [],
    }
    write(directory / "suite.json", original)
    write(directory / "qualified-suite.json", qualified)
    write(
        directory / "qualification-decision.json",
        {
            "requested_suite_sha256": identity(original),
            "qualified_suite_sha256": identity(qualified),
            "source": source,
        },
    )

    def rename(value):
        return value.replace("recipe-1-", "recipe-1-cpu-")

    snapshot = read(directory / "prepared/snapshot.json")
    snapshot["provenance"] = source
    for game in snapshot["games"]:
        game["id"] = rename(game["id"])
    snapshot["suite"] = {
        **qualified,
        "budgets": [11, 22],
        "budgets_by_game": {g["id"]: [11, 22] for g in snapshot["games"]},
    }
    snapshot["snapshot_id"] = identity({k: v for k, v in snapshot.items() if k != "snapshot_id"})
    write(directory / "prepared/snapshot.json", snapshot)
    allocation = read(directory / "sweep/allocation.json")
    allocation.update(
        snapshot_id=snapshot["snapshot_id"], game_ids=[rename(g) for g in allocation["game_ids"]]
    )
    write(directory / "sweep/allocation.json", allocation)
    for path in (directory / "sweep").glob("*/results.json"):
        data = read(path)
        data.update(
            snapshot_id=snapshot["snapshot_id"],
            snapshot_provenance=source,
            suite=snapshot["suite"],
            games=snapshot["games"],
        )
        data["methods"] = {
            name: {
                "source_sha256": source["source_sha256"],
                "software_sha256": identity(source),
                "private": False,
            }
            for name in original["methods"]
        }
        data["run_provenance"] = {
            **source,
            "execution": {
                "game_ids": [rename(g) for g in data["run_provenance"]["execution"]["game_ids"]]
            },
        }
        for row in data["records"]:
            row["game_id"] = rename(row["game_id"])
        data["resume_key"] = identity(
            {k: v for k, v in data.items() if k not in ("records", "campaign", "resume_key")}
        )
        write(path, data)
    campaign = {
        "source": source,
        "scripts": original_campaign["scripts"],
        "batches": [
            {
                "id": "batch-1",
                "phase": 3,
                "directory": str(directory),
                "suite_sha256": identity(original),
            }
        ],
    }
    write(retry / "campaign.json", campaign)
    write(retry / "jobs.json", {"plan_sha256": identity(campaign)})
    guard = tmp_path / "guard.json"
    write(
        guard,
        {
            "reason": "sustained_low_utilization",
            "guard_returncode": 75,
            "threshold_percent": 80,
            "utilization_verified": False,
            "command_completed": False,
        },
    )
    terminal = tmp_path / "terminal.json"
    write(
        terminal,
        {
            "job_id": "100_0",
            "state": "FAILED",
            "exit_code": "75:0",
            "batch_id": "batch-1",
            "original_campaign_sha256": digest(root / "campaign.json"),
            "original_suite_sha256": digest(old_dir / "suite.json"),
            "guard_sha256": digest(guard),
            "accounting": "100_0|100|FAILED|75:0|1|gpu01\n",
            "superseded_jobs": {
                "101_0": {"state": "CANCELLED", "accounting": "101_0|101|CANCELLED|0:0|1|None\n"}
            },
        },
    )
    entry = {
        "batch_id": "batch-1",
        "phase": 3,
        "reason": "sustained_low_gpu_utilization",
        "original_suite": pin(old_dir / "suite.json"),
        "guard": pin(guard),
        "terminal": pin(terminal),
        "recovery_campaign": {
            "root": str(retry),
            "sha256": digest(retry / "campaign.json"),
            "identity": identity(campaign),
            "source": source,
        },
        "recipe_mapping": {"recipe-1": "recipe-1-cpu"},
        "superseded_pending_jobs": ["101_0"],
    }
    manifest = tmp_path / "backend.json"
    write(
        manifest,
        {"version": 1, "original_campaign": pin(root / "campaign.json"), "supersessions": [entry]},
    )
    return root, retry, manifest


def test_backend_recovery_preserves_originals_and_sources(tmp_path):
    root, retry, manifest = recovery(tmp_path)
    before = {
        path: digest(path) for base in (root, retry) for path in base.rglob("*") if path.is_file()
    }
    data = assemble_campaign(root, 3, backend_supersession=manifest)
    assert len(data["records"]) == 16 and len(data["games"]) == 4
    assert (
        next(p for p in data["composition"]["components"] if p["batch"] == "batch-1")["status"]
        == "superseded"
    )
    assert all("recipe-1-cpu" in g["id"] or "recipe-0" in g["id"] for g in data["games"])
    assert all(
        "source_versions" in m and len(m["source_versions"]) == 2 for m in data["methods"].values()
    )
    for row in data["records"]:
        source = data["runs"][row["run_id"]]["source_sha256"]
        assert source == ("new-source" if "cpu" in row["game_id"] else "source")
    assert str(tmp_path) not in json.dumps(data)
    assert all(digest(path) == sha for path, sha in before.items())
    assert not (root / "batch-1/qualified-suite.json").exists()
    output = tmp_path / "site"
    write_report(data, output, public=True, compact=True)
    assert read(output / "data.json")["methods"] == data["methods"]


def test_replacements_cover_only_original_components(tmp_path):
    root, _, manifest = recovery(tmp_path)
    replacement = reruns(tmp_path, [root])
    data = assemble_campaign(root, 3, backend_supersession=manifest, replacements=replacement)
    assert data["methods"]["KernelSHAP"]["source_sha256"] == "new-source"
    assert "source_versions" not in data["methods"]["KernelSHAP"]
    assert "source_versions" in data["methods"]["PermutationSamplingSV"]
    assert len(read(replacement)["snapshots"]) == 1  # Recovery uses corrected source directly.
    assert len(data["records"]) == 16
    assert {r["run_id"] for r in data["records"]} == data["runs"].keys()


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "guard",
        "active",
        "wrong_batch",
        "cancel",
        "mapping",
        "source",
        "missing",
        "incomplete",
    ],
)
def test_backend_rejects_untrusted_or_incomplete_recovery(tmp_path, change):
    root, retry, manifest = recovery(tmp_path)
    value = read(manifest)
    entry = value["supersessions"][0]
    if change in ("guard", "active", "wrong_batch", "cancel"):
        key = "guard" if change == "guard" else "terminal"
        path = Path(entry[key]["path"])
        receipt = read(path)
        if change == "guard":
            receipt["reason"] = "command_finished"
        elif change == "active":
            receipt["state"] = "RUNNING"
        elif change == "wrong_batch":
            receipt["batch_id"] = "batch-0"
        else:
            receipt["superseded_jobs"] = {}
        write(path, receipt)
        entry[key] = pin(path)
    elif change == "hash":
        entry["original_suite"]["sha256"] = "wrong"
    elif change == "mapping":
        entry["recipe_mapping"] = {"recipe-1": "unrelated"}
    elif change == "source":
        entry["recovery_campaign"]["source"]["source_dirty"] = True
    else:
        path = retry / "batch-1/sweep/shard-000/results.json"
        result = read(path)
        if change == "missing":
            result["records"].pop()
        else:
            result["campaign"]["complete"] = False
        write(path, result)
    write(manifest, value)
    with pytest.raises((ValueError, KeyError)):
        assemble_campaign(root, 3, backend_supersession=manifest)


def test_backend_cannot_bypass_missing_original_without_manifest(tmp_path):
    root, _, _ = recovery(tmp_path)
    with pytest.raises(FileNotFoundError):
        assemble_campaign(root, 3)


@pytest.mark.parametrize(
    "field",
    [
        "device",
        "dataset",
        "n_players",
        "relative_budgets",
        "game_seeds",
        "method_parameters",
        "protocol",
    ],
)
def test_recovery_allows_only_backend_change_after_normal_authentication(tmp_path, field):
    root, retry, manifest = recovery(tmp_path)
    backend = load_backend(manifest, root, 3)
    data, context = _collect_campaign(root, 3, backend=backend)
    extra, retry_context = _collect_campaign(retry, 3)
    requested = next(iter(retry_context["requested"].values()))
    if field in ("device", "dataset", "n_players"):
        requested["spec"][field] = "changed"
    else:
        requested["suite"][field] = "changed"
    with pytest.raises(ValueError, match="changed"):
        merge_backend(data, context, extra, retry_context, backend)


def test_backend_aliases_are_removed_after_corrected_originals_join(tmp_path):
    root, retry, manifest = recovery(tmp_path)
    directory = retry / "batch-1"
    snapshot = read(directory / "prepared/snapshot.json")
    for name in snapshot["artifacts"]:
        shutil.copyfile(root / "batch-0/prepared" / name, directory / "prepared" / name)
        snapshot["artifacts"][name] = digest(directory / "prepared" / name)
    snapshot["snapshot_id"] = identity({k: v for k, v in snapshot.items() if k != "snapshot_id"})
    write(directory / "prepared/snapshot.json", snapshot)
    allocation = read(directory / "sweep/allocation.json")
    allocation["snapshot_id"] = snapshot["snapshot_id"]
    write(directory / "sweep/allocation.json", allocation)
    for path in (directory / "sweep").glob("*/results.json"):
        data = read(path)
        data["snapshot_id"] = snapshot["snapshot_id"]
        for row in data["records"]:
            row.update(
                status="duplicate",
                duplicate_of=row["game_id"].replace("recipe-1-cpu", "recipe-0"),
                mse=None,
                nmse=None,
            )
            row.pop("estimate")
        data["resume_key"] = identity(
            {k: v for k, v in data.items() if k not in ("records", "campaign", "resume_key")}
        )
        write(path, data)
    replacements = reruns(tmp_path, [root])
    data = assemble_campaign(root, 3, backend_supersession=manifest, replacements=replacements)
    assert len(data["games"]) == 2 and len(data["records"]) == 8
    assert len(data["duplicate_games"]) == 2
    assert all(
        row["status"] != "duplicate" and "recipe-0" in row["game_id"] for row in data["records"]
    )
    assert data["snapshot_id"] == identity(data["composition"])


def test_backend_rejects_conflicting_source_for_same_software_identity(tmp_path):
    root, retry, manifest = recovery(tmp_path)
    backend = load_backend(manifest, root, 3)
    data, context = _collect_campaign(root, 3, backend=backend)
    extra, retry_context = _collect_campaign(retry, 3)
    extra["methods"]["KernelSHAP"]["software_sha256"] = data["methods"]["KernelSHAP"][
        "software_sha256"
    ]
    with pytest.raises(ValueError, match="same software identity"):
        merge_backend(data, context, extra, retry_context, backend)


def test_backend_already_corrected_source_must_match_replacement_source(tmp_path):
    root, retry, manifest = recovery(tmp_path)
    backend = load_backend(manifest, root, 3)
    data, context = _collect_campaign(root, 3, backend=backend)
    extra, retry_context = _collect_campaign(retry, 3)
    data["composition"]["replacements"] = {"source": {"git_commit": "different"}}
    with pytest.raises(ValueError, match="same corrected source"):
        merge_backend(data, context, extra, retry_context, backend)


def test_backend_cannot_reuse_a_scientifically_excluded_recipe_id(tmp_path):
    root, retry, manifest = recovery(tmp_path)
    backend = load_backend(manifest, root, 3)
    data, context = _collect_campaign(root, 3, backend=backend)
    extra, retry_context = _collect_campaign(retry, 3)
    assert "recipe-2" in context["exclusions"]
    backend["entry"]["recipe_mapping"] = {"recipe-1": "recipe-2"}
    with pytest.raises(ValueError, match="all original requested recipes"):
        merge_backend(data, context, extra, retry_context, backend)
