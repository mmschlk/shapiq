"""Cross-snapshot exports preserve provenance, complete panels and global weights."""

from __future__ import annotations

import copy
import fcntl
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from shapiq_benchmark.campaign import _canonical_aliases, _historical_quality, assemble_campaign
from shapiq_benchmark.runner import digest, identity
from shapiq_benchmark.summary import summarize
from tests.shapiq_benchmark.test_report import decode_columns


def test_clustering_publication_check_preserves_frozen_quality(tmp_path: Path) -> None:
    """Versioned export diagnostics supplement existing quality without rewriting it."""
    values = np.array([0, 1e35, 30, 40])
    artifact = tmp_path / "game.npz"
    np.savez(artifact, values=values)
    original_bytes = artifact.read_bytes()
    game = {
        "artifact": "game.npz",
        "n_players": 2,
        "metadata": {
            "class": "shapiq_games.benchmark.unsupervised_cluster.base.ClusterExplanation",
            "background_indices": list(range(128)),
            "game_quality": {"protocol": "quality-v2", "role": "core", "control_reasons": []},
        },
    }
    original = copy.deepcopy(game)
    quality = _historical_quality(game, tmp_path)["game_quality"]
    assert quality["role"] == "control"
    assert quality["role_before_clustering_check"] == "core"
    assert quality["protocol"] == "quality-v2"
    assert quality["clustering_numerics"]["protocol"] == "calinski-harabasz-resolution-v1"
    assert game == original
    assert artifact.read_bytes() == original_bytes


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_campaign(
    root: Path,
    *,
    duplicate: bool = False,
    structured: bool = False,
    method_parameters: dict | None = None,
) -> dict:
    """Two complete batches and one explicitly excluded recipe, without external data."""
    source = {"source_dirty": False, "git_commit": "frozen", "source_sha256": "source"}
    shared = {
        "methods": ["KernelSHAP", "PermutationSamplingSV"],
        "relative_budgets": [1, 2],
        "game_seeds": [0, 1],
        "seeds": [0],
        "targets": [{"index": "SV", "order": 1}],
        "min_players": 11,
        "min_signal_ratio": 1e-6,
        "protocol": {"version": 1},
    }
    if method_parameters is not None:
        shared["method_parameters"] = method_parameters
    inventory = {**shared, "name": "phase-three", "phase_plan": {"counts": {"selected": 3}}}
    write(root / "phase-3-inventory.json", inventory)
    batches = []
    for number in range(3):
        batch_id = f"batch-{number}"
        directory = root / batch_id
        recipe = {"id": "same" if duplicate else f"recipe-{number}"}
        original = {
            **shared,
            "name": batch_id,
            "families": [recipe],
            "games": [],
            "phase_plan": {"inventory_sha256": identity(inventory["phase_plan"])},
        }
        if structured and number == 0:
            original.update(families=[], games=[recipe])
        qualified = copy.deepcopy(original)
        qualified.update(
            preparation_preflight={"version": 1, "families": []}, preparation_exclusions=[]
        )
        if number == 2:
            qualified.update(
                families=[],
                preparation_exclusions=[
                    {
                        "spec": recipe,
                        "reason": "projected_cost",
                        "maximum_seconds_per_instance": 28800,
                        "instances": [
                            {
                                "seed": 0,
                                "status": "excluded",
                                "reason": "projected_cost",
                                "error": "/private/raw",
                            }
                        ],
                    }
                ],
            )
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
        batches.append(
            {
                "id": batch_id,
                "phase": 3,
                "directory": str(directory.resolve()),
                "suite_sha256": identity(original),
            }
        )
        if number == 2:
            write(directory / "excluded.json", {"suite_sha256": identity(qualified)})
            continue
        artifact = directory / "prepared/game.npz"
        artifact.parent.mkdir(parents=True)
        for seed in shared["game_seeds"]:
            np.savez(
                artifact.with_name(f"game-{seed}.npz"),
                values=np.arange(2048, dtype=float) * (1 + number * 2 + seed),
            )
        games = [
            {
                "id": f"{recipe['id']}-i{seed}" + ("" if structured and number == 0 else "-sv-1"),
                "family": f"family-{number}",
                "stratum": "shared",
                "n_players": 11,
                "index": "SV",
                "order": 1,
                "artifact": f"game-{seed}.npz",
                "truth": {"energy": 1, "values": [1]},
                "metadata": {"instance_seed": seed, "cluster_id": f"model-{seed}"},
            }
            for seed in shared["game_seeds"]
        ]
        suite = {
            **qualified,
            "budgets": [11, 22],
            "budgets_by_game": {g["id"]: [11, 22] for g in games},
        }
        snapshot = {
            "schema_version": 1,
            "provenance": source,
            "suite": suite,
            "games": games,
            "artifacts": {g["artifact"]: digest(artifact.parent / g["artifact"]) for g in games},
            "coverage": [],
        }
        snapshot["snapshot_id"] = identity(snapshot)
        write(directory / "prepared/snapshot.json", snapshot)
        write(
            directory / "sweep/allocation.json",
            {
                "snapshot_id": snapshot["snapshot_id"],
                "cpus": [0, 1],
                "game_ids": [g["id"] for g in reversed(games)],
            },
        )
        methods = {
            name: {"source_sha256": "source", "software_sha256": identity(source), "private": False}
            for name in shared["methods"]
        }
        for name, parameters in (method_parameters or {}).items():
            if parameters:
                methods[name]["parameters"] = parameters
        for slot, game in enumerate(reversed(games)):
            result = {
                "schema_version": 1,
                "snapshot_id": snapshot["snapshot_id"],
                "snapshot_provenance": source,
                "suite": suite,
                "games": games,
                "methods": methods,
                "coverage": [],
                "run_provenance": {**source, "execution": {"game_ids": [game["id"]]}},
                "records": [],
            }
            result["resume_key"] = identity({k: v for k, v in result.items() if k != "records"})
            for method in methods:
                for budget in suite["budgets"]:
                    result["records"].append(
                        {
                            "game_id": game["id"],
                            "method": method,
                            "budget": budget,
                            "seed": 0,
                            "status": "ok",
                            "mse": float(number + 1),
                            "nmse": (100.0 if game["metadata"]["instance_seed"] else 0.0)
                            if number == 0
                            else 1.0,
                            "estimate": {"values": [99]},
                        }
                    )
            result["campaign"] = {"planned": 4, "completed": 4, "complete": True}
            path = directory / "sweep" / f"shard-{slot:03}"
            write(path / "results.json", result)
            (path / ".campaign.lock").touch()
    campaign = {"source": source, "batches": batches}
    write(root / "campaign.json", campaign)
    write(root / "jobs.json", {"plan_sha256": identity(campaign)})
    return campaign


def test_disjoint_campaign_keeps_exclusions_and_global_statistics(tmp_path: Path) -> None:
    make_campaign(tmp_path)
    data = assemble_campaign(tmp_path, 3)
    assert len(data["games"]) == 4 and len(data["records"]) == 16
    assert data["snapshot_id"] == identity(data["composition"])
    assert data["composition"]["components"][-1]["status"] == "excluded"
    assert len(data["suite"]["preparation_exclusions"]) == 1
    assert "/private" not in json.dumps(data) and '"estimate"' not in json.dumps(data)
    assert not any(g["id"].startswith("recipe-2") for g in data["games"])
    panel = next(
        p
        for p in summarize(data, bootstrap_draws=0)
        if p["family"] is None and p["relative_budget"] is None
    )
    assert all(row["mean"] == 25.5 and row["median"] == 1.0 for row in panel["rows"])
    assert all(row["missing"] == 0 and row["complete"] for row in panel["rows"])


@pytest.mark.parametrize("change", ["missing", "duplicate", "method", "source", "allocation"])
def test_changed_or_incomplete_shards_rejected(tmp_path: Path, change: str) -> None:
    make_campaign(tmp_path)
    path = tmp_path / "batch-0/sweep/shard-000/results.json"
    result = json.loads(path.read_text())
    if change == "missing":
        result["records"].pop()
    elif change == "duplicate":
        result["records"].append(result["records"][0])
    elif change == "method":
        result["methods"]["KernelSHAP"]["source_sha256"] = "old-estimator"
    elif change == "source":
        result["run_provenance"]["source_sha256"] = "wrong-checkout"
    else:
        result["run_provenance"]["execution"]["game_ids"] = ["another-game"]
    write(path, result)
    with pytest.raises(ValueError):
        assemble_campaign(tmp_path, 3)


@pytest.mark.parametrize(
    "change", ["artifact", "qualification", "journal", "inventory", "exclusion"]
)
def test_authenticated_inputs_cannot_change(tmp_path: Path, change: str) -> None:
    make_campaign(tmp_path)
    if change == "artifact":
        (tmp_path / "batch-0/prepared/game-0.npz").write_bytes(b"changed")
    else:
        path = {
            "qualification": "batch-0/qualified-suite.json",
            "journal": "jobs.json",
            "inventory": "phase-3-inventory.json",
            "exclusion": "batch-2/excluded.json",
        }[change]
        value = json.loads((tmp_path / path).read_text())
        if change == "qualification":
            value["families"] = []
        elif change == "journal":
            value["plan_sha256"] = "changed"
        elif change == "inventory":
            value["protocol"] = {"version": 100}
        else:
            value["suite_sha256"] = "changed"
        write(tmp_path / path, value)
    with pytest.raises(ValueError):
        assemble_campaign(tmp_path, 3)


def test_duplicate_games_cannot_inflate_coverage(tmp_path: Path) -> None:
    make_campaign(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="Duplicate game IDs"):
        assemble_campaign(tmp_path, 3)


def test_canonical_exact_game_prefers_evaluated_core_without_relabeling() -> None:
    """Registration order cannot hide core evidence behind an equivalent control."""
    games = [
        {"id": name, "metadata": {"game_quality": {"role": role}}}
        for name, role in [
            ("first-control", "control"),
            ("later-core", "core"),
            ("core-copy", "core"),
        ]
    ]
    records = [
        {"game_id": "first-control", "status": "ok"},
        {"game_id": "later-core", "status": "ok"},
        {"game_id": "core-copy", "status": "duplicate", "duplicate_of": "later-core"},
    ]
    fingerprints = {game["id"]: "same-payoffs" for game in games}
    before = copy.deepcopy(games)
    assert _canonical_aliases(games, records, fingerprints) == {
        "first-control": "later-core",
        "core-copy": "later-core",
    }
    assert games == before
    # An older registry may have skipped the only core recipe. Keep its actual
    # evaluated control representative; never invent measurements or relabel it.
    records[1].update(status="duplicate", duplicate_of="first-control")
    records[2]["duplicate_of"] = "first-control"
    assert set(_canonical_aliases(games, records, fingerprints).values()) == {"first-control"}


@pytest.mark.parametrize("failure", ["outside", "different", "cycle", "mixed", "conflict"])
def test_invalid_duplicate_provenance_fails_closed(failure: str) -> None:
    """Aliases must refer to the same authenticated game with actual measurements."""
    games = [{"id": "a"}, {"id": "b"}]
    records = [
        {"game_id": "a", "status": "ok"},
        {"game_id": "b", "status": "duplicate", "duplicate_of": "a"},
    ]
    fingerprints = {"a": "same", "b": "same"}
    if failure == "outside":
        records[1]["duplicate_of"] = "outside"
    elif failure == "different":
        fingerprints["b"] = "different"
    elif failure == "cycle":
        records[0].update(status="duplicate", duplicate_of="b")
    elif failure == "mixed":
        records.append({"game_id": "b", "status": "ok"})
    else:
        records.append({"game_id": "b", "status": "duplicate", "duplicate_of": "other"})
    with pytest.raises(ValueError):
        _canonical_aliases(games, records, fingerprints)


def test_structured_recipe_ids_are_preserved(tmp_path: Path) -> None:
    make_campaign(tmp_path, structured=True)
    data = assemble_campaign(tmp_path, 3)
    assert "recipe-0-i0" in {g["id"] for g in data["games"]}


def test_unsupported_cells_are_complete_but_not_successes(tmp_path: Path) -> None:
    """Known unsupported targets still belong to the accounted-for matrix."""
    make_campaign(tmp_path)
    for path in tmp_path.glob("batch-*/sweep/shard-*/results.json"):
        value = json.loads(path.read_text())
        for row in value["records"]:
            if row["method"] == "PermutationSamplingSV":
                row.update(status="unsupported", nmse=None, mse=None)
        write(path, value)
    data = assemble_campaign(tmp_path, 3)
    rows = [r for r in data["records"] if r["method"] == "PermutationSamplingSV"]
    assert len(rows) == 8 and all(r["status"] == "unsupported" for r in rows)


@pytest.mark.parametrize("warm_cache", [False, True])
def test_running_shard_is_not_published(tmp_path: Path, *, warm_cache: bool) -> None:
    """A writer's held lock blocks authentication even if its checkpoint says complete."""
    make_campaign(tmp_path)
    cache = tmp_path / "normalization" if warm_cache else None
    if warm_cache:
        assemble_campaign(tmp_path, 3, cache_dir=cache)
    with (tmp_path / "batch-0/sweep/shard-000/.campaign.lock").open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            assemble_campaign(tmp_path, 3, cache_dir=cache)


@pytest.mark.parametrize("cached", [False, True])
def test_export_cli_writes_composite_provenance_and_global_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cached: bool
) -> None:
    """The actual CLI connects authenticated assembly to the website writer."""
    campaign, output = tmp_path / "campaign", tmp_path / "site"
    make_campaign(campaign)
    monkeypatch.setattr(
        sys,
        "argv",
        ["export_phase.py", str(campaign), str(output), "--through-phase", "3"]
        + (["--cache-dir", str(tmp_path / "cache")] if cached else []),
    )
    script = Path(__file__).resolve().parents[2] / "benchmark/export_phase.py"
    runpy.run_path(str(script), run_name="__main__")
    data = json.loads((output / "data.json").read_text())
    assert data["snapshot_id"] == identity(data["composition"])
    assert data["records"] == [] and data["presets"] == []
    assert data["record_count"] == 16
    records, presets = [], []
    for shard in data["record_shards"]:
        path = output / shard["file"]
        assert digest(path) == shard["sha256"]
        payload = json.loads(path.read_text())
        assert payload["snapshot_id"] == data["snapshot_id"]
        assert payload["target"] == shard["target"] and payload["count"] == shard["count"]
        decoded = [{} for _ in range(payload["count"])]
        for field, column in payload["columns"].items():
            for index, value in enumerate(column["values"]):
                if index not in column.get("missing", []):
                    decoded[index][field] = (
                        column["dictionary"][value]
                        if "dictionary" in column and value is not None
                        else value
                    )
        records.extend(decoded)
        presets.extend(decode_columns(payload["presets"]))
    expected = assemble_campaign(campaign, 3)["records"]

    def key(row: dict) -> tuple:
        return row["game_id"], row["method"], row["budget"], row["seed"]

    assert sorted(records, key=key) == sorted(expected, key=key)
    assert (output / "records.js").is_file()
    assert json.loads((output / "about.json").read_text())["snapshot_id"] == data["snapshot_id"]
    panel = next(p for p in presets if p["family"] is None and p["relative_budget"] is None)
    assert all(r["mean"] == 25.5 and r["median"] == 1.0 for r in panel["rows"])
