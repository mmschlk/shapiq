"""Transient retries cannot broaden or silently change a published experiment."""

from __future__ import annotations

import copy
import json
import runpy
import sys
from pathlib import Path

import pytest

from shapiq_benchmark.campaign import assemble_campaign
from shapiq_benchmark.campaign_recovery import merge_recovery
from shapiq_benchmark.runner import identity

from .test_campaign_export import make_campaign, write


def recovery_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Build two independently authenticated campaigns, with a cross-campaign alias."""
    parent, retry = tmp_path / "parent", tmp_path / "retry"
    original = make_campaign(parent)
    supplement = make_campaign(retry)
    # Recipe 2 was a network outage in the parent, not a quality rejection.
    path = parent / "batch-2/qualified-suite.json"
    qualified = json.loads(path.read_text())
    qualified["preparation_exclusions"][0].update(
        reason="preflight_failed",
        instances=[
            {"seed": s, "status": "failed", "error_type": "HTTPError", "reason": "Gateway Time-out"}
            for s in [0, 1]
        ],
    )
    write(path, qualified)
    decision = parent / "batch-2/qualification-decision.json"
    content = json.loads(decision.read_text())
    content["qualified_suite_sha256"] = identity(qualified)
    write(decision, content)
    write(parent / "batch-2/excluded.json", {"suite_sha256": identity(qualified)})

    # Keep only the independently evaluated first batch and rename its recipe.
    directory = retry / "batch-0"
    for path in directory.rglob("*.json"):
        write(path, json.loads(path.read_text().replace("recipe-0", "recipe-2")))
    requested = json.loads((directory / "suite.json").read_text())
    qualified = json.loads((directory / "qualified-suite.json").read_text())
    write(
        directory / "qualification-decision.json",
        {
            "requested_suite_sha256": identity(requested),
            "qualified_suite_sha256": identity(qualified),
            "source": original["source"],
        },
    )
    snapshot = json.loads((directory / "prepared/snapshot.json").read_text())
    snapshot["snapshot_id"] = identity({k: v for k, v in snapshot.items() if k != "snapshot_id"})
    write(directory / "prepared/snapshot.json", snapshot)
    allocation = json.loads((directory / "sweep/allocation.json").read_text())
    allocation["snapshot_id"] = snapshot["snapshot_id"]
    write(directory / "sweep/allocation.json", allocation)
    for path in directory.glob("sweep/shard-*/results.json"):
        result = json.loads(path.read_text())
        result["snapshot_id"] = snapshot["snapshot_id"]
        result["resume_key"] = identity(
            {k: v for k, v in result.items() if k not in ("records", "campaign", "resume_key")}
        )
        for row in result["records"]:
            row.update(
                status="duplicate",
                duplicate_of=row["game_id"].replace("recipe-2", "recipe-0"),
                nmse=None,
                mse=None,
                estimate=None,
            )
        write(path, result)
    supplement["batches"] = supplement["batches"][:1]
    supplement["batches"][0]["suite_sha256"] = identity(requested)
    write(retry / "campaign.json", supplement)
    write(retry / "jobs.json", {"plan_sha256": identity(supplement)})
    return parent, retry


def test_cross_campaign_alias_resolves_only_after_complete_authentication(tmp_path: Path) -> None:
    parent, retry = recovery_pair(tmp_path)
    with pytest.raises(ValueError, match="outside this publication"):
        assemble_campaign(retry, 3)
    data = assemble_campaign(parent, 3, supplements=(retry,))
    assert len(data["games"]) == 4 and len(data["records"]) == 16
    assert not data["suite"]["preparation_exclusions"]
    assert data["composition"]["supplements"][0]["retried_recipe_ids"] == ["recipe-2"]
    assert data["snapshot_id"] == identity(data["composition"])
    assert data["duplicate_games"]


def test_retry_still_authenticates_every_shard(tmp_path: Path) -> None:
    parent, retry = recovery_pair(tmp_path)
    path = retry / "batch-0/sweep/shard-000/results.json"
    result = json.loads(path.read_text())
    result["records"].pop()
    write(path, result)
    with pytest.raises(ValueError):
        assemble_campaign(parent, 3, supplements=(retry,))


def test_same_supplement_cannot_be_counted_twice(tmp_path: Path) -> None:
    parent, retry = recovery_pair(tmp_path)
    with pytest.raises(ValueError, match="repeated recovery"):
        assemble_campaign(parent, 3, supplements=(retry, retry))


def components() -> tuple[dict, dict, dict, dict]:
    spec = {"id": "wine", "dataset": "wine_quality", "n_players": 11}
    suite = {
        "game_seeds": [0, 1, 2, 3],
        "targets": [{"index": "SV", "order": 1}],
        "relative_budgets": [0.5, 1, 2],
        "protocol": {"version": 2},
        "method_parameters": {"OddSHAP": {"ridge": 0.001}},
        "duplicate_registry": "/same/registry",
    }
    excluded = {
        "spec": spec,
        "reason": "preflight_failed",
        "instances": [
            {"seed": s, "status": "failed", "error_type": "HTTPError", "reason": "Gateway Time-out"}
            for s in range(4)
        ],
    }
    data = {
        "games": [{"id": "original"}],
        "records": [],
        "coverage": [],
        "methods": {},
        "runs": {},
        "suite": {
            "budgets_by_game": {},
            "preparation_exclusions": [
                excluded,
                {"spec": {"id": "quality"}, "reason": "unstable_imputation"},
            ],
        },
        "composition": {},
    }
    extra = {
        "games": [{"id": "recovered"}],
        "records": [],
        "coverage": [],
        "methods": {},
        "runs": {},
        "suite": {"budgets_by_game": {}, "preparation_exclusions": []},
        "composition": {},
    }
    ctx = {
        "root": Path("/parent"),
        "campaign": {"source": {"hash": "same"}, "scripts": {"launch": "same"}},
        "requested": {("families", "wine"): {"spec": spec, "suite": suite}},
        "exclusions": {"wine": excluded},
        "fingerprints": {},
    }
    retry = copy.deepcopy(ctx)
    retry["root"] = Path("/retry")
    return data, ctx, extra, retry


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "scripts",
        "spec",
        "seed",
        "target",
        "budget",
        "protocol",
        "ridge",
        "registry",
        "unrelated",
        "scientific",
        "mixed",
        "overlap",
    ],
)
def test_incompatible_or_unrelated_recovery_rejected(change: str) -> None:
    data, ctx, extra, retry = components()
    request = retry["requested"][("families", "wine")]
    if change in ("source", "scripts"):
        retry["campaign"][change] = {"wrong": True}
    elif change == "spec":
        request["spec"]["n_players"] = 12
    elif change in ("seed", "target", "budget", "protocol", "ridge", "registry"):
        key = {
            "seed": "game_seeds",
            "target": "targets",
            "budget": "relative_budgets",
            "protocol": "protocol",
            "ridge": "method_parameters",
            "registry": "duplicate_registry",
        }[change]
        request["suite"][key] = "changed"
    elif change == "unrelated":
        retry["requested"] = {("families", "other"): request}
    elif change == "scientific":
        ctx["exclusions"]["wine"]["reason"] = "unstable_imputation"
    elif change == "mixed":
        ctx["exclusions"]["wine"]["instances"][0]["reason"] = "unstable_imputation"
    else:
        extra["games"] = copy.deepcopy(data["games"])
    with pytest.raises(ValueError):
        merge_recovery(data, ctx, extra, retry)


def test_new_scientific_exclusion_and_unrelated_exclusions_are_preserved() -> None:
    data, ctx, extra, retry = components()
    extra["games"] = []
    extra["suite"]["preparation_exclusions"] = [
        {"spec": {"id": "wine"}, "reason": "unstable_imputation"}
    ]
    merge_recovery(data, ctx, extra, retry)
    assert {r["spec"]["id"] for r in data["suite"]["preparation_exclusions"]} == {"wine", "quality"}
    assert all(
        r["reason"] == "unstable_imputation" for r in data["suite"]["preparation_exclusions"]
    )


def test_recovered_preflight_has_only_latest_outcome() -> None:
    data, ctx, extra, retry = components()
    data["suite"]["preparation_preflight"] = {
        "version": 2,
        "families": [{"id": "wine", "status": "excluded"}, {"id": "quality", "status": "excluded"}],
    }
    extra["suite"]["preparation_preflight"] = {
        "version": 2,
        "families": [{"id": "wine", "status": "qualified"}],
    }
    merge_recovery(data, ctx, extra, retry)
    assert data["suite"]["preparation_preflight"]["families"] == [
        {"id": "quality", "status": "excluded"},
        {"id": "wine", "status": "qualified"},
    ]


def test_supplement_can_be_explicitly_excluded_again(tmp_path: Path) -> None:
    parent, retry = recovery_pair(tmp_path)
    directory = retry / "batch-0"
    qualified = json.loads((directory / "qualified-suite.json").read_text())
    recipe = qualified["families"].pop()
    qualified["preparation_exclusions"] = [
        {
            "spec": recipe,
            "reason": "unstable_imputation",
            "maximum_seconds_per_instance": 28800,
            "instances": [],
        }
    ]
    write(directory / "qualified-suite.json", qualified)
    decision = json.loads((directory / "qualification-decision.json").read_text())
    decision["qualified_suite_sha256"] = identity(qualified)
    write(directory / "qualification-decision.json", decision)
    write(directory / "excluded.json", {"suite_sha256": identity(qualified)})
    data = assemble_campaign(parent, 3, supplements=(retry,))
    assert len(data["games"]) == 4
    assert data["suite"]["preparation_exclusions"][0]["reason"] == "unstable_imputation"
    assert (
        data["composition"]["supplements"][0]["composition"]["components"][0]["status"]
        == "excluded"
    )


def test_earlier_supplement_remains_available_in_later_phase_export(tmp_path: Path) -> None:
    parent, retry = recovery_pair(tmp_path)
    campaign = json.loads((parent / "campaign.json").read_text())
    campaign["batches"][0]["phase"] = 4
    write(parent / "campaign.json", campaign)
    write(parent / "jobs.json", {"plan_sha256": identity(campaign)})
    write(
        parent / "phase-4-inventory.json",
        json.loads((parent / "phase-3-inventory.json").read_text()),
    )
    data = assemble_campaign(parent, 4, supplements=(retry,))
    assert data["composition"]["through_phase"] == 4
    assert data["composition"]["supplements"][0]["composition"]["through_phase"] == 3


def test_cli_supplement_exports_joined_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, retry = recovery_pair(tmp_path)
    output = tmp_path / "site"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_phase.py",
            str(parent),
            str(output),
            "--through-phase",
            "3",
            "--supplement",
            str(retry),
        ],
    )
    script = Path(__file__).resolve().parents[2] / "benchmark/export_phase.py"
    runpy.run_path(str(script), run_name="__main__")
    data = json.loads((output / "data.json").read_text())
    assert data["composition"]["supplements"][0]["retried_recipe_ids"] == ["recipe-2"]
    assert data["snapshot_id"] == identity(data["composition"])
