"""Small real campaigns exercise reviewed history/matrix union and its gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shapiq_benchmark import matrix_publication
from shapiq_benchmark.campaign import assemble_campaign
from shapiq_benchmark.matrix_publication import compose_matrix
from shapiq_benchmark.record_store import RecordStore
from shapiq_benchmark.report import ROW_FIELDS, write_report
from shapiq_benchmark.runner import digest, identity
from shapiq_benchmark.summary import summarize
from tests.shapiq_benchmark.test_campaign_export import make_campaign, write


def read(path: Path) -> dict:
    """Load a small test document."""
    return json.loads(path.read_text())


def pin(path: Path) -> dict:
    """Pin local immutable evidence just as an external reviewed plan does."""
    return {"path": str(path.resolve()), "sha256": digest(path)}


def hashes(root: Path) -> dict:
    """Capture small fixture inputs while excluding untracked lock files."""
    return {str(p.resolve()): digest(p) for p in root.rglob("*") if p.suffix in {".json", ".npz"}}


def campaign(root: Path, prefix: str, source_hash: str, factor: int) -> dict:
    """Reuse the actual campaign fixture, giving it independent IDs and provenance."""
    value = make_campaign(root)
    source = {**value["source"], "source_sha256": source_hash}
    for path in root.rglob("*.json"):
        text = path.read_text().replace("recipe-", prefix + "-recipe-")
        path.write_text(text)
    inventory = read(root / "phase-3-inventory.json")
    inventory["method_parameters"] = {}
    write(root / "phase-3-inventory.json", inventory)
    for batch in value["batches"]:
        directory = Path(batch["directory"])
        original, qualified = (
            read(directory / "suite.json"),
            read(directory / "qualified-suite.json"),
        )
        original["method_parameters"] = qualified["method_parameters"] = {}
        write(directory / "suite.json", original)
        write(directory / "qualified-suite.json", qualified)
        batch["suite_sha256"] = identity(original)
        write(
            directory / "qualification-decision.json",
            {
                "requested_suite_sha256": identity(original),
                "qualified_suite_sha256": identity(qualified),
                "source": source,
            },
        )
        if (directory / "excluded.json").exists():
            write(directory / "excluded.json", {"suite_sha256": identity(qualified)})
            continue
        snapshot = read(directory / "prepared/snapshot.json")
        for name in snapshot["artifacts"]:
            path = directory / "prepared" / name
            with np.load(path) as artifact:
                values = artifact["values"] * factor
            np.savez(path, values=values)
            snapshot["artifacts"][name] = digest(path)
        snapshot["provenance"] = source
        snapshot["suite"]["method_parameters"] = {}
        snapshot.pop("snapshot_id")
        snapshot["snapshot_id"] = identity(snapshot)
        write(directory / "prepared/snapshot.json", snapshot)
        allocation = read(directory / "sweep/allocation.json")
        allocation["snapshot_id"] = snapshot["snapshot_id"]
        write(directory / "sweep/allocation.json", allocation)
        for path in (directory / "sweep").glob("shard-*/results.json"):
            result = read(path)
            result.update(
                snapshot_id=snapshot["snapshot_id"],
                snapshot_provenance=source,
                suite=snapshot["suite"],
            )
            result["run_provenance"].update(source)
            for method in result["methods"].values():
                method.update(source_sha256=source_hash, software_sha256=identity(source))
            result["resume_key"] = identity(
                {k: v for k, v in result.items() if k not in {"records", "campaign", "resume_key"}}
            )
            write(path, result)
    value["source"] = source
    write(root / "campaign.json", value)
    write(root / "jobs.json", {"plan_sha256": identity(value)})
    return value


def fixture(root: Path, *, aliases: bool = False) -> tuple[Path, dict]:
    """Seal exact bridges, ledger and final wave reviews over a tiny complete matrix."""
    old_root, new_root = root / "historical", root / "matrix"
    campaign(old_root, "old", "a" * 64, 1)
    new = campaign(new_root, "new", "b" * 64, 1 if aliases else 7)
    if aliases:
        for result_path in new_root.glob("*/sweep/shard-*/results.json"):
            result = read(result_path)
            for row in result["records"]:
                row.update(
                    status="duplicate",
                    duplicate_of=row["game_id"].replace("new-", "old-"),
                    nmse=None,
                    mse=None,
                )
                row.pop("estimate", None)
            write(result_path, result)
    history = assemble_campaign(old_root, 3)
    # Historical unaffected methods may already span two recorded software versions.
    method = history["methods"]["PermutationSamplingSV"]
    versions = {method.pop("software_sha256"): method.pop("source_sha256"), "c" * 64: "d" * 64}
    method["source_versions"] = versions
    site = root / "site"
    write_report(history, site, public=True, compact=True)
    index = read(site / "data.json")
    archive = root / "original.zip"
    archive.write_bytes(
        b"Authenticated reproduction evidence placeholder; no scientific data in this fixture."
    )
    snapshots = []
    for path in old_root.glob("*/prepared/snapshot.json"):
        snap = read(path)
        snapshots.append(
            {
                **pin(path),
                "snapshot_id": snap["snapshot_id"],
                "completed_game_ids": [g["id"] for g in snap["games"]],
                "effective_roles": {
                    g["id"]: g["metadata"].get("game_quality", {}).get("role", "unqualified")
                    for g in history["games"]
                    if g["id"] in {v["id"] for v in snap["games"]}
                },
            }
        )
    # All historical recipes remain available. Only one is counted as current reuse.
    old_suite = read(old_root / "batch-0/suite.json")
    spec = old_suite["families"][0]
    key = matrix_publication._recipe_key("families", spec, old_suite)
    ledger = {
        "version": 1,
        "status": "PASS",
        "entries": [
            {"key": key, "kind": "families", "spec": spec, "config": old_suite, "complete": True}
        ],
        "deferred_reuse": [{"spec": {"id": "legacy"}}],
        "canonical_snapshots": snapshots,
        "panels": [{"archive": archive.name}],
        "authenticated_inputs": hashes(old_root),
    }
    ledger_path = root / "reuse.json"
    write(ledger_path, ledger)
    new["reuse"] = {**pin(ledger_path), "recipes": [{"id": spec["id"], "key": key, "phase": 3}]}
    write(new_root / "campaign.json", new)
    write(new_root / "jobs.json", {"plan_sha256": identity(new)})
    closure = hashes(new_root)
    audit = {
        "status": "PASS",
        "complete": True,
        "phase": 3,
        "plan_sha256": identity(new),
        "source": new["source"],
        "batches": [{"batch_id": b["id"]} for b in new["batches"]],
        "authenticated_inputs": {
            p: h
            for p, h in closure.items()
            if Path(p).name not in {"jobs.json", "phase-3-inventory.json"}
        },
    }
    audit_path = root / "audit.json"
    write(audit_path, audit)
    review_path = root / "review.json"
    write(
        review_path,
        {
            "status": "PASS",
            "complete": True,
            "phase": 3,
            "plan_sha256": identity(new),
            "blocking_issues": [],
            "audit_sha256": digest(audit_path),
            "authenticated_inputs": {**closure, str(audit_path): digest(audit_path)},
        },
    )
    shards = {d["file"]: d["sha256"] for d in index["record_shards"]}
    bridge_path = root / "bridge.json"
    history_inputs = {
        str(site / "data.json"): digest(site / "data.json"),
        **{str(site / n): h for n, h in shards.items()},
        str(archive): digest(archive),
        str(ledger_path): digest(ledger_path),
    }
    write(
        bridge_path,
        {
            "status": "PASS",
            "complete": True,
            "history_index_sha256": digest(site / "data.json"),
            "history_shards": shards,
            "reuse_sha256": digest(ledger_path),
            "archives": {archive.name: digest(archive)},
            "scope": "public-measurements-run-ids-and-provenance",
            "blocking_issues": [],
            "preserved_run_ids": True,
            "preserved_run_provenance": True,
            "record_count": index["record_count"],
            "game_count": len(index["games"]),
            "comparison_fields": sorted([*ROW_FIELDS, "run_id"]),
            "authenticated_inputs": history_inputs,
        },
    )
    equivalence_path = root / "equivalence.json"
    write(
        equivalence_path,
        {
            "status": "PASS",
            "complete": True,
            "history_index_sha256": digest(site / "data.json"),
            "historical_methods_sha256": identity(index["methods"]),
            "scope": "estimator-code-runtime-and-parameters",
            "blocking_issues": [],
            "approved_historical_methods": index["methods"],
            "required_corrections": {
                "KernelSHAP": matrix_publication._sources(index["methods"]["KernelSHAP"])
            },
            "matrix_source": new["source"],
            "authenticated_inputs": history_inputs,
        },
    )
    plan = {
        "version": 1,
        "campaign": pin(new_root / "campaign.json"),
        "source": new["source"],
        "through_wave": 3,
        "reuse": pin(ledger_path),
        "source_equivalence": pin(equivalence_path),
        "required_corrections": {
            "KernelSHAP": matrix_publication._sources(index["methods"]["KernelSHAP"])
        },
        "history": {
            "index": pin(site / "data.json"),
            "shards": shards,
            "bridge": pin(bridge_path),
            "archives": {archive.name: pin(archive)},
        },
        "waves": [{"phase": 3, "audit": pin(audit_path), "review": pin(review_path)}],
    }
    path = root / "publication.json"
    write(path, plan)
    return path, history


def test_union_preserves_history_and_exact_global_summaries(tmp_path: Path) -> None:
    """Use real collection, source union, full grids and global summary mathematics."""
    path, history = fixture(tmp_path)
    with RecordStore(tmp_path / "records.sqlite") as store:
        data = compose_matrix(
            path, plan_sha256=digest(path), record_store=store, cache_dir=tmp_path / "cache"
        )
        rows = list(store)
        assert len(rows) == 32 and len(data["games"]) == 8
        assert identity({"rows": rows[:16]}) == identity({"rows": history["records"]})
        assert all(data["runs"][key] == run for key, run in history["runs"].items())
        assert len(data["methods"]["KernelSHAP"]["source_versions"]) == 2
        assert len(data["methods"]["PermutationSamplingSV"]["source_versions"]) == 3
        assert len(data["composition"]["reused_recipes"]) == 1
        assert data["composition"]["historical_deferred_recipes"] == 1
        assert len(data["composition"]["planned_budgets_by_game"]) == 8
        assert str(tmp_path) not in json.dumps({k: v for k, v in data.items() if k != "records"})
        assert summarize(data, bootstrap_draws=4) == summarize(
            {**data, "records": rows}, bootstrap_draws=4
        )
        assert data["snapshot_id"] == identity(data["composition"])
    # Cache hits must preserve the complete union, not cached batch summaries.
    with RecordStore(tmp_path / "again.sqlite") as store:
        again = compose_matrix(
            path, plan_sha256=digest(path), record_store=store, cache_dir=tmp_path / "cache"
        )
        assert identity({"rows": list(store)}) == identity({"rows": rows})
        assert again["snapshot_id"] == data["snapshot_id"]


@pytest.mark.parametrize(
    "kind",
    [
        "pending",
        "phase",
        "audit_hash",
        "closure",
        "batch",
        "source",
        "bridge",
        "equivalence",
        "missing_wave",
        "campaign_name",
    ],
)
def test_gate_rejects_before_raw_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """A copied PASS or incomplete review never permits raw panel collection."""
    path, _history = fixture(tmp_path)
    plan = read(path)
    pin_name = "review"
    if kind in {"batch", "source"}:
        pin_name = "audit"
    entry = plan["waves"][0][pin_name]
    if kind == "bridge":
        entry = plan["history"]["bridge"]
    elif kind == "equivalence":
        entry = plan["source_equivalence"]
    receipt_path = Path(entry["path"])
    receipt = read(receipt_path)
    if kind == "pending":
        receipt["complete"] = False
    elif kind == "phase":
        receipt["phase"] = 9
    elif kind == "audit_hash":
        receipt["audit_sha256"] = "0" * 64
    elif kind == "closure":
        receipt["authenticated_inputs"].pop(
            next(iter(read(Path(plan["waves"][0]["audit"]["path"]))["authenticated_inputs"]))
        )
    elif kind == "batch":
        receipt["batches"].pop()
    elif kind == "source":
        receipt["source"]["source_sha256"] = "c" * 64
    elif kind == "bridge":
        receipt["history_index_sha256"] = "0" * 64
    elif kind == "equivalence":
        receipt["historical_methods_sha256"] = "0" * 64
    elif kind == "missing_wave":
        plan["waves"] = []
    elif kind == "campaign_name":
        original = Path(plan["campaign"]["path"])
        alternate = original.with_name("other-campaign.json")
        alternate.write_bytes(original.read_bytes())
        plan["campaign"] = pin(alternate)
    write(receipt_path, receipt)
    entry["sha256"] = digest(receipt_path)
    write(path, plan)

    def forbidden(*args, **kwargs):
        message = "Raw collection ran before review gates"
        raise AssertionError(message)

    monkeypatch.setattr(matrix_publication, "_collect_campaign", forbidden)
    with RecordStore(tmp_path / "records.sqlite") as store, pytest.raises(ValueError):
        try:
            compose_matrix(path, plan_sha256=digest(path), record_store=store)
        finally:
            assert len(store) == 0


@pytest.mark.parametrize(
    "kind",
    [
        "role",
        "collision",
        "recipe",
        "unreviewed",
        "late_change",
        "missing_canonical",
        "mixed_alias",
    ],
)
def test_late_validation_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Identity, effective roles, alias evidence and stable hashes guard the final append."""
    path, _history = fixture(tmp_path)
    original = matrix_publication._collect_campaign
    fingerprint = matrix_publication._history_fingerprints

    if kind == "role":

        def wrong_role(data, ledger):
            data["games"][0]["metadata"]["game_quality"]["role"] = "wrong"
            return fingerprint(data, ledger)

        monkeypatch.setattr(matrix_publication, "_history_fingerprints", wrong_role)
    else:

        def changed(*args, **kwargs):
            data, context = original(*args, **kwargs)
            if kind == "collision":
                data["games"][0]["id"] = "old-recipe-0-i0-sv-1"
            elif kind == "recipe":
                requested = next(iter(context["requested"].values()))
                context["requested"][("families", "old-recipe-0")] = {
                    **requested,
                    "spec": {**requested["spec"], "n_players": 12},
                }
            elif kind == "unreviewed":
                context["inputs"]["unknown.json"] = "0" * 64
            elif kind == "late_change":
                path.write_text(path.read_text() + " ")
            else:
                rows = list(data["records"])
                alias = rows[0]["game_id"]
                canonical = alias.replace("new-", "old-") if kind == "mixed_alias" else "absent"
                if kind == "mixed_alias":
                    old_fingerprints, _ = fingerprint(_history, read(tmp_path / "reuse.json"))
                    context["fingerprints"][alias] = old_fingerprints[canonical]
                for row in rows:
                    if row["game_id"] == alias and (kind != "mixed_alias" or row is rows[0]):
                        row.update(status="duplicate", duplicate_of=canonical)
                replacement = data["records"].fork()
                replacement.extend(rows)
                data["records"] = replacement
            return data, context

        monkeypatch.setattr(matrix_publication, "_collect_campaign", changed)
    with RecordStore(tmp_path / "records.sqlite") as store, pytest.raises(ValueError):
        try:
            compose_matrix(path, plan_sha256=digest(path), record_store=store)
        finally:
            assert len(store) == 0


def test_duplicates_resolve_only_after_historical_union(tmp_path: Path) -> None:
    """Actual raw duplicate markers can target historical tables absent from the new campaign."""
    path, history = fixture(tmp_path, aliases=True)
    with RecordStore(tmp_path / "rows.sqlite") as store:
        data = compose_matrix(path, plan_sha256=digest(path), record_store=store)
        assert identity({"rows": list(store)}) == identity({"rows": history["records"]})
        assert data["games"] == history["games"]
        assert len(data["duplicate_games"]) == 4
        assert all(a["duplicate_of"].startswith("old-") for a in data["duplicate_games"])
        assert len(data["composition"]["planned_budgets_by_game"]) == 8
        assert len(data["suite"]["budgets_by_game"]) == 4
        assert data["runs"] == history["runs"]


@pytest.mark.parametrize(
    "kind", ["correction", "scope", "blocker", "run_ids", "count", "comparison_fields"]
)
def test_independent_claims_are_specific(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Repinned generic PASS claims cannot replace exact scientific bridge assertions."""
    path, _ = fixture(tmp_path)
    plan = read(path)
    entry = (
        plan["source_equivalence"]
        if kind in {"correction", "scope", "blocker"}
        else plan["history"]["bridge"]
    )
    receipt_path = Path(entry["path"])
    receipt = read(receipt_path)
    if kind == "correction":
        # Even matching plan/receipt correction claims reject historical obsolete rows.
        plan["required_corrections"] = receipt["required_corrections"] = {
            "KernelSHAP": {"c" * 64: "d" * 64}
        }
    elif kind == "scope":
        receipt["scope"] = "source names looked plausible"
    elif kind == "blocker":
        receipt["blocking_issues"] = ["comparison not completed"]
    elif kind == "run_ids":
        receipt["preserved_run_ids"] = False
    elif kind == "count":
        receipt["record_count"] += 1
    elif kind == "comparison_fields":
        receipt["comparison_fields"].remove("run_id")
    write(receipt_path, receipt)
    entry["sha256"] = digest(receipt_path)
    write(path, plan)

    def forbidden(*args, **kwargs):
        message = "Collection ran before required independent assertions"
        raise AssertionError(message)

    monkeypatch.setattr(matrix_publication, "_collect_campaign", forbidden)
    with RecordStore(tmp_path / "rows.sqlite") as store, pytest.raises(ValueError):
        compose_matrix(path, plan_sha256=digest(path), record_store=store)


@pytest.mark.parametrize("missing", ["records", "artifact", "journal"])
def test_missing_reviewed_input_stops_before_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    """Even consistently repinned incomplete closures cannot authorize collection."""
    path, _ = fixture(tmp_path)
    plan = read(path)
    audit_path = Path(plan["waves"][0]["audit"]["path"])
    review_path = Path(plan["waves"][0]["review"]["path"])
    audit, review = read(audit_path), read(review_path)
    suffix = {"records": "results.json", "artifact": "game-0.npz", "journal": "jobs.json"}[missing]
    removed = next(p for p in review["authenticated_inputs"] if p.endswith(suffix))
    audit["authenticated_inputs"].pop(removed, None)
    review["authenticated_inputs"].pop(removed)
    write(audit_path, audit)
    plan["waves"][0]["audit"] = pin(audit_path)
    review["audit_sha256"] = digest(audit_path)
    review["authenticated_inputs"][str(audit_path)] = digest(audit_path)
    write(review_path, review)
    plan["waves"][0]["review"] = pin(review_path)
    write(path, plan)

    def forbidden(*args, **kwargs):
        message = "Unreviewed input reached collection"
        raise AssertionError(message)

    monkeypatch.setattr(matrix_publication, "_collect_campaign", forbidden)
    with (
        RecordStore(tmp_path / "rows.sqlite") as store,
        pytest.raises(ValueError, match="independent review"),
    ):
        compose_matrix(path, plan_sha256=digest(path), record_store=store)
