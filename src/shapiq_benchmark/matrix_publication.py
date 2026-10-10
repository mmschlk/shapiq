"""Compose reviewed matrix waves with their original published history on disk.

The externally pinned plan and independent receipts authorize the input panel;
this function neither contacts the scheduler nor approves publication. Presets
must subsequently be recomputed over the complete, deduplicated union.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

from shapiq_benchmark.campaign import _canonical_aliases, _collect_campaign
from shapiq_benchmark.duplicates import payoff_fingerprint, remove_aliases
from shapiq_benchmark.published import (
    _read,
    _require,
    _sources,
    merge_method_catalogs,
    read_published,
)
from shapiq_benchmark.report import ROW_FIELDS
from shapiq_benchmark.results_io import result_inputs
from shapiq_benchmark.runner import digest, identity, load_snapshot

if TYPE_CHECKING:
    from shapiq_benchmark.record_store import RecordStore

CONFIG_FIELDS = (
    "game_seeds",
    "seeds",
    "relative_budgets",
    "methods",
    "method_parameters",
    "targets",
    "min_players",
    "min_signal_ratio",
)


def _recipe_key(kind: str, spec: dict, config: dict) -> str:
    return identity(
        {
            "kind": kind,
            "spec": {k: v for k, v in spec.items() if k != "id"},
            "config": {k: config[k] for k in CONFIG_FIELDS},
        }
    )


def _gate(path: Path, sha256: str) -> tuple[dict, dict, dict, dict, dict]:
    """Authenticate every completion/bridge receipt before opening raw measurements."""
    path = path.resolve()
    checked: dict[str, str] = {str(path): sha256}
    plan = _read(path, sha256)

    def read(pin: dict) -> dict:
        location = (path.parent / pin["path"]).resolve()
        _require(
            str(location) not in checked or checked[str(location)] == pin["sha256"],
            "Conflicting publication input pins",
        )
        value = _read(location, pin["sha256"])
        checked[str(location)] = pin["sha256"]
        return value

    def closure(receipt: dict) -> dict:
        inputs = receipt.get("authenticated_inputs")
        _require(isinstance(inputs, dict) and bool(inputs), "Receipt has no authenticated closure")
        inputs = receipt["authenticated_inputs"]
        for name, expected in inputs.items():
            location = Path(name)
            _require(
                location.is_absolute() and location.resolve() == location,
                "Receipt input paths must be absolute and resolved",
            )
            _require(name not in checked or checked[name] == expected, "Conflicting receipt pins")
            if name not in checked:
                _require(digest(location) == expected, "Reviewed input changed")
                checked[name] = expected
        return inputs

    def reviewed(pin: dict, *, independent: bool = True) -> dict:
        receipt = read(pin)
        _require(
            receipt.get("status") == "PASS" and receipt.get("complete") is True,
            "Final independent review is missing or incomplete",
        )
        if independent:
            _require(
                receipt.get("blocking_issues") == [], "Independent review has unresolved findings"
            )
        closure(receipt)
        return receipt

    _require(
        type(plan.get("version")) is int
        and plan["version"] == 1
        and type(plan.get("through_wave")) is int
        and plan["through_wave"] > 0,
        "Invalid matrix publication plan",
    )
    _require(
        (path.parent / plan["campaign"]["path"]).resolve().name == "campaign.json",
        "Publication plan must pin the collected campaign.json",
    )
    campaign = read(plan["campaign"])
    _require(identity(campaign["source"]) == identity(plan["source"]), "Wrong matrix source")
    phases = sorted({b["phase"] for b in campaign["batches"] if b["phase"] <= plan["through_wave"]})
    _require(
        bool(phases)
        and phases[-1] == plan["through_wave"]
        and all(type(wave["phase"]) is int for wave in plan["waves"])
        and [wave["phase"] for wave in plan["waves"]] == phases,
        "Missing, repeated, or unexpected reviewed waves",
    )
    audited = {}
    for wave in plan["waves"]:
        audit = reviewed(wave["audit"], independent=False)
        review = reviewed(wave["review"])
        expected = {b["id"] for b in campaign["batches"] if b["phase"] == wave["phase"]}
        actual = [b["batch_id"] for b in audit["batches"]]
        _require(
            type(audit["phase"]) is int
            and type(review["phase"]) is int
            and audit["phase"] == review["phase"] == wave["phase"]
            and audit["plan_sha256"] == review["plan_sha256"] == identity(campaign)
            and identity(audit["source"]) == identity(campaign["source"])
            and len(actual) == len(set(actual))
            and set(actual) == expected,
            "Wave audit differs from the exact campaign, source, phase, or batches",
        )
        audit_path = str((path.parent / wave["audit"]["path"]).resolve())
        _require(
            review["audit_sha256"] == wave["audit"]["sha256"]
            and review["authenticated_inputs"].get(audit_path) == wave["audit"]["sha256"]
            and all(
                review["authenticated_inputs"].get(p) == h
                for p, h in audit["authenticated_inputs"].items()
            ),
            "Independent review does not bind the complete wave audit",
        )
        audited.update(review["authenticated_inputs"])
    ledger = read(plan["reuse"])
    _require(
        ledger.get("status") == "PASS"
        and type(ledger.get("version")) is int
        and ledger["version"] == 1
        and campaign["reuse"]["sha256"] == plan["reuse"]["sha256"]
        and Path(campaign["reuse"]["path"]).resolve()
        == (path.parent / plan["reuse"]["path"]).resolve(),
        "Wrong historical reuse ledger",
    )
    closure(ledger)
    history = plan["history"]
    index = read(history["index"])
    bridge = reviewed(history["bridge"])
    archives = {}
    for name, pin in history["archives"].items():
        location = (path.parent / pin["path"]).resolve()
        _require(
            location.name == name
            and bridge["authenticated_inputs"].get(str(location)) == pin["sha256"],
            "Historical archive is not bound by the public-row bridge",
        )
        archives[name] = pin["sha256"]
    _require(
        set(archives) == {p["archive"] for p in ledger["panels"]}
        and bridge["history_index_sha256"] == history["index"]["sha256"]
        and bridge["history_shards"] == history["shards"]
        and bridge["reuse_sha256"] == plan["reuse"]["sha256"]
        and bridge["archives"] == archives
        and bridge.get("scope") == "public-measurements-run-ids-and-provenance"
        and bridge.get("preserved_run_ids") is True
        and bridge.get("preserved_run_provenance") is True
        and type(bridge.get("record_count")) is int
        and bridge["record_count"] == index["record_count"]
        and type(bridge.get("game_count")) is int
        and bridge["game_count"] == len(index["games"])
        and bridge.get("comparison_fields") == sorted([*ROW_FIELDS, "run_id"]),
        "Historical bridge binds another panel",
    )
    index_path = (path.parent / history["index"]["path"]).resolve()
    for location, expected in {
        index_path: history["index"]["sha256"],
        **{index_path.parent / n: h for n, h in history["shards"].items()},
    }.items():
        _require(
            bridge["authenticated_inputs"].get(str(location)) == expected,
            "Historical bridge omits published input bytes",
        )
    equivalence = reviewed(plan["source_equivalence"])
    _require(
        equivalence["history_index_sha256"] == history["index"]["sha256"]
        and equivalence["historical_methods_sha256"] == identity(index["methods"])
        and identity(equivalence["matrix_source"]) == identity(campaign["source"])
        and equivalence.get("scope") == "estimator-code-runtime-and-parameters"
        and identity(equivalence.get("approved_historical_methods", {}))
        == identity(index["methods"])
        and bool(plan["required_corrections"])
        and identity(equivalence.get("required_corrections", {}))
        == identity(plan["required_corrections"]),
        "Source equivalence binds another history or matrix source",
    )
    for method, allowed in plan["required_corrections"].items():
        _require(
            method in index["methods"]
            and all(
                allowed.get(software) == source
                for software, source in _sources(index["methods"][method]).items()
            ),
            "Historical corrected method contains an obsolete source",
        )
    return plan, campaign, ledger, checked, audited


def _raw_membership(root: Path, campaign: dict, phase: int, approved: dict) -> None:
    """Check reviewed file membership before collection opens measurement journals."""

    def require(path: Path) -> Path:
        path = path.resolve()
        _require(
            path.is_relative_to(root) and str(path) in approved,
            "Required campaign input lacks independent review",
        )
        return path

    def metadata(path: Path) -> dict:
        path = require(path)
        return _read(path, approved[str(path)])

    require(root / "campaign.json")
    require(root / "jobs.json")
    require(root / f"phase-{phase}-inventory.json")
    for batch in campaign["batches"]:
        if batch["phase"] > phase:
            continue
        directory = root / batch["id"]
        require(directory / "suite.json")
        require(directory / "qualification-decision.json")
        qualified = metadata(directory / "qualified-suite.json")
        if not qualified.get("families") and not qualified.get("games"):
            require(directory / "excluded.json")
            continue
        snapshot = metadata(directory / "prepared/snapshot.json")
        for name, sha in snapshot["artifacts"].items():
            artifact = require(directory / "prepared" / name)
            _require(approved[str(artifact)] == sha, "Reviewed artifact pin differs from snapshot")
        allocation = metadata(directory / "sweep/allocation.json")
        for slot in range(min(len(allocation["cpus"]), len(allocation["game_ids"]))):
            manifest = require(directory / "sweep" / f"shard-{slot:03}" / "results.json")
            for companion in result_inputs(manifest):
                require(companion)


def _history_fingerprints(data: dict, ledger: dict) -> tuple[dict, dict]:
    """Use only imported public games, binding their effective roles to frozen tables."""
    public = {g["id"]: g for g in data["games"]}
    found, fingerprints, recipes = set(), {}, {}
    for entry in ledger["canonical_snapshots"]:
        path = Path(entry["path"])
        _require(
            ledger["authenticated_inputs"].get(str(path)) == entry["sha256"],
            "Historical snapshot lacks ledger authentication",
        )
        snapshot, root = load_snapshot(path, historical=True)
        _require(snapshot["snapshot_id"] == entry["snapshot_id"], "Historical snapshot changed")
        for kind in ("families", "games"):
            for spec in snapshot["suite"].get(kind, []):
                key = (kind, spec["id"])
                value = _recipe_key(kind, spec, snapshot["suite"])
                _require(
                    key not in recipes or recipes[key] == value, "Historical recipe ID collision"
                )
                recipes[key] = value
        selected = set(entry["completed_game_ids"]) & public.keys()
        for game in snapshot["games"]:
            if game["id"] not in selected:
                continue
            current = public[game["id"]]
            _require(
                game["id"] not in found
                and all(
                    identity({"v": current[k]}) == identity({"v": game[k]})
                    for k in ("family", "stratum", "index", "order", "n_players")
                )
                and current.get("metadata", {}).get("game_quality", {}).get("role", "unqualified")
                == entry["effective_roles"][game["id"]],
                "Historical game identity or effective role changed",
            )
            found.add(game["id"])
            artifact = root / game["artifact"]
            _require(
                ledger["authenticated_inputs"].get(str(artifact))
                == snapshot["artifacts"][game["artifact"]],
                "Historical artifact lacks ledger authentication",
            )
            fingerprint = payoff_fingerprint(game, root)
            if fingerprint is not None:
                fingerprints[game["id"]] = fingerprint
    _require(
        found == public.keys(), "Published history is not completely represented in the ledger"
    )
    return fingerprints, recipes


def compose_matrix(
    plan_path: Path,
    *,
    plan_sha256: str,
    record_store: RecordStore,
    cache_dir: Path | None = None,
) -> dict:
    """Return an authenticated, globally deduplicated panel in an empty owned store.

    Keep the store owner open. This supplies rows to bounded selectors/summaries;
    the legacy report writer does not support a complete disk-backed panel.
    """
    _require(len(record_store) == 0, "Matrix composition requires an empty record store")
    plan_path = plan_path.resolve()
    plan, campaign, ledger, checked, audited = _gate(plan_path, plan_sha256)
    root = (plan_path.parent / plan["campaign"]["path"]).resolve().parent
    _raw_membership(root, campaign, plan["through_wave"], audited)
    history_pin = plan["history"]["index"]
    history = read_published(
        (plan_path.parent / history_pin["path"]).resolve(),
        index_sha256=history_pin["sha256"],
        shard_sha256=plan["history"]["shards"],
        record_store=record_store.fork(),
    )
    historical_fingerprints, old_recipes = _history_fingerprints(history, ledger)
    matrix, context = _collect_campaign(
        root, plan["through_wave"], cache_dir=cache_dir, record_store=record_store.fork()
    )
    _require(
        identity(context["campaign"]) == identity(campaign),
        "Collected campaign differs from reviewed plan",
    )
    for name, sha in context["inputs"].items():
        _require(
            audited.get(str(root / name)) == sha, "Collected input was not independently reviewed"
        )
    for (kind, name), requested in context["requested"].items():
        _require(
            (kind, name) not in old_recipes
            or old_recipes[kind, name] == _recipe_key(kind, requested["spec"], requested["suite"]),
            "Recipe ID reuses a different historical configuration",
        )
    old_ids = {g["id"] for g in history["games"]}
    _require(
        not old_ids.intersection(g["id"] for g in matrix["games"]),
        "Historical/matrix game ID collision",
    )
    for key in ("relative_budgets", "seeds", "game_seeds", "min_signal_ratio"):
        _require(
            identity({"v": history["suite"].get(key)}) == identity({"v": matrix["suite"].get(key)}),
            f"Incompatible historical/matrix {key}",
        )
    methods = merge_method_catalogs(history["methods"], matrix["methods"])
    runs = copy.deepcopy(history["runs"])
    for name, run in matrix["runs"].items():
        _require(
            name not in runs or identity(runs[name]) == identity(run), "Conflicting original run ID"
        )
        runs[name] = run
    reused = [r for r in campaign["reuse"]["recipes"] if r["phase"] <= plan["through_wave"]]
    entries = {e["key"]: e for e in ledger["entries"]}
    _require(
        len(entries) == len(ledger["entries"]) and len({r["key"] for r in reused}) == len(reused),
        "Repeated semantic reuse keys",
    )
    for entry in reused:
        evidence = entries[entry["key"]]
        _require(
            evidence["complete"] is True
            and evidence["spec"]["id"] == entry["id"]
            and entry["key"] == _recipe_key(evidence["kind"], evidence["spec"], evidence["config"]),
            "Current matrix reuse differs from complete semantic ledger entry",
        )
    union = history["records"]
    union.extend(matrix["records"])
    suite = copy.deepcopy(matrix["suite"])
    old_grids = history["suite"].get(
        "budgets_by_game", dict.fromkeys(old_ids, history["suite"]["budgets"])
    )
    suite["budgets_by_game"] = {**old_grids, **suite["budgets_by_game"]}
    suite["budgets"] = sorted({b for grid in suite["budgets_by_game"].values() for b in grid})
    composition = {
        "version": 1,
        "kind": "matrix-with-published-history",
        "plan_sha256": plan_sha256,
        "through_wave": plan["through_wave"],
        "historical_snapshot_id": history["snapshot_id"],
        "historical_composition_sha256": identity(history.get("composition", {})),
        "historical_suite": history["suite"],
        "matrix_composition": matrix["composition"],
        "reuse_sha256": plan["reuse"]["sha256"],
        "reused_recipes": reused,
        "historical_deferred_recipes": len(ledger["deferred_reuse"]),
        "planned_budgets_by_game": copy.deepcopy(suite["budgets_by_game"]),
    }
    data = {
        "schema_version": 1,
        "snapshot_provenance": matrix["snapshot_provenance"],
        "suite": suite,
        "games": [*history["games"], *matrix["games"]],
        "methods": methods,
        "runs": runs,
        "records": union,
        "coverage": [*history.get("coverage", []), *matrix["coverage"]],
        "composition": composition,
    }
    aliases = _canonical_aliases(
        data["games"], union, {**historical_fingerprints, **context["fingerprints"]}
    )
    remove_aliases(data, aliases)
    prior_aliases = {a["game_id"]: a["duplicate_of"] for a in history.get("duplicate_games", [])}
    _require(
        not prior_aliases.keys() & {g["id"] for g in data["games"]}, "Historical alias ID collision"
    )
    for alias, canonical in prior_aliases.items():
        _require(canonical in old_ids, "Historical alias has no published canonical")
        aliases[alias] = aliases.get(canonical, canonical)
    data["duplicate_games"] = [
        {"game_id": a, "duplicate_of": c} for a, c in sorted(aliases.items())
    ]
    composition["aliases"] = data["duplicate_games"]
    used_runs = {r["run_id"] for r in union}
    data["runs"] = {key: run for key, run in runs.items() if key in used_runs}
    _require(
        all(digest(Path(p)) == sha for p, sha in checked.items()), "Composition input bytes changed"
    )
    _require(len(record_store) == 0, "Matrix composition destination changed")
    record_store.extend(union)
    data.update(records=record_store, snapshot_id=identity(composition))
    return data
