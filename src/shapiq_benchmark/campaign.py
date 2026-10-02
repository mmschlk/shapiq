"""Authenticate disjoint campaign batches and assemble one public comparison panel."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import itertools
import json
import math
import re
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

from shapiq_benchmark.report import merge_results, public_preparation
from shapiq_benchmark.runner import digest, identity, load_snapshot

if TYPE_CHECKING:
    from collections.abc import Callable


def _require(condition: bool, message: str) -> None:  # noqa: FBT001 -- assertion helper
    if not condition:
        raise ValueError(message)


def _prepared_suite(suite: dict, games: list[dict]) -> dict:
    """Recompute only fields added by freezing."""
    expected = dict(suite)
    expected["budgets_by_game"] = {
        game["id"]: sorted({math.ceil(r * game["n_players"]) for r in suite["relative_budgets"]})
        for game in games
    }
    expected["budgets"] = sorted({b for grid in expected["budgets_by_game"].values() for b in grid})
    return expected


def _game_ids(suite: dict) -> set[str]:
    """Match preparation's IDs without constructing models or evaluating games."""
    seeds = suite["game_seeds"]
    enumerated = {
        f"{spec['id']}-i{seed}-{re.sub(r'[^a-zA-Z0-9_-]', '-', target['index']).lower()}-{target['order']}"
        for spec, seed, target in itertools.product(
            suite.get("families", []), seeds, suite["targets"]
        )
    }
    structured = {
        f"{spec['id']}-i{seed}" for spec, seed in itertools.product(suite.get("games", []), seeds)
    }
    return enumerated | structured


def _qualification(original: dict, qualified: dict) -> None:
    """Qualification may exclude families, never alter recipes or shared settings."""
    additions = {"families", "preparation_preflight", "preparation_exclusions"}
    _require(
        {k: v for k, v in original.items() if k not in additions}
        == {k: v for k, v in qualified.items() if k not in additions},
        "Qualification changed declared settings or structured games",
    )
    requested = {spec["id"]: spec for spec in original.get("families", [])}
    kept = {spec["id"]: spec for spec in qualified.get("families", [])}
    excluded = {
        row["spec"]["id"]: row["spec"] for row in qualified.get("preparation_exclusions", [])
    }
    _require(
        len(requested) == len(original.get("families", []))
        and len(kept) == len(qualified.get("families", []))
        and len(excluded) == len(qualified.get("preparation_exclusions", []))
        and not kept.keys() & excluded.keys()
        and {**kept, **excluded} == requested,
        "Qualification must account for every unchanged recipe exactly once",
    )


def _batch_results(
    directory: Path, snapshot: dict, source: dict, read: Callable[[Path], dict]
) -> dict:
    """Verify stopped shards, their exact allocation, complete cells and source identity."""
    allocation = read(directory / "sweep/allocation.json")
    ids, cpus = allocation["game_ids"], allocation["cpus"]
    _require(
        allocation["snapshot_id"] == snapshot["snapshot_id"]
        and len(ids) == len(set(ids)) == len(snapshot["games"])
        and set(ids) == {g["id"] for g in snapshot["games"]}
        and bool(cpus)
        and len(cpus) == len(set(cpus)),
        "Sweep allocation differs from the authenticated snapshot",
    )
    methods = {
        name: {
            "source_sha256": source["source_sha256"],
            "software_sha256": identity(source),
            "private": False,
        }
        for name in snapshot["suite"]["methods"]
    }
    paths = [
        directory / "sweep" / f"shard-{slot:03}" / "results.json"
        for slot in range(min(len(cpus), len(ids)))
    ]
    _require(
        set((directory / "sweep").glob("shard-*/results.json")) == set(paths),
        "Missing or unexpected result shards",
    )
    with ExitStack() as stack:
        for slot, path in enumerate(paths):
            lock = stack.enter_context((path.parent / ".campaign.lock").open())
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            result = read(path)
            expected = {
                "schema_version": snapshot["schema_version"],
                "snapshot_id": snapshot["snapshot_id"],
                "snapshot_provenance": source,
                "suite": snapshot["suite"],
                "games": snapshot["games"],
                "methods": methods,
                "coverage": snapshot.get("coverage", []),
            }
            _require(
                all(result.get(k) == v for k, v in expected.items()),
                "Result panel or method source differs from its authenticated snapshot",
            )
            provenance = result["run_provenance"]
            selected = ids[slot :: len(cpus)]
            _require(
                {k: v for k, v in provenance.items() if k != "execution"} == source
                and provenance["execution"]["game_ids"]
                == [g["id"] for g in snapshot["games"] if g["id"] in selected],
                "Result execution source or shard game allocation changed",
            )
            _require(
                identity(
                    {
                        k: v
                        for k, v in result.items()
                        if k not in ("records", "campaign", "resume_key")
                    }
                )
                == result["resume_key"],
                "Result resume identity changed",
            )
            suite = snapshot["suite"]
            planned = {
                (game, method, budget, seed)
                for game in selected
                for method in methods
                for budget in suite["budgets_by_game"][game]
                for seed in suite["seeds"]
            }
            measured = [
                tuple(row[k] for k in ("game_id", "method", "budget", "seed"))
                for row in result["records"]
            ]
            _require(
                len(measured) == len(set(measured)) and set(measured) == planned,
                "Result has missing, duplicate, or unexpected cells",
            )
            _require(
                result.get("campaign")
                == {"planned": len(planned), "completed": len(planned), "complete": True},
                "Result campaign is incomplete",
            )
        return merge_results(paths)


def assemble_campaign(root: Path, through_phase: int) -> dict:
    """Return sanitized report data with an authenticated composition manifest.

    Every batch through the selected phase must be complete or explicitly excluded.
    The frozen campaign's own source identifies historical measurements; this
    exporting checkout need not be identical. Compute summary statistics on the
    united panel, never by averaging summaries of its constituent batches.
    """
    root = root.resolve()
    inputs = {}

    def read(path: Path) -> dict:
        _require(path.resolve().is_relative_to(root), "Campaign input escapes its directory")
        content = path.read_bytes()
        inputs[str(path.relative_to(root))] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    campaign, journal = read(root / "campaign.json"), read(root / "jobs.json")
    _require(
        identity(campaign) == journal["plan_sha256"], "Campaign differs from submission journal"
    )
    source = campaign["source"]
    _require(
        source.get("source_dirty") is False and bool(source.get("git_commit")),
        "Campaign source is not a clean frozen revision",
    )
    batches = [b for b in campaign["batches"] if b["phase"] <= through_phase]
    _require(
        bool(batches) and through_phase in {b["phase"] for b in campaign["batches"]},
        "Unknown or empty campaign phase",
    )
    inventory = read(root / f"phase-{through_phase}-inventory.json")
    suite = {
        k: copy.deepcopy(inventory[k])
        for k in (
            "name",
            "protocol",
            "phase_plan",
            "min_players",
            "min_signal_ratio",
            "relative_budgets",
            "seeds",
            "game_seeds",
            "methods",
        )
        if k in inventory
    }
    suite.update(budgets=[], budgets_by_game={}, preparation_exclusions=[])
    data: dict = {
        "schema_version": 1,
        "snapshot_provenance": source,
        "suite": suite,
        "games": [],
        "coverage": [],
        "methods": {},
        "runs": {},
        "records": [],
    }
    seen, components, preflight = set(), [], None
    for batch in batches:
        directory = root / batch["id"]
        _require(
            directory.resolve().is_relative_to(root)
            and directory.resolve() == Path(batch["directory"]).resolve(),
            "Batch directory differs from submitted plan",
        )
        original, qualified = (
            read(directory / "suite.json"),
            read(directory / "qualified-suite.json"),
        )
        _require(
            identity(original) == batch["suite_sha256"], "Batch suite differs from submitted plan"
        )
        if batch["phase"] == through_phase:
            _require(
                original["phase_plan"]["inventory_sha256"] == identity(inventory["phase_plan"])
                and original["protocol"] == inventory["protocol"],
                "Phase inventory differs from submitted suites",
            )
        decision = read(directory / "qualification-decision.json")
        _require(
            decision
            == {
                "requested_suite_sha256": identity(original),
                "qualified_suite_sha256": identity(qualified),
                "source": source,
            },
            "Qualification decision changed",
        )
        _qualification(original, qualified)
        for key in (
            "relative_budgets",
            "seeds",
            "game_seeds",
            "methods",
            "targets",
            "min_players",
            "min_signal_ratio",
        ):
            _require(original.get(key) == inventory.get(key), f"Batch has incompatible {key}")
        sanitized = public_preparation(qualified)
        suite["preparation_exclusions"].extend(sanitized.get("preparation_exclusions", []))
        current = sanitized.get("preparation_preflight")
        if current:
            if preflight is None:
                preflight = {**current, "families": []}
            _require(
                {k: v for k, v in current.items() if k != "families"}
                == {k: v for k, v in preflight.items() if k != "families"},
                "Preparation gate differs across batches",
            )
            preflight["families"].extend(current.get("families", []))
        component = {
            "batch": batch["id"],
            "phase": batch["phase"],
            "qualified_suite_sha256": identity(qualified),
        }
        if not qualified.get("families") and not qualified.get("games"):
            marker = read(directory / "excluded.json")
            _require(
                marker["suite_sha256"] == identity(qualified),
                "Exclusion marker differs from qualified suite",
            )
            component["status"] = "excluded"
        else:
            _require(
                not (directory / "excluded.json").exists(), "Nonempty batch has an exclusion marker"
            )
            snapshot, _ = load_snapshot(directory / "prepared")
            _require(
                read(directory / "prepared/snapshot.json") == snapshot,
                "Snapshot changed during authentication",
            )
            _require(
                snapshot["provenance"] == source
                and snapshot["suite"] == _prepared_suite(qualified, snapshot["games"]),
                "Prepared snapshot differs from qualification or source",
            )
            ids = [g["id"] for g in snapshot["games"]]
            _require(
                len(ids) == len(set(ids)) and set(ids) == _game_ids(qualified),
                "Prepared games do not match all qualified recipe instances and targets",
            )
            _require(not seen.intersection(ids), "Duplicate game IDs across campaign batches")
            seen.update(ids)
            panel = _batch_results(directory, snapshot, source, read)
            for key in ("games", "coverage", "records"):
                data[key].extend(panel[key])
            for name, method in panel["methods"].items():
                _require(
                    name not in data["methods"] or data["methods"][name] == method,
                    "Conflicting method sources",
                )
                data["methods"][name] = method
            data["runs"].update(panel["runs"])
            suite["budgets_by_game"].update(snapshot["suite"]["budgets_by_game"])
            component.update(status="complete", snapshot_id=snapshot["snapshot_id"])
        components.append(component)
    _require(bool(data["games"]), "No qualified games are available for publication")
    suite["budgets"] = sorted({b for grid in suite["budgets_by_game"].values() for b in grid})
    if preflight is not None:
        suite["preparation_preflight"] = preflight
    _require(
        all(digest(root / path) == value for path, value in inputs.items()),
        "Campaign inputs changed during export",
    )
    composition = {
        "version": 1,
        "through_phase": through_phase,
        "campaign_sha256": identity(campaign),
        "components": components,
        "inputs": inputs,
    }
    data.update(composition=composition, snapshot_id=identity(composition))
    return data
