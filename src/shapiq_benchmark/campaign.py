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

import numpy as np

from shapiq_benchmark.campaign_backend import load_backend, merge_backend
from shapiq_benchmark.campaign_recovery import merge_recovery
from shapiq_benchmark.campaign_replacements import replace_methods
from shapiq_benchmark.duplicates import payoff_fingerprint, remove_aliases
from shapiq_benchmark.publication_cache import normalization_policy, normalized_batch
from shapiq_benchmark.quality import (
    clustering_diagnostics,
    model_validation_check,
    payoff_diagnostics,
)
from shapiq_benchmark.report import merge_results, public_preparation
from shapiq_benchmark.results_io import read_results, result_inputs
from shapiq_benchmark.runner import digest, identity, load_snapshot

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq_benchmark.record_store import RecordStore


def _require(condition: bool, message: str) -> None:  # noqa: FBT001 -- assertion helper
    if not condition:
        raise ValueError(message)


def _historical_quality(game: dict, root: Path) -> dict:
    """Supplement authenticated tables with score-independent publication checks.

    This annotates the report, never the frozen snapshot or original payoffs.
    Missing stochastic qualification is explicit rather than assumed to pass.
    The clustering resolution check also applies to already-qualified tables.
    """
    metadata = game.get("metadata", {})
    if game.get("oracle", "table") != "table":
        return {}
    cluster = metadata.get("class") == (
        "shapiq_games.benchmark.unsupervised_cluster.base.ClusterExplanation"
    )
    existing = metadata.get("game_quality")
    if existing and not cluster:
        return {}
    with np.load(root / game["artifact"], allow_pickle=False) as archive:
        quality = (
            copy.deepcopy(existing)
            if existing
            else payoff_diagnostics(archive["values"], game["n_players"])
        )
        clustering_diagnostics(archive["values"], metadata, quality)
    diagnostics = {"game_quality": quality}
    if existing:
        return diagnostics
    if metadata.get("quality", {}).get("validation"):
        gate = model_validation_check(metadata)
        diagnostics["model_validation_gate"] = gate
        if not gate["passed"]:
            quality["control_reasons"].append("predictor_below_validation_dummy")
    if metadata.get("stochastic_frozen") and not metadata.get("imputation_stability"):
        quality["control_reasons"].append("stochastic_oracle_stability_unqualified")
    if quality["control_reasons"]:
        quality["role"] = "control"
    return diagnostics


def _canonical_aliases(games: list[dict], records: list[dict], fingerprints: dict) -> dict:
    """Keep an evaluated core representative of each exact game, without relabeling it."""
    marked = {}
    for row in records:
        if row["status"] == "duplicate":
            alias, canonical = row["game_id"], row["duplicate_of"]
            _require(
                marked.setdefault(alias, canonical) == canonical, "Conflicting duplicate markers"
            )
    for alias, canonical in marked.items():
        _require(canonical in fingerprints, "Duplicate refers to a game outside this publication")
        _require(
            alias in fingerprints and fingerprints[alias] == fingerprints[canonical],
            "Duplicate marker disagrees with authenticated payoff tables",
        )
        _require(canonical not in marked, "Duplicate canonical was not evaluated")
    measured = {row["game_id"] for row in records if row["status"] != "duplicate"}
    _require(not measured.intersection(marked), "Game mixes duplicate and evaluated cells")
    groups = {}
    for game in games:
        if game["id"] in fingerprints:
            groups.setdefault(fingerprints[game["id"]], []).append(game)
    aliases = {}
    for group in groups.values():
        candidates = [game for game in group if game["id"] in measured]
        _require(bool(candidates), "Exact game has no evaluated representative")
        canonical = min(
            candidates,
            key=lambda game: (
                game.get("metadata", {}).get("game_quality", {}).get("role") != "core",
                game["id"],
            ),
        )["id"]
        aliases.update({game["id"]: canonical for game in group if game["id"] != canonical})
    return aliases


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
    """Qualification may exclude unchanged recipes, never alter shared settings."""
    additions = {"families", "games", "preparation_preflight", "preparation_exclusions"}
    _require(
        {k: v for k, v in original.items() if k not in additions}
        == {k: v for k, v in qualified.items() if k not in additions},
        "Qualification changed declared settings",
    )
    requested_rows = [*original.get("families", []), *original.get("games", [])]
    kept_rows = [*qualified.get("families", []), *qualified.get("games", [])]
    requested = {spec["id"]: spec for spec in requested_rows}
    kept = {spec["id"]: spec for spec in kept_rows}
    excluded = {
        row["spec"]["id"]: row["spec"] for row in qualified.get("preparation_exclusions", [])
    }
    _require(
        len(requested) == len(requested_rows)
        and len(kept) == len(kept_rows)
        and len(excluded) == len(qualified.get("preparation_exclusions", []))
        and not kept.keys() & excluded.keys()
        and {**kept, **excluded} == requested,
        "Qualification must account for every unchanged recipe exactly once",
    )


def _batch_results(
    directory: Path,
    snapshot: dict,
    source: dict,
    read: Callable[[Path], dict],
    *,
    normalize: Callable[[list[Path]], dict] = merge_results,
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
            **(
                {"parameters": snapshot["suite"]["method_parameters"][name]}
                if snapshot["suite"].get("method_parameters", {}).get(name)
                else {}
            ),
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
        return normalize(paths)


def _public_batch(
    directory: Path,
    snapshot: dict,
    source: dict,
    read: Callable[[Path], dict],
    root: Path,
    inputs: dict[str, str],
    cache_dir: Path | None,
    policy: dict,
) -> dict:
    """Normalize only after the unchanged complete-shard validator succeeds."""

    def normalize(paths: list[Path]) -> dict:
        def build() -> dict:
            panel = merge_results(paths)
            public_games = {game["id"]: game for game in panel["games"]}
            fingerprints = {}
            for game in snapshot["games"]:
                public_games[game["id"]].setdefault("metadata", {}).update(
                    _historical_quality(game, directory / "prepared")
                )
                fingerprint = payoff_fingerprint(game, directory / "prepared")
                if fingerprint is not None:
                    fingerprints[game["id"]] = fingerprint
            return {"panel": panel, "fingerprints": fingerprints}

        if cache_dir is None:
            return build()
        # Use the hashes captured by validation, not newly accepted
        # hashes. This private closure does not alter public provenance.
        authenticated = {
            str(root / name): sha
            for name, sha in inputs.items()
            if (root / name).is_relative_to(directory)
        }
        authenticated.update(
            {str(directory / "prepared" / name): sha for name, sha in snapshot["artifacts"].items()}
        )
        return normalized_batch(cache_dir, authenticated, policy, build)

    return _batch_results(directory, snapshot, source, read, normalize=normalize)


def _collect_campaign(
    root: Path,
    through_phase: int,
    *,
    earlier_phase: bool = False,
    backend: dict | None = None,
    cache_dir: Path | None = None,
    record_store: RecordStore | None = None,
) -> tuple[dict, dict]:
    """Return sanitized report data with an authenticated composition manifest.

    Every batch through the selected phase must be complete or explicitly excluded.
    The frozen campaign's own source identifies historical measurements; this
    exporting checkout need not be identical. Compute summary statistics on the
    united panel, never by averaging summaries of its constituent batches.
    """
    _require(record_store is None or len(record_store) == 0, "Assembly record store must be empty")
    root = root.resolve()
    inputs = {}
    policy = normalization_policy() if cache_dir is not None else {}

    def read(path: Path) -> dict:
        _require(path.resolve().is_relative_to(root), "Campaign input escapes its directory")
        content = path.read_bytes()
        inputs[str(path.relative_to(root))] = hashlib.sha256(content).hexdigest()
        value = json.loads(content)
        if value.get("storage_format") == "snapshot-journal-v1":
            for companion in result_inputs(path)[1:]:
                _require(companion.is_relative_to(root), "Result companion escapes campaign")
                inputs[str(companion.relative_to(root))] = digest(companion)
            value = read_results(path)
        return value

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
    if earlier_phase and batches:
        through_phase = max(batch["phase"] for batch in batches)
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
            "method_parameters",
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
        "records": record_store if record_store is not None else [],
    }
    seen, components, preflight = set(), [], None
    requested, exclusions = {}, {}
    game_fingerprints = {}
    for batch in batches:
        directory = root / batch["id"]
        _require(
            directory.resolve().is_relative_to(root)
            and directory.resolve() == Path(batch["directory"]).resolve(),
            "Batch directory differs from submitted plan",
        )
        original = read(directory / "suite.json")
        _require(
            identity(original) == batch["suite_sha256"], "Batch suite differs from submitted plan"
        )
        if batch["phase"] == through_phase:
            _require(
                original["phase_plan"]["inventory_sha256"] == identity(inventory["phase_plan"])
                and original["protocol"] == inventory["protocol"],
                "Phase inventory differs from submitted suites",
            )
        for kind in ("families", "games"):
            for spec in original.get(kind, []):
                key = (kind, spec["id"])
                _require(
                    key not in requested, "Duplicate game IDs: requested recipe across batches"
                )
                requested[key] = {"spec": spec, "suite": original}
        if backend is not None and batch["id"] == backend["entry"]["batch_id"]:
            _require(original == backend["suite"], "Superseded suite changed")
            for key in (
                "relative_budgets",
                "seeds",
                "game_seeds",
                "methods",
                "targets",
                "min_players",
                "min_signal_ratio",
                "method_parameters",
            ):
                _require(original.get(key) == inventory.get(key), f"Batch has incompatible {key}")
            components.append(
                {
                    "batch": batch["id"],
                    "phase": batch["phase"],
                    "status": "superseded",
                    "requested_suite_sha256": identity(original),
                    "reason": backend["entry"]["reason"],
                }
            )
            continue
        qualified = read(directory / "qualified-suite.json")
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
        exclusions.update(
            {row["spec"]["id"]: row for row in qualified.get("preparation_exclusions", [])}
        )
        for key in (
            "relative_budgets",
            "seeds",
            "game_seeds",
            "methods",
            "targets",
            "min_players",
            "min_signal_ratio",
            "method_parameters",
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
            snapshot, _ = load_snapshot(directory / "prepared", historical=True)
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
            normalized = _public_batch(
                directory, snapshot, source, read, root, inputs, cache_dir, policy
            )
            panel = normalized["panel"]
            game_fingerprints.update(normalized["fingerprints"])
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
    return data, {
        "root": root,
        "campaign": campaign,
        "requested": requested,
        "exclusions": exclusions,
        "fingerprints": game_fingerprints,
        "inputs": inputs,
    }


def assemble_campaign(
    root: Path,
    through_phase: int,
    *,
    supplements: tuple[Path, ...] = (),
    replacements: Path | None = None,
    backend_supersession: Path | None = None,
    cache_dir: Path | None = None,
    record_store: RecordStore | None = None,
) -> dict:
    """Authenticate complete campaigns, resolving aliases only after joining retries.

    An optional empty ``record_store`` spills cumulative rows to disk. Its owner
    must stay open while consuming the returned records through bounded selectors;
    ``iter_summaries`` supports stored records; the report writer still requires
    in-memory record lists.
    """
    backend = (
        load_backend(backend_supersession, root, through_phase) if backend_supersession else None
    )
    data, context = _collect_campaign(
        root, through_phase, backend=backend, cache_dir=cache_dir, record_store=record_store
    )
    contexts = [context]
    for supplement in supplements:
        extra, extra_context = _collect_campaign(
            supplement,
            through_phase,
            earlier_phase=True,
            cache_dir=cache_dir,
            record_store=record_store.fork() if record_store is not None else None,
        )
        merge_recovery(data, context, extra, extra_context)
        contexts.append(extra_context)
    if replacements is not None:
        # Authorize recorded alias markers while retaining all rows until the
        # final union is deduplicated. Recovery already uses corrected methods.
        aliases = _canonical_aliases(data["games"], data["records"], context["fingerprints"])
        data["duplicate_games"] = [
            {"game_id": game, "duplicate_of": canonical} for game, canonical in aliases.items()
        ]
        data = replace_methods(data, replacements)
    if backend is not None:
        extra, extra_context = _collect_campaign(
            backend["root"],
            through_phase,
            earlier_phase=True,
            cache_dir=cache_dir,
            record_store=record_store.fork() if record_store is not None else None,
        )
        merge_backend(data, context, extra, extra_context, backend)
        contexts.append(extra_context)
    _require(bool(data["games"]), "No qualified games are available for publication")
    remove_aliases(
        data, _canonical_aliases(data["games"], data["records"], context["fingerprints"])
    )
    if replacements is not None:
        used_runs = {row["run_id"] for row in data["records"]}
        data["runs"] = {key: run for key, run in data["runs"].items() if key in used_runs}
    # Recheck every component after the potentially lengthy combined export.
    for component in contexts:
        _require(
            all(
                digest(component["root"] / path) == sha for path, sha in component["inputs"].items()
            ),
            "Campaign inputs changed during composition",
        )
    data["snapshot_id"] = identity(data["composition"])
    return data
