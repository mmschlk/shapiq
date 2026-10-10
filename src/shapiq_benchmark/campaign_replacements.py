"""Replace complete method panels while preserving authenticated frozen games.

The manifest declares version 1, full expected execution ``source``, ``methods``,
and ``snapshots`` entries containing an original ``snapshot_id``, a ``snapshot``
path and explicit ``results`` paths. Paths resolve relative to the manifest.
Publication still requires an independent audit that all writers have stopped;
stable hashes and completion markers alone do not establish scheduler state.
"""

from __future__ import annotations

import itertools
import json
from typing import TYPE_CHECKING

from shapiq_benchmark.record_store import RecordStore
from shapiq_benchmark.report import merge_results
from shapiq_benchmark.results_io import read_results, result_inputs
from shapiq_benchmark.runner import digest, identity, load_snapshot

if TYPE_CHECKING:
    from pathlib import Path


def _require(condition: bool, message: str) -> None:  # noqa: FBT001
    if not condition:
        raise ValueError(message)


def _components(composition: dict) -> dict:
    """Index original snapshots, including transient-recovery supplements."""
    found = {}
    for part in composition["components"]:
        if part["status"] == "complete":
            found[part["snapshot_id"]] = (part, composition["inputs"])
    for supplement in composition.get("supplements", []):
        extra = _components(supplement["composition"])
        _require(not found.keys() & extra.keys(), "Repeated original snapshot")
        found.update(extra)
    return found


def _key(row: dict) -> tuple:
    return tuple(row[k] for k in ("game_id", "method", "budget", "seed"))


def replace_methods(data: dict, manifest_path: Path) -> dict:
    """Authenticate all replacements before returning a new, unsummarized report.

    Original sweep files are located beside each prepared snapshot, following the
    campaign layout. Their hashes must match the original composition, so even
    omitted public aliases retain their original capability/duplicate decisions.
    A later phase cannot publish until its complete replacements are supplied too.
    """
    _require(
        not data.get("record_shards") and "replacements" not in data["composition"],
        "Replacements require an assembled original campaign",
    )
    manifest_path = manifest_path.resolve()
    checked: dict[Path, str] = {}
    inputs = {}

    def authenticate(path: Path, label: str, expected: str | None = None) -> None:
        path = path.resolve()
        actual = digest(path)
        _require(expected is None or actual == expected, "Original campaign input changed")
        _require(path not in checked or checked[path] == actual, "Replacement input changed")
        checked[path] = actual
        inputs[label] = actual

    authenticate(manifest_path, "manifest")
    manifest = json.loads(manifest_path.read_text())
    _require(manifest.get("version") == 1, "Unknown replacement manifest version")
    methods, source = manifest["methods"], manifest["source"]
    _require(
        bool(methods)
        and len(set(methods)) == len(methods)
        and set(methods) <= data["methods"].keys(),
        "Replacement methods must be unique original methods",
    )
    _require(
        set(source) == set(data["snapshot_provenance"])
        and bool(source.get("source_sha256"))
        and source.get("source_dirty") is False
        and bool(source.get("git_commit")),
        "Replacement requires full expected provenance from a clean frozen revision",
    )
    originals = _components(data["composition"])
    entries = manifest["snapshots"]
    _require(
        len(entries) == len(originals)
        and {entry["snapshot_id"] for entry in entries} == originals.keys(),
        "Replacement snapshots must cover every original component",
    )
    public_ids = {game["id"] for game in data["games"]}
    aliases = {row["game_id"] for row in data.get("duplicate_games", [])}
    replacements = data["records"].fork() if isinstance(data["records"], RecordStore) else []
    runs, metadata = {}, {}
    for number, entry in enumerate(entries):
        prefix = f"snapshot-{number}"
        snapshot_path = (manifest_path.parent / entry["snapshot"]).resolve()
        if snapshot_path.is_dir():
            snapshot_path /= "snapshot.json"
        component, original_inputs = originals[entry["snapshot_id"]]
        batch = component["batch"]
        authenticate(
            snapshot_path,
            f"{prefix}/snapshot",
            original_inputs[f"{batch}/prepared/snapshot.json"],
        )
        snapshot, artifact_root = load_snapshot(snapshot_path, historical=True)
        _require(snapshot["snapshot_id"] == entry["snapshot_id"], "Wrong original snapshot")
        for index, (name, sha) in enumerate(snapshot["artifacts"].items()):
            authenticate(artifact_root / name, f"{prefix}/artifact-{index}", sha)
        games = {g["id"]: g for g in snapshot["games"]}
        _require(set(games) <= public_ids | aliases, "Snapshot contains unrelated games")
        suite = snapshot["suite"]
        expected_methods = {
            name: {
                "source_sha256": source["source_sha256"],
                "software_sha256": identity(source),
                "private": False,
                **(
                    {"parameters": suite["method_parameters"][name]}
                    if suite.get("method_parameters", {}).get(name)
                    else {}
                ),
            }
            for name in methods
        }
        _require(not metadata or metadata == expected_methods, "Replacement parameters differ")
        metadata = expected_methods
        baseline = {}
        batch_root = snapshot_path.parent.parent
        baseline_paths = sorted(
            path
            for path in original_inputs
            if path.startswith(f"{batch}/sweep/shard-") and path.endswith("/results.json")
        )
        for slot, relative in enumerate(baseline_paths):
            path = batch_root.parent / relative
            for index, companion in enumerate(result_inputs(path)):
                relative_input = str(companion.resolve().relative_to(batch_root.parent))
                authenticate(
                    companion,
                    f"{prefix}/baseline-{slot}-{index}",
                    original_inputs[relative_input],
                )
            for row in read_results(path)["records"]:
                if row["method"] in methods:
                    _require(_key(row) not in baseline, "Duplicate original cell")
                    baseline[_key(row)] = (row["status"], row.get("duplicate_of"))
        paths = [(manifest_path.parent / p).resolve() for p in entry["results"]]
        _require(bool(paths) and len(paths) == len(set(paths)), "Empty or repeated result paths")
        seen = set()
        for slot, path in enumerate(paths):
            for index, companion in enumerate(result_inputs(path)):
                authenticate(companion, f"{prefix}/result-{slot}-{index}")
            result = read_results(path)
            expected = {
                "schema_version": snapshot["schema_version"],
                "snapshot_id": snapshot["snapshot_id"],
                "snapshot_provenance": snapshot["provenance"],
                "suite": suite,
                "games": snapshot["games"],
                "coverage": snapshot.get("coverage", []),
                "methods": expected_methods,
            }
            _require(
                all(result.get(k) == v for k, v in expected.items()), "Replacement panel changed"
            )
            run = result["run_provenance"]
            _require(
                {k: v for k, v in run.items() if k != "execution"} == source,
                "Unexpected replacement software",
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
                "Replacement resume identity changed",
            )
            selected = run["execution"]["game_ids"]
            _require(
                bool(selected)
                and len(selected) == len(set(selected))
                and set(selected) <= games.keys(),
                "Invalid replacement game selection",
            )
            planned = {
                (gid, method, budget, seed)
                for gid in selected
                for method, budget, seed in itertools.product(
                    methods, suite["budgets_by_game"][gid], suite["seeds"]
                )
            }
            cells = [_key(row) for row in result["records"]]
            _require(
                len(cells) == len(set(cells)) and set(cells) == planned and not seen & planned,
                "Missing, repeated, or unexpected replacement cells",
            )
            _require(
                result.get("campaign")
                == {"planned": len(planned), "completed": len(planned), "complete": True},
                "Replacement campaign is incomplete",
            )
            for row in result["records"]:
                old = baseline.get(_key(row))
                _require(old is not None, "Replacement has no original cell")
                for status in ("unsupported", "duplicate"):
                    _require(
                        (row["status"] == status) == (old[0] == status),
                        "Replacement changed capability or duplicate decision",
                    )
                _require(
                    row.get("duplicate_of") == old[1],
                    "Replacement changed duplicate representative",
                )
                _require(
                    row["status"] != "duplicate" or row["game_id"] in aliases,
                    "Public game lacks replacement measurements",
                )
            seen.update(planned)
        _require(seen == baseline.keys(), "Replacement does not cover the complete snapshot")
        panel = merge_results(paths)  # Existing score checks and public sanitization.
        replacements.extend(row for row in panel["records"] if row["game_id"] in public_ids)
        runs.update(panel["runs"])
    if isinstance(data["records"], RecordStore):
        records = data["records"].replaced(methods, replacements)
    else:
        expected_cells = {_key(row) for row in data["records"] if row["method"] in methods}
        _require(
            len(replacements) == len(expected_cells)
            and {_key(row) for row in replacements} == expected_cells,
            "Replacement does not cover the complete public panel",
        )
        records = [row for row in data["records"] if row["method"] not in methods] + replacements
    _require(all(digest(path) == sha for path, sha in checked.items()), "Replacement input changed")
    used_runs = {row["run_id"] for row in records}
    composition = {
        **data["composition"],
        "replacements": {"version": 1, "methods": methods, "source": source, "inputs": inputs},
    }
    return {
        **{k: v for k, v in data.items() if k not in ("presets", "record_count")},
        "methods": {**data["methods"], **metadata},
        "runs": {k: v for k, v in {**data["runs"], **runs}.items() if k in used_runs},
        "records": records,
        "composition": composition,
        "snapshot_id": identity(composition),
    }
