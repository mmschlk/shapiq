"""Authenticate one GPU-to-CPU recovery without rewriting its failed campaign.

The explicit manifest binds an audited terminal/guard receipt and a separately
submitted CPU campaign. Scheduler evidence is saved JSON, so exports stay portable.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from shapiq_benchmark.runner import digest, identity

if TYPE_CHECKING:
    from pathlib import Path


def _require(condition: bool, message: str) -> None:  # noqa: FBT001
    if not condition:
        raise ValueError(message)


def load_backend(manifest_path: Path, root: Path, phase: int) -> dict:
    """Validate explicit supersession evidence before skipping any original batch."""
    manifest_path = manifest_path.resolve()
    checked, inputs = {}, {}

    def read(entry: dict, label: str) -> dict:
        path = (manifest_path.parent / entry["path"]).resolve()
        sha = digest(path)
        _require(sha == entry["sha256"], "Backend recovery input hash changed")
        checked[path] = inputs[label] = sha
        value = json.loads(path.read_text())
        _require(
            "identity" not in entry or identity(value) == entry["identity"],
            "Backend identity changed",
        )
        return value

    manifest = read({"path": str(manifest_path), "sha256": digest(manifest_path)}, "manifest")
    _require(
        manifest.get("version") == 1 and len(manifest["supersessions"]) == 1,
        "Expected one explicit backend supersession",
    )
    original_path = (manifest_path.parent / manifest["original_campaign"]["path"]).resolve()
    _require(original_path == root.resolve() / "campaign.json", "Wrong original campaign")
    campaign = read(manifest["original_campaign"], "original-campaign")
    entry = manifest["supersessions"][0]
    batches = [b for b in campaign["batches"] if b["id"] == entry["batch_id"]]
    _require(
        len(batches) == 1 and batches[0]["phase"] == entry["phase"] <= phase,
        "Supersession does not identify a selected original batch",
    )
    batch = batches[0]
    suite_path = (manifest_path.parent / entry["original_suite"]["path"]).resolve()
    _require(suite_path == root.resolve() / batch["id"] / "suite.json", "Wrong superseded suite")
    suite = read(entry["original_suite"], "original-suite")
    _require(identity(suite) == batch["suite_sha256"], "Superseded suite differs from campaign")
    _require(
        not list((suite_path.parent / "sweep").glob("shard-*/results.json")),
        "Cannot supersede an already evaluated batch",
    )
    guard, terminal = read(entry["guard"], "guard"), read(entry["terminal"], "terminal")
    _require(
        entry["reason"] == "sustained_low_gpu_utilization"
        and guard.get("reason") == "sustained_low_utilization"
        and guard.get("guard_returncode") == 75
        and guard.get("threshold_percent") == 80
        and guard.get("utilization_verified") is False
        and guard.get("command_completed") is False,
        "Recovery requires the recorded GPU utilization guard failure",
    )
    _require(
        terminal["batch_id"] == batch["id"]
        and terminal["original_campaign_sha256"] == inputs["original-campaign"]
        and terminal["original_suite_sha256"] == inputs["original-suite"]
        and terminal["guard_sha256"] == inputs["guard"]
        and terminal["state"] == "FAILED"
        and terminal["exit_code"] == "75:0",
        "Terminal receipt does not bind the failed batch and guard",
    )
    rows = [
        line.split("|")
        for line in terminal["accounting"].splitlines()
        if line.split("|")[0] == terminal["job_id"]
    ]
    _require(
        len(rows) == 1 and rows[0][2:4] == ["FAILED", "75:0"],
        "Original GPU task did not terminate with the guard exit code",
    )
    cancelled = terminal["superseded_jobs"]
    _require(
        bool(cancelled) and set(cancelled) == set(entry["superseded_pending_jobs"]),
        "Superseded dependent jobs need exact cancellation receipts",
    )
    for job, receipt in cancelled.items():
        rows = [
            line.split("|")
            for line in receipt["accounting"].splitlines()
            if line.split("|")[0] == job
        ]
        _require(
            receipt["state"] == "CANCELLED"
            and len(rows) == 1
            and rows[0][2].split()[0] == "CANCELLED",
            "Superseded job remains active",
        )
    retry = entry["recovery_campaign"]
    retry_root = (manifest_path.parent / retry["root"]).resolve()
    _require(retry_root != root.resolve(), "Recovery must have its own campaign")
    recovery = read({**retry, "path": str(retry_root / "campaign.json")}, "recovery-campaign")
    _require(
        recovery["source"] == retry["source"]
        and set(recovery["source"]) == set(campaign["source"])
        and recovery["source"].get("source_dirty") is False
        and bool(recovery["source"].get("git_commit"))
        and recovery["scripts"] == campaign["scripts"],
        "Recovery source or preparation code changed",
    )
    _require(
        len(recovery["batches"]) == 1 and recovery["batches"][0]["phase"] == entry["phase"],
        "Recovery campaign must contain only the superseding batch",
    )
    return {
        "entry": entry,
        "suite": suite,
        "root": retry_root,
        "campaign": recovery,
        "checked": checked,
        "inputs": inputs,
    }


def merge_backend(data: dict, context: dict, extra: dict, retry: dict, backend: dict) -> None:
    """Join fully authenticated recovery cells, retaining their own software versions."""
    entry, original = backend["entry"], backend["suite"]
    _require(retry["campaign"] == backend["campaign"], "Recovery campaign changed")
    replacements = data["composition"].get("replacements")
    _require(
        replacements is None or replacements["source"] == extra["snapshot_provenance"],
        "Recovery must use the same corrected source as the method replacements",
    )
    mapping = entry["recipe_mapping"]
    _require(
        not set(mapping.values()) & {key[1] for key in context["requested"]},
        "Recovery recipe IDs must be distinct from all original requested recipes",
    )
    old_specs = {s["id"]: s for s in original.get("families", [])}
    recovered = {key[1]: value for key, value in retry["requested"].items() if key[0] == "families"}
    _require(
        not original.get("games")
        and len(recovered) == len(retry["requested"])
        and set(mapping) == old_specs.keys()
        and set(mapping.values()) == recovered.keys()
        and len(set(mapping.values())) == len(mapping),
        "Incomplete backend recipe mapping",
    )
    ignored = {"name", "phase_plan", "families", "backend_recovery"}
    for old_id, new_id in mapping.items():
        old, new = old_specs[old_id], recovered[new_id]
        _require(
            old_id != new_id
            and old["device"] == "cuda"
            and new["spec"] == {**old, "id": new_id, "device": "cpu"},
            "Backend recovery changed more than recipe ID and device",
        )
        _require(
            {k: v for k, v in original.items() if k not in ignored}
            == {k: v for k, v in new["suite"].items() if k not in ignored},
            "Backend recovery changed seeds, budgets, or scientific gates",
        )
    _require(
        not {g["id"] for g in data["games"]} & {g["id"] for g in extra["games"]},
        "Backend recovery overlaps original game IDs",
    )
    for name, method in extra["methods"].items():
        published = method
        previous = data["methods"].get(name)
        if previous is not None and previous != method:
            _require(
                previous["software_sha256"] != method["software_sha256"],
                "Conflicting method metadata for the same software identity",
            )
            versions = {"source_sha256", "software_sha256"}
            common = {k: v for k, v in previous.items() if k not in versions}
            _require(
                common == {k: v for k, v in method.items() if k not in versions},
                "Recovered method changed parameters or public status",
            )
            published = {
                **common,
                "source_versions": {
                    previous["software_sha256"]: previous["source_sha256"],
                    method["software_sha256"]: method["source_sha256"],
                },
            }
        data["methods"][name] = published
    for run_id, run in extra["runs"].items():
        _require(
            run_id not in data["runs"] or data["runs"][run_id] == run,
            "Conflicting recovery execution provenance",
        )
        data["runs"][run_id] = run
    for key in ("games", "records", "coverage"):
        data[key].extend(extra[key])
    suite = data["suite"]
    suite["preparation_exclusions"].extend(extra["suite"]["preparation_exclusions"])
    if "preparation_preflight" in extra["suite"]:
        incoming = extra["suite"]["preparation_preflight"]
        current = suite.setdefault("preparation_preflight", {**incoming, "families": []})
        _require(
            {k: v for k, v in current.items() if k != "families"}
            == {k: v for k, v in incoming.items() if k != "families"},
            "Backend recovery changed qualification gates",
        )
        current["families"].extend(incoming["families"])
    suite["budgets_by_game"].update(extra["suite"]["budgets_by_game"])
    suite["budgets"] = sorted({b for grid in suite["budgets_by_game"].values() for b in grid})
    context["fingerprints"].update(retry["fingerprints"])
    data["composition"]["backend_recovery"] = {
        "batch": entry["batch_id"],
        "reason": entry["reason"],
        "recipe_mapping": mapping,
        "source": extra["snapshot_provenance"],
        "inputs": backend["inputs"],
        "composition": extra["composition"],
    }
    _require(
        all(digest(path) == sha for path, sha in backend["checked"].items()),
        "Backend recovery evidence changed",
    )
