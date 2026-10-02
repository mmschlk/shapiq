"""Package a verified public snapshot and baseline results for local reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from shapiq_benchmark.report import merge_results
from shapiq_benchmark.results_io import read_results
from shapiq_benchmark.runner import load_snapshot


def bundle(snapshot_dir: Path, results: Path | list[Path], output: Path) -> None:
    """Export only authenticated regular artifacts and explicitly public baseline methods."""
    result_paths = [results] if isinstance(results, Path) else list(results)
    if not result_paths:
        message = "At least one result file is required."
        raise ValueError(message)
    manifest = snapshot_dir / "snapshot.json"
    if manifest.is_symlink() or not manifest.is_file():
        message = "Snapshot manifest must be a regular file."
        raise ValueError(message)
    declared = json.loads(manifest.read_text())
    artifacts = []
    for relative in declared["artifacts"]:
        path = Path(relative)
        if (
            path.is_absolute()
            or "\\" in relative
            or any(part in ("", ".", "..") for part in relative.split("/"))
            or relative == "snapshot.json"
        ):
            message = f"Unsafe artifact path: {relative}"
            raise ValueError(message)
        artifact = snapshot_dir / path
        if not artifact.is_file() or any(
            (snapshot_dir / parent).is_symlink() for parent in (path, *path.parents)
        ):
            message = f"Artifact must be a regular file without symlinks: {relative}"
            raise ValueError(message)
        artifacts.append((relative, artifact))
    snapshot, _ = load_snapshot(snapshot_dir)
    baselines = [read_results(path) for path in result_paths]
    expected = {
        "schema_version": snapshot["schema_version"],
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_provenance": snapshot["provenance"],
        "suite": snapshot["suite"],
        "games": snapshot["games"],
    }
    for baseline in baselines:
        if any(baseline.get(key) != value for key, value in expected.items()):
            message = "Baseline results must match the snapshot, games, suite, and provenance."
            raise ValueError(message)
        if any(method.get("private") is not False for method in baseline["methods"].values()):
            message = "Public bundles cannot include private candidate methods."
            raise ValueError(message)
    sanitized = merge_results(result_paths)  # Validate all shards together, including conflicts.
    cell_fields = ("game_id", "method", "budget", "seed")
    failure_fields = ("error_type", "failure_reason", "minimum_budget")
    failures = {
        tuple(row[field] for field in cell_fields): {
            field: row[field] for field in failure_fields if field in row
        }
        for row in sanitized["records"]
    }
    for baseline in baselines:
        for row in baseline["records"]:
            row.pop("error", None)
            for field in failure_fields:
                row.pop(field, None)
            row.update(failures[tuple(row[field] for field in cell_fields)])
    inputs = [manifest, *result_paths, *(path for _, path in artifacts)]
    if output.resolve() in {path.resolve() for path in inputs}:
        message = "Bundle output must not replace an input file."
        raise ValueError(message)
    files = {
        "snapshot/snapshot.json": manifest.read_bytes(),
        **{f"snapshot/{name}": path.read_bytes() for name, path in artifacts},
    }
    for index, baseline in enumerate(baselines):
        name = "results.json" if len(baselines) == 1 else f"shard-{index:04d}.json"
        files[f"baselines/{name}"] = (
            json.dumps(baseline, indent=2, allow_nan=False) + "\n"
        ).encode()
    provenance = snapshot["provenance"]
    files["README.md"] = f"""# Reproduce this benchmark locally

Snapshot: `{snapshot["snapshot_id"]}`
Preparation repository commit: `{provenance.get("git_commit", "unrecorded")}`
Preparation shapiq version: `{provenance.get("shapiq", "unrecorded")}`

Full preparation provenance is in `snapshot/snapshot.json`; baseline execution
versions and hardware are in the individual `baselines/*.json` result files. Source hashes record the
actual code; a dirty preparation checkout cannot be reconstructed from its commit
alone. This archive includes frozen game artifacts, truth, and baseline results.

Extract into `local_benchmark/bundle/` in a compatible shapiq checkout with the
benchmark dependencies installed. Check `SHA256SUMS` from the extracted directory
with `sha256sum --check SHA256SUMS`. The runner also verifies snapshot identities
and artifact hashes. From the repository root:

```bash
uv sync --locked --extra benchmark
uv run python -m shapiq_benchmark.runner --snapshot local_benchmark/bundle/snapshot --candidate local_benchmark/candidate.py:factory --output local_benchmark/candidate-results
uv run python -m shapiq_benchmark.report --results local_benchmark/bundle/baselines/*.json local_benchmark/candidate-results/results.json --output local_benchmark/report
uv run python -m http.server 8000 --bind 127.0.0.1 --directory local_benchmark/report
```

The trusted local adapter exports `factory(n, index, order, seed)` and returns an
object with `approximate(budget, game)` returning `InteractionValues`. Access game
values only through the supplied counted callable. Install candidate-specific
dependencies separately. The runner's default campaign limit is ten minutes;
use `--resume` with the same arguments to continue an unfinished campaign.

Candidate results remain local; these commands do not upload or publish them.
Partial coverage is not a full-panel ranking. Accuracy can be compared on these
identical games, but baseline timings from another machine are not local runtime
rankings. Rerun built-in baselines locally by omitting `--candidate` and choosing
a separate output directory.
""".encode()
    files["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in files.items()
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)


def main() -> None:
    """Export a public reproduction archive from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle(args.snapshot, args.results, args.output)


if __name__ == "__main__":
    main()
