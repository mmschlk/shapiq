"""Parallel, resumable operational export of an already terminal benchmark pool.

Collection shards and reference checks survive a later packaging failure. No
estimator is executed and the frozen scientific modules are used unchanged.
"""

# ruff: noqa: T201 -- command-line progress for long-running export jobs

from __future__ import annotations

import argparse
import inspect
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from collect_pool import Inputs, digest, frozen_inventory, require


def write_json(path: Path, value: dict) -> None:
    """Commit a completed cache entry atomically."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def reference_task(task: tuple) -> dict:
    """Check one snapshot, retaining a hash-bound reusable result."""
    from export_pool import check_snapshot

    from shapiq_benchmark.runner import identity, load_snapshot

    path, expected_sha, source_sha, checker_sha, output = task
    require(digest(path) == expected_sha, "Reference snapshot changed")
    key = identity({"snapshot": expected_sha, "source": source_sha, "checker": checker_sha})
    output = Path(output)
    if output.exists():
        saved = json.loads(output.read_text())
        require(
            saved["key"] == key and identity(saved["result"]) == saved["result_sha256"],
            "Reference checkpoint differs",
        )
        return saved["result"]
    snapshot, root = load_snapshot(Path(path))
    checks, fingerprints = check_snapshot(snapshot, root)
    require(digest(path) == expected_sha, "Reference snapshot changed during replay")
    result = {"checks": checks, "fingerprints": fingerprints}
    write_json(output, {"key": key, "result": result, "result_sha256": identity(result)})
    return result


def parallel_references(config: dict, audit: dict, directory: Path, workers: int) -> tuple:
    """Replay independent saved games concurrently, keeping deterministic order."""
    import export_pool

    from shapiq_benchmark.runner import identity

    directory.mkdir(parents=True, exist_ok=True)
    # Alias/publication changes do not invalidate completed numerical checks.
    checker_sha = identity(
        {
            name: inspect.getsource(getattr(export_pool, name))
            for name in ("coefficients", "close", "table_reference", "check_snapshot")
        }
    )
    tasks = []
    for case in audit["cases"]:
        path = (
            Path(config["pool_directory"])
            / "tasks"
            / f"case-{case['case']:06d}"
            / "prepared/snapshot.json"
        )
        if str(path.absolute()) in audit["input_hashes"]:
            tasks.append(
                (
                    str(path),
                    audit["input_hashes"][str(path.absolute())],
                    config["source"]["sha256"],
                    checker_sha,
                    str(directory / f"case-{case['case']:06d}.json"),
                )
            )
    print(f"Checking {len(tasks)} saved references with {workers} workers", flush=True)
    checks, fingerprints = [], {}
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("fork")
    ) as pool:
        for result in pool.map(reference_task, tasks):
            checks.extend(result["checks"])
            require(
                not fingerprints.keys() & result["fingerprints"].keys(), "Repeated reference game"
            )
            fingerprints.update(result["fingerprints"])
    print("Reference checks complete", flush=True)
    return checks, fingerprints


def collect_task(task: tuple) -> dict:
    """Run an independent collector shard using the shared frozen runtime."""
    from pool_checkpoints import create_shard

    config, directory, cases = task
    return create_shard(Path(config), Path(directory), cases)


def parallel_collection(config_path: Path, directory: Path, workers: int) -> Path:
    """Collect contiguous case shards and merge exact rows with SQLite."""
    from pool_checkpoints import merge_shards

    combined = directory / "combined"
    if (combined / "checkpoint.json").exists():
        merge_shards(config_path, [], combined)
        print("Reusing completed collection checkpoint", flush=True)
        return combined
    config = json.loads(config_path.read_text())
    inputs = Inputs()
    inventory = frozen_inventory(config, inputs.pinned(config["suite"]), inputs)
    ids = [task["case"] for task in inventory]
    require(ids == sorted(set(ids)) and ids, "Inventory case order differs")
    size = (len(ids) + workers - 1) // workers
    groups = [ids[i : i + size] for i in range(0, len(ids), size)]
    directory.mkdir(parents=True, exist_ok=True)
    shards = [directory / f"shard-{i:03d}" for i in range(len(groups))]
    print(f"Collecting {len(ids)} cases in {len(shards)} durable parallel shards", flush=True)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("fork")
    ) as pool:
        for i, _result in enumerate(
            pool.map(
                collect_task,
                [
                    (str(config_path), str(path), group)
                    for path, group in zip(shards, groups, strict=True)
                ],
            ),
            start=1,
        ):
            print(f"Collection shard {i}/{len(shards)} complete", flush=True)
    merge_shards(config_path, shards, combined)
    print("Authenticated collection checkpoint complete", flush=True)
    return combined


def main() -> None:
    """Resume completed operational checkpoints; never resume estimator attempts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    require(1 <= args.workers <= 64, "Use 1 to 64 export workers")
    args.directory.mkdir(parents=True, exist_ok=True)
    checkpoint = parallel_collection(args.config, args.directory / "collection", args.workers)
    from export_pool import export

    result = export(
        args.config,
        args.directory / "public-report",
        args.directory / "working.sqlite",
        args.directory / "audit.json",
        checkpoint=checkpoint,
        cache=args.directory / "cache",
        workers=args.workers,
        compress=True,
    )
    print(
        json.dumps(
            {key: result[key] for key in ("status", "report_id", "public_rows", "output_bytes")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
