"""Export reviewed matrix waves and published history into partitioned public data."""

from __future__ import annotations

import argparse
from pathlib import Path

from shapiq_benchmark.matrix_publication import compose_matrix
from shapiq_benchmark.partitioned import write_partitioned_report
from shapiq_benchmark.record_store import RecordStore


def main() -> None:
    """Authenticate the publication plan and stream its records through temporary storage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Publication plan pinning completed wave reviews")
    parser.add_argument("output", type=Path, help="New directory for public data assets")
    parser.add_argument("--plan-sha256", required=True, help="Independently verified plan checksum")
    parser.add_argument(
        "--database", type=Path, required=True, help="New temporary SQLite file on lab storage"
    )
    parser.add_argument("--cache-dir", type=Path, help="Private authenticated batch cache")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or any(
        path is not None and (path.resolve() == output or output in path.resolve().parents)
        for path in (args.database, args.cache_dir)
    ):
        parser.error("Output must be a new directory separate from the database and cache.")
    with RecordStore(args.database) as records:
        data = compose_matrix(
            args.plan,
            plan_sha256=args.plan_sha256,
            record_store=records,
            cache_dir=args.cache_dir,
        )
        write_partitioned_report(data, args.output)


if __name__ == "__main__":
    main()
