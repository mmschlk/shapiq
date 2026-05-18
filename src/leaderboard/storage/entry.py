from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from dotenv import load_dotenv

from leaderboard.storage.connection import MongoDBClient
from leaderboard.storage.data_classes import RunConfig
from leaderboard.storage.metrics import MetricsLoader



def _get_config(db: MongoDBClient, idx: int) -> RunConfig:
    configs = db.get_unique_configs()
    if not configs:
        print("No configurations found in the database.")
        sys.exit(0)
    if idx < 0 or idx >= len(configs):
        print(f"ERROR: --config-index must be between 0 and {len(configs) - 1}.", file=sys.stderr)
        sys.exit(1)
    return configs[idx]


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_upload(db: MongoDBClient, args: argparse.Namespace) -> None:
    if not os.path.exists(args.file):
        print(f"ERROR: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    entries, skipped = [], 0
    with open(args.file, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entries.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(f"  WARNING: skipping line {lineno} — {exc}")
                skipped += 1

    db.insert_many(entries)
    print(f"Inserted {len(entries)} runs ({skipped} lines skipped).")


def cmd_configs(db: MongoDBClient, _args: argparse.Namespace) -> None:
    configs = db.get_unique_configs()
    print(f"\nFound {len(configs)} unique configuration(s):\n")
    for i, cfg in enumerate(configs):
        print(f"  [{i}] {cfg}")
    print()


def cmd_metrics(db: MongoDBClient, args: argparse.Namespace) -> None:
    config = _get_config(db, args.config_index)
    print(f"\nAggregated metrics for config [{args.config_index}]:\n  {config}\n")
    aggregated = MetricsLoader(db).aggregate(config)
    print(json.dumps(aggregated, indent=2) if aggregated else "  (no metric data found)")


def cmd_count(db: MongoDBClient, args: argparse.Namespace) -> None:
    config = _get_config(db, args.config_index)
    print(f"Config [{args.config_index}] has {db.count_by_config(config)} run(s).")


def cmd_games(db: MongoDBClient, _args: argparse.Namespace) -> None:
    games = db.get_games()
    print(f"\nDistinct games ({len(games)}):")
    for g in games:
        print(f"  - {g}")
    print()


def cmd_approximators(db: MongoDBClient, _args: argparse.Namespace) -> None:
    approximators = db.get_approximators()
    print(f"\nDistinct approximators ({len(approximators)}):")
    for a in approximators:
        print(f"  - {a}")
    print()


def cmd_delete_all(db: MongoDBClient, args: argparse.Namespace) -> None:
    if not args.confirm:
        print("Pass --confirm to actually delete all documents.")
        return
    print(f"Deleted {db.delete_all()} document(s).")


def cmd_delete_config(db: MongoDBClient, args: argparse.Namespace) -> None:
    config = _get_config(db, args.config_index)
    print(f"Deleted {db.delete_by_config(config)} run(s) for config [{args.config_index}].")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="entry.py", description="CLI for the shapiq MongoDB storage module.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("upload", help="Upload runs from a JSONL file.")
    p.add_argument("--file", required=True)

    sub.add_parser("configs", help="List all unique RunConfigs.")

    p = sub.add_parser("metrics", help="Show aggregated metrics for a config.")
    p.add_argument("--config-index", type=int, default=0)

    p = sub.add_parser("count", help="Count runs for a config.")
    p.add_argument("--config-index", type=int, default=0)

    sub.add_parser("games", help="List distinct game names.")
    sub.add_parser("approximators", help="List distinct approximator names.")

    p = sub.add_parser("delete-all", help="Delete ALL documents.")
    p.add_argument("--confirm", action="store_true")

    p = sub.add_parser("delete-config", help="Delete runs for a specific config.")
    p.add_argument("--config-index", type=int, default=0)

    return parser


_COMMANDS = {
    "upload": cmd_upload,
    "configs": cmd_configs,
    "metrics": cmd_metrics,
    "count": cmd_count,
    "games": cmd_games,
    "approximators": cmd_approximators,
    "delete-all": cmd_delete_all,
    "delete-config": cmd_delete_config,
}


def main() -> None:
    args = _build_parser().parse_args()
    mongoDBClient = MongoDBClient.from_env()
    with mongoDBClient as db:
        _COMMANDS[args.command](db, args)


if __name__ == "__main__":
    main()