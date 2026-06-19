"""Command line interface for managing leaderboard storage entries in MongoDB, including uploading runs, listing configurations, and aggregating metrics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from leaderboard.storage.connection import DatabaseClientFactory

if TYPE_CHECKING:
    from leaderboard.storage.connection import DatabaseClient
    from leaderboard.storage.data_classes import RunConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _get_config(db: DatabaseClient, idx: int) -> RunConfig:
    """Helper function to retrieve a RunConfig by index from the database."""
    configs = db.get_unique_configs()
    if not configs:
        logging.info("No configurations found in the database.")
        sys.exit(0)
    if idx < 0 or idx >= len(configs):
        logging.error("--config-index must be between 0 and %s.", len(configs) - 1)
        sys.exit(1)
    return configs[idx]


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def cmd_upload(db: DatabaseClient, args: argparse.Namespace) -> None:
    """Upload runs from a JSONL file, where each line is a JSON object representing a run document to be inserted into MongoDB."""
    if not Path(args.file).exists():
        logging.error("File not found: %s", args.file)
        sys.exit(1)

    entries, skipped = [], 0
    with Path.open(args.file, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw_stripped = raw.strip()
            if not raw_stripped:
                continue
            try:
                entries.append(json.loads(raw_stripped))
            except json.JSONDecodeError as exc:
                logging.warning("Skipping line %s — %s", lineno, exc)
                skipped += 1

    db.insert_many(entries)
    logging.info("Inserted %s runs (%s lines skipped).", len(entries), skipped)


def cmd_configs(db: DatabaseClient, _args: argparse.Namespace) -> None:
    """List all unique RunConfigs in the database with their index for reference in other commands."""
    configs = db.get_unique_configs()
    logging.info("Found %s unique configuration(s):", len(configs))
    for _i, cfg in enumerate(configs):
        logging.info("  [%s] %s", _i, cfg)


def cmd_investigate(db: DatabaseClient, _args: argparse.Namespace) -> None:
    """Investigate duplicate configurations."""
    configs = db.get_unique_configs()
    logging.info("Found %s unique configuration(s):", len(configs))
    for _i, cfg in enumerate(configs):
        runs = db.get_by_config(cfg)

        # Check for duplicates by looking at runs with the same config and seed
        seed_counts = {}
        for r in runs:
            seed = r.get("seed")
            if not seed:
                seed = r.get("approx_seed")
            seed_counts[seed] = seed_counts.get(seed, 0) + 1

        duplicates = {s: c for s, c in seed_counts.items() if c > 1}
        if duplicates:
            logging.info(
                "Config [%s] has %s duplicate seed(s): %s", _i, len(duplicates), duplicates
            )
            logging.info("Config details: %s", cfg)


def cmd_delete_duplicates(db: DatabaseClient, _args: argparse.Namespace) -> None:
    """Delete duplicate runs for each configuration, keeping only one run per seed per unique config."""
    configs = db.get_unique_configs()
    logging.info("Found %s unique configuration(s):", len(configs))
    total_deleted = 0
    for _i, cfg in enumerate(configs):
        runs = db.get_by_config(cfg)

        # Group runs by seed
        seed_to_runs = {}
        for r in runs:
            seed = r.get("seed")
            if not seed:
                seed = r.get("approx_seed")
            seed_to_runs.setdefault(seed, []).append(r)

        # Identify duplicates and delete all but one run per seed
        for runs_with_seed in seed_to_runs.values():
            if len(runs_with_seed) > 1:
                # Keep the first run and delete the rest
                to_delete = runs_with_seed[1:]
                for r in to_delete:
                    deleted_count = db.delete_by_id(r["run_id"])
                    total_deleted += deleted_count

    logging.info("Deleted %s duplicate run(s).", total_deleted)


def cmd_count(db: DatabaseClient, args: argparse.Namespace) -> None:
    """Count runs for a specific configuration, identified by its index from the configs command."""
    config = _get_config(db, args.config_index)
    logging.info("Config [%s] has %s run(s).", args.config_index, db.count_by_config(config))


def cmd_games(db: DatabaseClient, _args: argparse.Namespace) -> None:
    """List distinct game names."""
    games = db.get_games()
    logging.info("Distinct games (%s):", len(games))
    for g in games:
        logging.info("  - %s", g)


def cmd_approximators(db: DatabaseClient, _args: argparse.Namespace) -> None:
    """List distinct approximator names."""
    approximators = db.get_approximators()
    logging.info("Distinct approximators (%s):", len(approximators))
    for a in approximators:
        logging.info("  - %s", a)
    logging.info()


def cmd_delete_all(db: DatabaseClient, args: argparse.Namespace) -> None:
    """Delete ALL documents in the collection. Use with caution! Requires --confirm flag to execute."""
    if not args.confirm:
        logging.info("Pass --confirm to actually delete all documents.")
        return
    logging.info("Deleted %s document(s).", db.delete_all())


def cmd_delete_config(db: DatabaseClient, args: argparse.Namespace) -> None:
    """Delete all runs for a specific configuration, identified by its index from the configs command."""
    config = _get_config(db, args.config_index)
    logging.info(
        "Deleted %s run(s) for config [%s].", db.delete_by_config(config), args.config_index
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="entry.py", description="CLI for the shapiq MongoDB storage module."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("upload", help="Upload runs from a JSONL file.")
    p.add_argument("--file", required=True)

    sub.add_parser("configs", help="List all unique RunConfigs.")

    p = sub.add_parser("count", help="Count runs for a config.")
    p.add_argument("--config-index", type=int, default=0)

    sub.add_parser("investigate", help="Investigate duplicate configurations.")
    sub.add_parser("delete-duplicates", help="Delete duplicate runs for each configuration.")

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
    "count": cmd_count,
    "games": cmd_games,
    "approximators": cmd_approximators,
    "investigate": cmd_investigate,
    "delete-duplicates": cmd_delete_duplicates,
    "delete-all": cmd_delete_all,
    "delete-config": cmd_delete_config,
}


def main() -> None:
    """Entry point for the CLI. Parses arguments, connects to MongoDB, and dispatches to the appropriate command function."""
    args = _build_parser().parse_args()
    mongoDBClient = DatabaseClientFactory.create_client("mongodb", db_args={})
    with mongoDBClient as db:
        _COMMANDS[args.command](db, args)


if __name__ == "__main__":
    main()
