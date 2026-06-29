"""Storage Interaction Language (SIL) CLI.

A command-line REPL for interacting with leaderboard database backends.
Supports managing multiple active storage connections and composing
sequences of read/write/transfer commands.

Usage:
    python -m leaderboard.storage.cli.cli
    python -m leaderboard.storage.cli.cli --backend local
    python -m leaderboard.storage.cli.cli --backend mongodb
"""

from __future__ import annotations

import argparse
import sys

from .repl import StorageREPL


def main() -> None:
    """Entry point for the Storage Interaction Language CLI."""
    parser = argparse.ArgumentParser(
        prog="sil",
        description="Storage Interaction Language - interactive CLI for leaderboard database backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m leaderboard.storage.cli.cli
  python -m leaderboard.storage.cli.cli --backend local
  python -m leaderboard.storage.cli.cli --backend mongodb
        """,
    )
    parser.add_argument(
        "--backend",
        choices=["local", "mongodb", "huggingface"],
        help="Immediately open a storage connection on startup.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output.",
    )
    args = parser.parse_args()

    repl = StorageREPL(use_color=not args.no_color)

    if args.backend:
        repl.cmd_add(args.backend)

    try:
        repl.run()
    except KeyboardInterrupt:
        repl.print_info("\nInterrupted. Closing all connections.")
        repl.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
