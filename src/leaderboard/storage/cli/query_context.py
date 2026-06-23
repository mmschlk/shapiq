"""Query context for custom command sequences.

A `QueryContext` accumulates a series of filter and field commands issued inside a sequence block (terminated by `eoc`).  When `execute()` is called it evaluates them against the selected storage.

Supported commands inside a sequence block
------------------------------------------
get <field> <value>
    Filter the current result set so that only entries where `entry[field] == value`` remain.  The first `get` call fetches all documents from the storage first.

list <field>
    Print the distinct values of *field* across the current result set (also stores them so the next command can filter on them).

count
    Print the number of entries currently in the result set.

sort <field> [asc|desc]
    Sort the result set by *field* (default: asc).

show [n]
    Pretty-print the first *n* entries (default: 10).

eoc
    End of commands - execute and print summary.
"""

from __future__ import annotations

from typing import Any, Protocol


class _StorageClient(Protocol):
    """Minimal protocol for a storage client usable by QueryContext."""

    def get_all(self) -> list[dict[str, Any]]:
        """Return all documents in the storage."""
        ...


class QueryContext:
    """Holds state for a multi-step query sequence."""

    def __init__(self, storage_id: str) -> None:
        """Initialize a new QueryContext for the given *storage_id*."""
        self.storage_id = storage_id
        self._rows: list[dict[str, Any]] | None = None  # None = not yet fetched
        self._commands: list[tuple[str, list[str]]] = []

    # ------------------------------------------------------------------
    # Command accumulation
    # ------------------------------------------------------------------

    def add_command(self, verb: str, args: list[str]) -> None:
        """Add a command to the sequence."""
        self._commands.append((verb, args))

    def has_commands(self) -> bool:
        """Return True if any commands have been accumulated."""
        return bool(self._commands)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, client: _StorageClient) -> list[str]:
        """Run accumulated commands against *client*.

        Returns a list of output lines to display.
        """
        output: list[str] = []
        rows: list[dict[str, Any]] = list(client.get_all())

        for verb_user, args in self._commands:
            verb = verb_user.lower()

            if verb == "get":
                if len(args) < 2:
                    output.append("  [error] get requires <field> <value>")
                    continue
                field = args[0]
                raw_value = _coerce(args[1])
                before = len(rows)
                rows = [r for r in rows if str(r.get(field, "")) == str(raw_value)]
                output.append(
                    f"  get {field!r} == {raw_value!r}  →  {len(rows)} / {before} entries kept"
                )

            elif verb == "list":
                if not args:
                    output.append("  [error] list requires <field>")
                    continue
                field = args[0]
                values = sorted({str(r.get(field, "")) for r in rows})
                output.append(f"  list {field!r}  ({len(values)} distinct value(s)):")
                output.extend(f"    • {v}" for v in values)

            elif verb == "count":
                output.append(f"  count  →  {len(rows)} entries")

            elif verb == "sort":
                if not args:
                    output.append("  [error] sort requires <field>")
                    continue
                field = args[0]
                reverse = len(args) > 1 and args[1].lower() == "desc"
                rows = sorted(rows, key=lambda r: str(r.get(field, "")), reverse=reverse)
                direction = "desc" if reverse else "asc"
                output.append(f"  sort {field!r} {direction}")

            elif verb == "show":
                n = int(args[0]) if args and args[0].isdigit() else 10
                output.append(f"  show (first {min(n, len(rows))} of {len(rows)} entries):")
                for i, row in enumerate(rows[:n]):
                    output.append(f"    [{i}] {_fmt_row(row)}")

            else:
                output.append(f"  [error] unknown sequence command: {verb!r}")

        output.append(f"\n  Sequence complete - {len(rows)} entries in final result set.")
        return output


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _coerce(value: str) -> int | float | str:
    """Try to cast a CLI string argument to int or float; fall back to str."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    # Strip surrounding quotes if the user typed them explicitly
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _fmt_row(row: dict[str, Any]) -> str:
    """Compact single-line representation of a document."""
    items = []
    for k, v in row.items():
        if k.startswith("_"):
            continue
        sv = str(v)
        if len(sv) > 40:
            sv = sv[:37] + "..."
        items.append(f"{k}={sv!r}")
    return "  ".join(items) if items else "{}"
