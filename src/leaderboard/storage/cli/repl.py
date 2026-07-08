"""Storage Interaction Language REPL.

Top-level command grammar
--------------------------
  Active storage management
    list storages              List open connections and their IDs.
    add <backend>              Open a new connection (prompts for params).
    close <storage_id>         Close an open connection.
    close all                  Close every open connection.

  Storage transfer / mutation
    insert <src> to <dst>                Transfer all documents from src to dst.
    insert safe <src> to <dst>           Safe transfer (no duplicates).
    insert safe <src> to <dst> using <mode>
        mode: merge | replace | skip

    delete from <storage_id>             Prompt for what to delete.
    delete entries <src> from <dst>      Delete entries found in <src> from <dst>.

  Custom sequence
    sequence [<storage_id>]    Start a multi-command sequence (default: first active storage).
      Inside a sequence:
        get <field> <value>    Filter entries by field == value.
        list <field>           Print distinct values of field.
        count                  Print entry count.
        sort <field> [asc|desc]
        show [n]               Print first n entries.
        eoc                    End of commands - run and display results.
        help                   Show sequence help.
        abort                  Abort without executing.

  General
    help                       Show this message.
    exit / quit                Close all connections and exit.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any

from leaderboard.storage.connection import (
    DatabaseClient,
    DatabaseClientFactory,
    DBClientError,
    UnsupportedDatabaseBackendError,
)

from . import formatting as fmt
from .query_context import QueryContext
from .registry import StorageRegistry

# ---------------------------------------------------------------------------
# Parameter prompts per backend
# ---------------------------------------------------------------------------

_BACKEND_PARAMS: dict[str, list[tuple[str, str, str]]] = {
    "local": [
        ("LOCAL_DB_PATH", "Database file path", "~/.leaderboard/db.jsonl"),
    ],
    "mongodb": [
        ("MONGODB_URI", "MongoDB URI", "mongodb://localhost:27017"),
        ("MONGODB_DB", "Database name", "shapiq-leaderboard"),
        ("MONGODB_COLLECTION", "Collection name", "runs"),
    ],
    "huggingface": [
        ("HF_REPO_ID", "HuggingFace repo ID", "username/leaderboard"),
        ("HF_TOKEN", "HuggingFace token", "(from .env)"),
        ("HF_FILENAME", "JSONL filename in repo", "runs.jsonl"),
    ],
}

_BACKEND_ALIASES = {"local", "mongodb", "huggingface", "hf"}
_ALIAS_MAP = {"hf": "huggingface"}


class StorageREPL:
    """Interactive Read-Eval-Print Loop for the Storage Interaction Language."""

    def __init__(self, *, use_color: bool = True) -> None:
        """Initialize a new REPL instance."""
        fmt.set_color(enabled=use_color)
        self._registry = StorageRegistry()
        self._in_sequence = False
        self._seq_ctx: QueryContext | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the REPL loop until exit."""
        self._banner()
        while True:
            try:
                line = self._readline()
            except EOFError:
                self.print_info("\nEOF received. Exiting.")
                self.shutdown()
                return

            line = line.strip()
            if not line:
                continue

            if self._in_sequence:
                self._handle_sequence_input(line)
            else:
                self._dispatch(line)

    def shutdown(self) -> None:
        """Close all connections and clean up before exiting."""
        self._registry.close_all()

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def print_ok(self, msg: str) -> None:
        """Print a success message."""
        print(fmt.ok(msg))  # noqa: T201

    def print_warn(self, msg: str) -> None:
        """Print a warning message."""
        print(fmt.warn(msg))  # noqa: T201

    def print_error(self, msg: str) -> None:
        """Print an error message."""
        print(fmt.error(msg))  # noqa: T201

    def print_info(self, msg: str) -> None:
        """Print an informational message."""
        print(fmt.info(msg))  # noqa: T201

    def _banner(self) -> None:
        """Print the REPL banner and usage hint."""
        print()  # noqa: T201
        print(fmt.header("  Storage Interaction Language (SIL)"))  # noqa: T201
        print(fmt.dim("  Type 'help' for available commands, 'exit' to quit."))  # noqa: T201
        print()  # noqa: T201

    def _readline(self) -> str:
        """Read a line of input from the user, with prompt."""
        active = self._registry.ids()
        if self._in_sequence:
            sid = self._seq_ctx.storage_id if self._seq_ctx else "?"
            prompt = f"  {fmt.bold(fmt.cyan('seq'))}({fmt.magenta(sid)})> "
        else:
            prompt = fmt.prompt(active)
        return input(prompt)

    # ------------------------------------------------------------------
    # Top-level dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, line: str) -> None:
        """Parse and dispatch a top-level command line.

        Args:
            line (str): The command line to parse and dispatch.
        """
        tokens = _tokenise(line)
        if not tokens:
            return
        verb = tokens[0].lower()

        if verb in ("help", "?"):
            self._help_top()
        elif verb in ("exit", "quit", "q"):
            self._cmd_exit()
        elif verb == "list":
            self._cmd_list(tokens[1:])
        elif verb == "add":
            self.cmd_add(tokens[1] if len(tokens) > 1 else None)
        elif verb == "close":
            self._cmd_close(tokens[1:])
        elif verb == "insert":
            self._cmd_insert(tokens[1:])
        elif verb == "delete":
            self._cmd_delete(tokens[1:])
        elif verb in ("sequence", "seq"):
            self._cmd_sequence(tokens[1:])
        else:
            self.print_error(f"Unknown command: {verb!r}. Type 'help' for options.")

    # ------------------------------------------------------------------
    # Top-level commands
    # ------------------------------------------------------------------

    def _help_top(self) -> None:
        """Print the top-level help message."""
        print()  # noqa: T201
        print(fmt.header("  Active Storage Management"))  # noqa: T201
        _hline("list storages", "List all open connections with their IDs.")
        _hline(
            "add <backend>",
            "Open a new connection. backends: local | mongodb | huggingface",
        )
        _hline("close <id>", "Close a connection (e.g. close local1).")
        _hline("close all", "Close every open connection.")
        print()  # noqa: T201
        print(fmt.header("  Storage Transfer & Mutation"))  # noqa: T201
        _hline("insert <src> to <dst>", "Copy all documents from src to dst.")
        _hline(
            "insert safe <src> to <dst> [using <mode>]",
            "Copy without duplicates. mode: merge | replace | skip  (default: merge)",
        )
        _hline("delete from <id>", "Interactively delete entries from a storage.")
        _hline(
            "delete entries <src> from <dst> ",
            "Delete every entry in dst that also appears in src. Matching by document ID (default) or exact content.",
        )
        print()  # noqa: T201
        print(fmt.header("  Custom Sequence"))  # noqa: T201
        _hline(
            "sequence [<id>]",
            "Start a multi-command sequence against a storage (default: first active).",
        )
        print()  # noqa: T201
        print(fmt.header("  General"))  # noqa: T201
        _hline("help", "Show this message.")
        _hline("exit / quit", "Close all connections and exit.")
        print()  # noqa: T201

    def _cmd_list(self, list_args: list[str]) -> None:
        """Handle the 'list' command, which lists the active storages. If no subcommand is given, defaults to 'list storages'.

        Args:
            list_args: List of command arguments (excluding the 'list' verb).
        """
        sub = list_args[0].lower() if list_args else "storages"
        if sub in ("storage", "storages", "connections"):
            self._list_storages()
        else:
            self.print_error(f"Unknown list target: {sub!r}.  Did you mean 'list storages'?")

    def _list_storages(self) -> None:
        """List all active storage connections."""
        ids = self._registry.ids()
        if not ids:
            self.print_warn("No active storage connections. Use 'add <backend>' to open one.")
            return
        print()  # noqa: T201
        print(fmt.header(f"  Active connections ({len(ids)})"))  # noqa: T201
        for sid in ids:
            meta = self._registry.meta(sid)
            backend = meta.get("backend", "?")
            args = meta.get("args", {})
            args_str = "  ".join(f"{k}={v!r}" for k, v in args.items() if v)
            print(f"    {fmt.storage_id(sid)}  {fmt.dim(backend)}  {fmt.dim(args_str)}")  # noqa: T201
        print()  # noqa: T201

    def cmd_add(self, backend: str | None) -> None:
        """Prompt for backend if not given, then prompt for params and connect.

        Args:
            backend: The backend to connect to.
        """
        if backend is None:
            print()  # noqa: T201
            print(fmt.header("  Available backends:"))  # noqa: T201
            for b in ("local", "mongodb", "huggingface"):
                print(f"    {fmt.cyan(b)}")  # noqa: T201
            print()  # noqa: T201
            backend = input(f"  {fmt.bold('Backend')}> ").strip().lower()

        backend = _ALIAS_MAP.get(backend, backend)
        if backend not in _BACKEND_PARAMS:
            self.print_error(
                f"Unknown backend {backend!r}. Choose from: local, mongodb, huggingface."
            )
            return

        param_defs = _BACKEND_PARAMS[backend]
        print()  # noqa: T201
        print(fmt.dim("  Leave blank to use .env defaults for each parameter."))  # noqa: T201
        db_args: dict[str, str] = {}
        for env_key, label, hint in param_defs:
            val = input(f"  {fmt.bold(label)} {fmt.dim(f'[{hint}]')}> ").strip()
            if val:
                db_args[env_key] = val

        print(fmt.dim(f"  Connecting to {backend}..."))  # noqa: T201
        print(fmt.dim(f"\t\tParameters: {db_args}"))  # noqa: T201

        if (
            backend == "local"
            and "LOCAL_DB_PATH" in db_args
            and not Path(db_args["LOCAL_DB_PATH"]).exists()
        ):
            create = (
                input(
                    f"  Database file {db_args['LOCAL_DB_PATH']} does not exist. Create it? (y/n)> "
                )
                .strip()
                .lower()
            )
            if create == "y":
                db_args["CREATE"] = True
            else:
                self.print_warn("Aborted. Database file does not exist.")
                return

        try:
            client = DatabaseClientFactory.create_client(backend, db_args)
            healthy = client.test_connection()
        except DBClientError as exc:
            self.print_error(f"Failed to connect: {exc}")
            return

        if not healthy:
            self.print_error("Connection test failed. Check your parameters and try again.")
            client.close()
            return

        sid = self._registry.add(backend, client, db_args)
        self.print_ok(f"Connected - storage ID: {fmt.storage_id(sid)}")

    def _cmd_close(self, close_args: list[str]) -> None:
        """Handle the 'close' command, which closes a storage connection.

        Args:
            close_args: List of command arguments (excluding the 'close' verb).
        """
        if not close_args:
            self.print_error("Usage: close <storage_id> | close all")
            return
        target = close_args[0].lower()
        if target == "all":
            n = len(self._registry)
            self._registry.close_all()
            self.print_ok(f"Closed {n} connection(s).")
            return
        sid = close_args[0]
        if sid not in self._registry:
            self.print_error(f"No active storage {fmt.storage_id(sid)}. Use 'list storages'.")
            return
        self._registry.close(sid)
        self.print_ok(f"Closed {fmt.storage_id(sid)}.")

    # ------------------------------------------------------------------
    # Insert command
    # ------------------------------------------------------------------

    def _cmd_insert(self, insert_args: list[str]) -> None:
        """Handle the 'insert' command, which transfers documents from one storage to another.

        Usage:
            insert [safe] <src> to <dst> [using <mode>]

        Args:
            insert_args: List of command arguments (excluding the 'insert' verb).
        """
        if not insert_args:
            self.print_error("Usage: insert [safe] <src> to <dst> [using <mode>]")
            return

        safe = False
        rest = insert_args[:]
        if rest and rest[0].lower() == "safe":
            safe = True
            rest = rest[1:]

        # expect: <src> to <dst> [using <mode>]
        try:
            to_idx = [t.lower() for t in rest].index("to")
        except ValueError:
            self.print_error("Usage: insert [safe] <src> to <dst> [using <mode>]")
            return

        src_id = rest[to_idx - 1] if to_idx >= 1 else None
        after_to = rest[to_idx + 1 :]

        # <dst> [using <mode>]
        mode = "merge"
        if len(after_to) >= 3 and after_to[1].lower() == "using":
            dst_id = after_to[0]
            mode = after_to[2].lower()
        else:
            dst_id = after_to[0] if after_to else None

        if not src_id or not dst_id:
            self.print_error("Usage: insert [safe] <src> to <dst> [using <mode>]")
            return

        try:
            src = self._resolve_storage(src_id)
            dst = self._resolve_storage(dst_id)
        except DBClientError as exc:
            self.print_error(f"Failed to resolve storage: {exc}")
            return

        if mode not in ("merge", "replace", "skip"):
            self.print_error(f"Unknown mode {mode!r}. Choose from: merge, replace, skip.")
            return

        print(fmt.dim(f"  Fetching documents from {src_id}..."))  # noqa: T201
        try:
            docs = src.get_all()
        except DBClientError as exc:
            self.print_error(f"Failed to read from {fmt.storage_id(src_id)}: {exc}")
            return

        if not docs:
            self.print_warn(f"No documents found in {fmt.storage_id(src_id)}.")
            return

        print(fmt.dim(f"  Inserting {len(docs)} document(s) into {dst_id}..."))  # noqa: T201
        try:
            if safe:
                inserted = dst.safe_insert_many(docs, mode=mode)
                self.print_ok(
                    f"Safe-inserted {inserted} / {len(docs)} document(s) "
                    f"({fmt.storage_id(src_id)} → {fmt.storage_id(dst_id)}, mode={mode})."
                )
            else:
                dst.insert_many(docs)
                self.print_ok(
                    f"Inserted {len(docs)} document(s) "
                    f"({fmt.storage_id(src_id)} → {fmt.storage_id(dst_id)})."
                )
        except DBClientError as exc:
            self.print_error(f"Insert failed: {exc}")

    # ------------------------------------------------------------------
    # Delete command
    # ------------------------------------------------------------------

    def _cmd_delete(self, args: list[str]) -> None:
        """Handle the 'delete' command, which can either delete from a storage or delete entries from one storage that exist in another.

        Usage:
            delete from <id>
            delete entries <src> from <dst>
        Args:
            args: List of command arguments (excluding the 'delete' verb).
        """
        if not args:
            self.print_error("Usage: delete from <id>  |  delete entries <src> from <dst>")
            return

        sub = args[0].lower()

        if sub == "from":
            # delete from <id>
            if len(args) < 2:
                self.print_error("Usage: delete from <id>")
                return
            self._delete_interactive(args[1])

        elif sub == "entries":
            # delete entries <src> from <dst>
            try:
                from_idx = [t.lower() for t in args].index("from")
            except ValueError:
                self.print_error("Usage: delete entries <src> from <dst>")
                return
            src_id = args[from_idx - 1] if from_idx >= 2 else None
            dst_id = args[from_idx + 1] if from_idx + 1 < len(args) else None

            if not src_id or not dst_id:
                self.print_error("Usage: delete entries <src> from <dst> ")
                return
            self._delete_entries(src_id, dst_id)

        else:
            self.print_error("Usage: delete from <id>  |  delete entries <src> from <dst> ")

    def _delete_interactive(self, sid: str) -> None:
        """Prompt the user for what to delete from a storage.

        Args:
            sid: The storage ID to delete from.
        """
        try:
            client = self._resolve_storage(sid)
        except DBClientError as exc:
            self.print_error(f"Failed to resolve storage: {exc}")
            return

        print()  # noqa: T201
        print(fmt.header(f"  Delete from {fmt.storage_id(sid)}"))  # noqa: S608, T201 -- CLI header, not query
        print(fmt.dim("  Options:"))  # noqa: T201
        _hline("all", "Delete every document.")
        _hline("by config", "Delete documents matching a specific config.")
        print()  # noqa: T201
        choice = input(f"  {fmt.bold('Delete what')}> ").strip().lower()

        if choice == "all":
            confirm = input(
                f"  {fmt.yellow('Are you sure? This deletes everything in')} "
                f"{fmt.storage_id(sid)}.  Type YES to confirm: "
            ).strip()
            if confirm != "YES":
                self.print_warn("Aborted.")
                return
            try:
                n = client.delete_all()
                self.print_ok(f"Deleted {n} document(s) from {fmt.storage_id(sid)}.")
            except DBClientError as exc:
                self.print_error(f"Delete failed: {exc}")

        elif choice in ("by config", "config"):
            print(fmt.dim("  Enter config fields (blank line to finish):"))  # noqa: T201
            config_fields: dict[str, Any] = {}
            while True:
                kv = input("    field=value> ").strip()
                if not kv:
                    break
                if "=" not in kv:
                    self.print_warn("  Format: field=value")
                    continue
                k, v = kv.split("=", 1)
                config_fields[k.strip()] = v.strip()
            if not config_fields:
                self.print_warn("No fields specified. Aborted.")
                return
            # Build a minimal RunConfig-like object or pass as dict
            try:
                n = client.delete_by_filter(config_fields)  # type: ignore[arg-type]
                self.print_ok(f"Deleted {n} document(s) from {fmt.storage_id(sid)}.")
            except DBClientError as exc:
                self.print_error(f"Delete failed: {exc}")
        else:
            self.print_warn(f"Unknown option {choice!r}. Aborted.")

    def _delete_entries(self, src_id: str, dst_id: str) -> None:
        """Delete entries from dst that also exist in src, matching by document ID.

        Args:
            src_id: The source storage ID to fetch entries from.
            dst_id: The destination storage ID to delete entries from.
        """
        try:
            src = self._resolve_storage(src_id)
            dst = self._resolve_storage(dst_id)
        except DBClientError as exc:
            self.print_error(f"Failed to resolve storage: {exc}")
            return

        print(fmt.dim(f"  Fetching entry IDs from {src_id}..."))  # noqa: T201
        try:
            src_docs = src.get_all()
        except DBClientError as exc:
            self.print_error(f"Failed to read from {fmt.storage_id(src_id)}: {exc}")
            return

        deleted = 0
        failed = 0
        print(fmt.dim(f"  Deleting {len(src_docs)} matching entries from {dst_id}..."))  # noqa: T201
        for doc in src_docs:
            doc_id = doc.get("run_id")
            if doc_id:
                try:
                    deleted += dst.delete_by_id(str(doc_id))
                except DBClientError:
                    failed += 1
            else:
                failed += 1

        self.print_ok(f"Deleted {deleted} / {len(src_docs)} entries from {fmt.storage_id(dst_id)}.")
        if failed:
            self.print_warn(f"{failed} entries could not be matched (no ID field).")

    # ------------------------------------------------------------------
    # Sequence command
    # ------------------------------------------------------------------

    def _cmd_sequence(self, seq_args: list[str]) -> None:
        """Handle the 'sequence' command, which starts a multi-command sequence.

        Args:
            seq_args: List of command arguments (excluding the 'sequence' verb).
        """
        if self._in_sequence:
            self.print_warn("Already inside a sequence. Use 'eoc' to end or 'abort' to cancel.")
            return

        # Resolve target storage
        if seq_args:
            sid = seq_args[0]
        else:
            ids = self._registry.ids()
            if not ids:
                self.print_error("No active storage. Use 'add <backend>' first.")
                return
            sid = ids[0]
            if len(ids) > 1:
                self.print_info(
                    f"Multiple storages active; defaulting to {fmt.storage_id(sid)}. "
                    "Specify an ID to override: 'sequence <id>'"
                )

        if sid not in self._registry:
            self.print_error(f"Unknown storage {fmt.storage_id(sid)}.")
            return

        self._in_sequence = True
        self._seq_ctx = QueryContext(sid)
        print()  # noqa: T201
        print(  # noqa: T201
            fmt.dim(
                f"  Sequence started on {fmt.storage_id(sid)}. "
                "Type 'help' for commands, 'eoc' to run, 'abort' to cancel."
            )
        )

    def _handle_sequence_input(self, line: str) -> None:
        """Handle input while inside a sequence.

        Args:
            line: The input line to process.
        """
        tokens = _tokenise(line)
        if not tokens:
            return
        verb = tokens[0].lower()

        if verb == "help":
            self._help_sequence()
        elif verb == "abort":
            self._in_sequence = False
            self._seq_ctx = None
            self.print_warn("Sequence aborted.")
        elif verb == "eoc":
            self._execute_sequence()
        elif verb in ("get", "list", "count", "sort", "show"):
            if self._seq_ctx is None:
                self.print_error("No active sequence context. This should not happen.")
                return
            self._seq_ctx.add_command(verb, tokens[1:])
            print(fmt.dim(f"  + {line}"))  # noqa: T201
        else:
            self.print_error(f"Unknown sequence command: {verb!r}. Type 'help' or 'abort'.")

    def _execute_sequence(self) -> None:
        """Execute the accumulated sequence of commands and display results."""
        if self._seq_ctx is None:
            self.print_error("No active sequence to execute.")
            return

        ctx = self._seq_ctx
        self._in_sequence = False
        self._seq_ctx = None

        if not ctx.has_commands():
            self.print_warn("Sequence was empty - nothing to execute.")
            return

        client = self._registry.get(ctx.storage_id)
        if client is None:
            self.print_error(f"Storage {fmt.storage_id(ctx.storage_id)} is no longer active.")
            return

        print()  # noqa: T201
        print(fmt.header(f"  Executing sequence on {fmt.storage_id(ctx.storage_id)}"))  # noqa: T201
        try:
            output_lines = ctx.execute(client)
        except (DBClientError, ValueError, KeyError) as exc:
            self.print_error(f"Sequence execution failed: {exc}")
            return

        for line in output_lines:
            print(line)  # noqa: T201
        print()  # noqa: T201

    def _help_sequence(self) -> None:
        """Print help for sequence commands."""
        print()  # noqa: T201
        print(fmt.header("  Sequence commands"))  # noqa: T201
        _hline(
            "get <field> <value>",
            'Filter entries where field == value. e.g. get game_name "SOUM"',
        )
        _hline("list <field>", "Print distinct values of field across current result set.")
        _hline("count", "Print number of entries currently in the result set.")
        _hline("sort <field> [asc|desc]", "Sort result set by field.")
        _hline("show [n]", "Print first n entries (default: 10).")
        _hline("eoc", "End of commands - execute and show results.")
        _hline("abort", "Cancel the sequence without executing.")
        print()  # noqa: T201

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def _cmd_exit(self) -> None:
        """Handle the 'exit' command, which closes all connections and exits."""
        n = len(self._registry)
        if n:
            self.print_info(f"Closing {n} connection(s)...")
        self.shutdown()
        print(fmt.dim("  Goodbye."))  # noqa: T201
        sys.exit(0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_storage(self, sid: str) -> DatabaseClient:
        """Return the client for *sid*, or print an error if not found."""
        client = self._registry.get(sid)
        if client is None:
            self.print_error(
                f"No active storage {fmt.storage_id(sid)}. "
                "Use 'list storages' to see open connections."
            )

        if client is None:
            supported_backends = ", ".join(sorted(_BACKEND_PARAMS.keys()))
            raise UnsupportedDatabaseBackendError(sid, supported_backends)

        return client


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _tokenise(line: str) -> list[str]:
    """Split a line into tokens, respecting quoted strings."""
    try:
        return shlex.split(line)
    except ValueError:
        return line.split()


def _hline(cmd: str, desc: str) -> None:
    print(f"    {fmt.cyan(cmd):<48}  {fmt.dim(desc)}")  # noqa: T201
