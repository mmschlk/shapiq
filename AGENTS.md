# AGENTS.md

Before changing code or documentation, read `CONTEXT.md` for project language and `docs/adr/` for accepted design decisions when that directory exists.

Keep `CONTEXT.md` glossary-only. Record durable design decisions as ADRs only when the decision is hard to reverse, surprising without context, and the result of a real trade-off.

After code, configuration, or documentation changes, run `uv run --group lint prek run --all-files` when feasible.
