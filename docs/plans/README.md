# Working Plans

Living planning documents for the current phase of the shapiq v2 rewrite. Each issue file tracks
one goal from problem to acceptance criteria: update its status and work breakdown as work lands,
and fold durable outcomes into `docs/adr/` and `docs/design/` (then delete the issue file) once
they are recorded there.

## Where the rewrite stands

The functional approximator core is built and tested: permutation sampling for SV, SII, and STII,
the kernel-regression FSII approximator, and the exact explainer for SV, BV, SII, BII, STII, and
FSII. Budget semantics follow ADR 0004 (exact spending, seed prelude, pending masking,
deduplication as the sole exception), sampled streams are bit-identical under budget splits, and
approximation history, rollback, and deduplication are covered by the test suite (78 tests green
as of writing). Deliberately deferred so far: everything past synthetic games — real models,
masking, and vector-valued predictions.

## North star

Explain a torch model with baseline masking, end to end:

> **Masker** + model → **ModelMaskedPredictor** → **LinkFunction** → **MaskedGame** →
> **Approximator** → **ExplanationArray**

This was postponed until the approximator API had been stress-tested by estimators; issue 2
delivers it.

## Issues

| # | File | Goal | Status |
| --- | --- | --- | --- |
| 1 | [Interaction index objects](issue-1-interaction-index-objects.md) | Interaction indices become typed objects; index support is visible to the type checker | done (ADR 0005) |
| 2 | [Torch baseline example](issue-2-torch-baseline-example.md) | End-to-end example explaining a torch model with baseline masking | done |
| 3 | [Vector-valued games](issue-3-vector-valued-games.md) | Un-bake the scalar-values-only enforcement from the evidence layer | done (ADR 0006) |
| 4 | [Sampling performance](issue-4-sampling-performance.md) | Remove per-unit Python loops from the sampling path | not started |
| 5 | [Index-dispatched entry points](issue-5-index-dispatched-entry-points.md) | One entry point per method family; the index object dispatches to concrete implementations | done (ADR 0007) |
| 6 | [Exact index families](issue-6-exact-index-families.md) | Port the full exact index zoo with the capability taxonomy and declared generalizations | done (ADR 0008) |

## Recommended order: 1 → 3 → 2 → 4

- **Issue 1 first.** It changes the grammar — constructor signatures and index metadata — that
  every later signature inherits; doing it late means migrating more call sites.
- **Issue 3 second.** It changes the evidence contract that issue 2's full milestone needs.
- **Issue 2 third.** It delivers the north star and supplies the realistic workload that issue 4
  benchmarks against. Its milestone A (scalar link) has no dependencies and may start any time.
- **Issue 4 last.** Measure against issue 2's workload before optimizing.

Issues 3 and 4 both modify `src/shapiq/explainers/_evidence.py` — run them in sequence, not in
parallel worktrees.

**Issue 5** requires issue 1 and does not block the north star; it slots naturally either right
after issue 1 or after issue 4. Issue 1 should anticipate it by dispatching on index types and
capabilities internally, never on name strings.

## Working agreement

An issue is done when:

- `uv run pytest tests/shapiq`, `uv run ty check src/shapiq`, and
  `uv run --group lint prek run --all-files` are green;
- `CONTEXT.md` is updated whenever project language changes;
- decisions that are hard to reverse, surprising without context, and the result of a real
  trade-off are recorded as ADRs (issues 1 and 3 are each expected to produce one; the next free
  numbers are 0005 and 0006);
- the issue file's status line and work breakdown reflect reality.
