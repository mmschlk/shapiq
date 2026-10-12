# Compact benchmark handoff

The active design is [scope version 2](COMPREHENSIVE.md): 512 recipes and 2,048
intended instances across all 24 datasets. Estimator fixes and speedups remain
in their separate PRs; the benchmark does not silently change core algorithms.

## Complete and publish the selected cohort

- Authenticate exact recipe, game, source, settings, budget and seed matches before
  reusing historical results. Preserve successful matched outcomes and original
  run IDs; the broader historical cohort is not compact coverage.
- Complete 100% successful supported coverage except ShaplEIG and SPEX. Distinguish
  reference coverage, attempted coverage and successful coverage; keep exclusions,
  failures and missing work visible without invented scores.
- Run one consolidated ownership, reference, score and coverage validation.
  Publish the replacement data and About page together, with source-aware
  reproduction artifacts, then verify the live scores, coverage and filters.

## Keep the integration small

- Keep reusable construction, execution, scoring and result I/O in
  `src/shapiq_benchmark`. Keep Slurm placement, staging and monitoring optional.
- Maintain one active plan, the mathematical About page, portable README commands
  and the short operations guide. Do not revive removed campaign launchers.
- Preserve explicit execution limits, source overlays and original result mappings
  in replay instructions. A generic snapshot ZIP alone may not include historical
  patches or the full environment.

## Delete obsolete material safely

- Remove unused plans, launchers, duplicate working copies and rebuildable caches.
  Check executable imports, fixture dependencies and registered worktrees first.
- Retain the evidence required for selected results and any promised published
  reproduction. Materialize and verify those dependencies before deleting their
  old locations; out-of-scope artifacts need not all be retained indefinitely.
- Keep large campaign artifacts on lab storage. Do not delete an entire historical
  directory merely because its name is old: staging and audit pins can cross it.
- Record actual deletions and remaining dependency holds in the campaign cleanup
  manifest. Avoid rewriting historical receipts to make missing files appear valid.

## Leave an actionable handoff

Record the current scope, owned jobs, actual CPU accounting, remaining cells,
artifact locations, source/replay requirements and publication status. Run the
relevant Python and browser tests, resolve PR failures, and update the PR description
to the final compact implementation. The README should remain sufficient for a
small local comparison and adding a private estimator.
