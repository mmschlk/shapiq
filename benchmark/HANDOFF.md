# Compact benchmark handoff

The active design is [scope version 2](COMPREHENSIVE.md): 512 recipes and 2,048
prepared instances across all 24 datasets. Estimator fixes and speedups remain
in their separate PRs; the benchmark does not silently change core algorithms.

## Published result

- [Live benchmark](https://www.rtealwitter.com/shapiq/) and
  [reproduction release](https://github.com/rtealwitter/shapiq/releases/tag/benchmark-compact-2026-10-12).
- All 20 required estimators: **1,071,525 / 1,071,525 successful supported
  evaluations**. ShaplEIG and SPEX are the two explicit partial-coverage methods.
- Final outputs, scores, coverage, filters and live delivery were independently
  verified. The campaign is complete; the watcher and reminder schedule are off.
- `benchmark/site/data-source.json` pins the deployed release bytes. Scientific
  inputs and replay instructions are in the README and release artifacts.

## Requirements for future replacements

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

## Upstream integration remains open

The published compact release is complete, but PR #602 is **not merge-ready**
against upstream `da6381c3` (PR #616). That change replaces the benchmark/game
APIs. Resolve this separately from website edits; do not overwrite either API
with an `ours`/`theirs` conflict resolution.

1. Preserve upstream `shapiq_benchmark.runner.run`, `models.build_model` and
   `datasets.load_dataset`. Our same-named helpers have different contracts;
   move them to explicit `campaign_runner`, `campaign_models` and
   `campaign_datasets` modules, updating their imports, CLI commands and tests.
   Upstream's new `datasets/` package otherwise silently shadows our `datasets.py`.
2. Adapt construction code in `prepare.py`, `families.py`, `games.py`,
   `structured.py` and `native_tree_backend.py`. Upstream removes
   `shapiq.datasets`, `shapiq_games.datasets`, `shapiq_games.benchmark`,
   `shapiq.explainer.nn.games`, `shapiq.explainer.product_kernel.game` and
   `shapiq.tree.interventional.game`; their replacements live in
   `shapiq_benchmark.datasets` and `shapiq_games`. Verify constructor semantics,
   preprocessing, coalition payoffs and exact references before treating new
   preparations as equivalent. Published replay keeps its pinned source/data.
3. Retain both branches' dataset/model/runner/metric tests under distinct names.
   Merge dependency extras and regenerate the lock in an isolated environment.
   Upstream now type-checks the whole benchmark package and changes native tree
   kernels: rebuild native extensions there and run both API test suites.
4. Check a small portable prepare/evaluate/report run and a frozen-result replay,
   then rerun full CI before calling the integration ready. Do not regenerate or
   relabel the published cohort to make the new APIs fit.
