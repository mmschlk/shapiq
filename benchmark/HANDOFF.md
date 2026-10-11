# Benchmark handoff cleanup plan

The goal is a reusable benchmark package and a small, understandable integration
with shapiq. The benchmark PR currently changes no files under `src/shapiq`;
estimator fixes and speedups stay in their separate PRs.

## Finish the results

- Complete the supported, never-attempted evaluations described in
  [COMPREHENSIVE.md](COMPREHENSIVE.md). Keep failures and missing references visible.
- Preserve frozen games, source, seeds, prior attempts and published run IDs.
  Distinguish reference coverage, attempted coverage and successful coverage.
- Publish the combined cohort and verify the actual deployed scores and filters.

## Make the entry points clear

- Rewrite `benchmark/README.md` around the current design and a short local example:
  prepare games, run estimators, summarize results, export the browser report.
- Keep one scientific plan, the mathematical About page and one operations guide.
  Move superseded campaign narratives out of the main reading path.
- Rewrite PR #602's description around its final scope and current validation;
  remove references to superseded phases and counts.

## Separate reusable code from campaign operations

- Keep game construction, execution, scoring, result I/O and export in
  `src/shapiq_benchmark`, with explicit inputs and no required Hopper paths.
- Keep Slurm launchers, node-local dependency staging and monitoring in optional
  operational scripts. Document the pinned environment and concurrency control.
- Inventory imports before retiring legacy campaign/matrix helpers: historical
  exporters still use them. Avoid a broad refactor while jobs are active.

## Remove only disposable material

- Retain authenticated snapshots, journals, source revisions, release artifacts
  and the manifests needed to reproduce published results.
- After publication, identify rebuildable caches, duplicate working copies and
  obsolete launch scripts. Record what replaces each item before removing it.
- Keep large campaign artifacts on lab storage, outside the package and website.

## Leave a reproducible handoff

- Provide one small portable end-to-end example and commands for reproducing a
  published game, adding an estimator, exporting a report and publishing the site.
- Run the existing relevant Python/browser tests and resolve current PR failures.
- Record final artifact locations, current limitations and remaining maintenance
  work. Keep library changes minimal and independently reviewable.
