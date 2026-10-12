# Shapley estimator benchmark

[Results](https://www.rtealwitter.com/shapiq/) · [About](site/about.md) ·
[Scientific plan](COMPREHENSIVE.md) · [Recipes](suites/comprehensive.csv)

The active **scope version 2** compares **22 estimator configurations** across
individual predictions, data valuation and feature selection on **24 datasets**.
Its **512 recipes × four construction seeds = 2,048 prepared instances** have
exact references and **100% successful supported coverage for the 20 required
estimators** (1,071,525 evaluations). ShaplEIG and SPEX remain the two declared
partial-coverage methods. The [verified compact release](https://github.com/rtealwitter/shapiq/releases/tag/benchmark-compact-2026-10-12)
is published and live.
The selection retains 477 core recipes and adds 35 balanced KNN/TNN recipes;
35 core recipes failing the model-quality check on any seed are excluded in full. Generic linear/random-forest games have at most 12 players;
RF tree games use 8/16/32 features; selected KNN/TNN games use 32/64 training rows.
Valuation input width is exactly `min(12, dataset width)`, with no width sweep or
special-model contrasts. Historical broader results remain a separate cohort
and do not enter the published compact comparison.

## Prepare → evaluate → report

Use a separate development environment; never install into one used by running
jobs. The small pilot exercises the same snapshot, counted-query runner and
report format as the full design:

```bash
uv sync --locked --extra benchmark
uv run python -m shapiq_benchmark.prepare --suite benchmark/suites/pilot.json --output benchmark/results/pilot
uv run python -m shapiq_benchmark.runner --snapshot benchmark/results/pilot --output benchmark/results/baselines --settings benchmark/settings.json --max-seconds 86400
uv run python -m shapiq_benchmark.report --results benchmark/results/baselines/results.json --output benchmark/results/report
uv run python -m http.server 8000 --bind 127.0.0.1 --directory benchmark/results/report
```

Generate the selected suite from the CSV for replay or a new independent run.
The active campaign reuses authenticated prepared snapshots; it does not rerun
preparation merely to regenerate this manifest:

```bash
uv run python -m shapiq_benchmark.comprehensive benchmark/suites/comprehensive.csv benchmark/results/suite.json
```

- **Scientific inputs:** recipe CSV, model profiles, construction seeds, estimator
  parameters, three estimator seeds and seven budgets (0.5–32 × players).
- **Frozen snapshot:** authenticated game artifacts and exact coefficient vectors.
  Evaluation does not rebuild games or retune models.
- **Execution settings:** [settings.json](settings.json), kept separate from the
  scientific inputs. Explicit `--settings` overrides historical suite timeouts.
- **Resume:** repeat the runner command with `--resume`. Completed cells are
  skipped; changed source, settings, games or hardware reject resumption. A
  completed allocation can still leave cells pending: inspect `campaign.complete`.
- **Parallel execution:** `benchmark/sweep.py SNAPSHOT OUTPUT --workers N
  --settings benchmark/settings.json --seconds 86400` assigns disjoint game panels
  to allocated CPU cores. Repeating it resumes the same shards. It does not submit
  Slurm jobs or reproduce the campaign's recovery/admission history.

## Execution limits and failures

| Phase | Fresh-run allowance | Includes |
| --- | ---: | --- |
| Startup | 600 seconds | Imports, game loading, estimator construction |
| Estimation | 3,600 seconds | `approximate`, including game queries |
| Output | 600 seconds | Scoring and serialization |

The runner reserves all three allowances before starting a cell. `--max-seconds`
is a bounded worker invocation; use a larger allocation allowance for process
cleanup and checkpoint writes. Timings record both constructor-plus-estimation
(`seconds`) and estimation alone (`execution_seconds`). Shared-node timings are
**diagnostic**, not isolated hardware comparisons.

- Normal workers have an **8 GiB virtual address-space cap**. `--memory-gb`
  overrides it explicitly and is recorded in provenance.
- Parent allocations need enough physical memory and a compatible hard
  address-space limit. Historical recovery memory policies are replay metadata,
  not default allocations for the compact plan.
- Startup/output failures are operational failures, separate from estimator
  timeouts. Failed cells have no invented scores or timings. The generic runner
  does not retry failures automatically; campaign recovery records one scored
  owner per cell and retains previous attempts as evidence.

These are defaults for **new runs**. Historical successes keep their actual
recorded limits. Superseded campaign workers have been cancelled.

## Reproduce the executed estimator source

A fresh checkout and the recipe CSV reproduce the design, not necessarily every
historical measurement. Exact replay also needs the recorded source revision,
locked dependencies, frozen snapshot and any qualified source patch.

`--overlays overlays.json` supports trusted, per-method Python module patches in
fresh child processes. Paths are relative to the manifest; a descriptor is:

```json
{
  "SVARMIQ": {
    "module": "shapiq.approximator.montecarlo.base",
    "path": "sources/svarmiq_base.py",
    "sha256": "<64-character SHA256 of patched file>",
    "base_sha256": "<64-character SHA256 of original module>"
  }
}
```

Both files are authenticated before execution. Patch module/hash identities are
recorded with the base source identity and prevent incompatible resumption or
merging. Each method gets its own subprocess, so SVARMIQ and SHAPIQ can use their
separately qualified versions of the same module. Overlay code is trusted local
code, like an estimator plugin; it is not a sandbox.

Historical campaigns have separate qualified patches for **SVARMIQ caching**,
**SHAPIQ contribution caching** and **ProxySPEX high-dimensional sampling weights**.
They remain outside deployed `shapiq`. Use the source files and replay adapters
from the corresponding authenticated campaign/release supplement; the generic
snapshot ZIP does not package these patches or the full environment. The compact
release includes its own authenticated source/replay supplement.

## Add an estimator

Copy [example_estimator.py](example_estimator.py), implement its factory, then:

```bash
uv run python -m shapiq_benchmark.runner --snapshot benchmark/results/pilot --output benchmark/results/candidate --candidate benchmark/example_estimator.py:factory --settings benchmark/settings.json --max-seconds 86400
```

The factory receives `n`, `index`, `order` and `seed`, and returns an estimator
with `approximate(budget, game)`. Every requested coalition row counts, including
duplicates. Candidates stay private and cannot be combined with source overlays.
Pass both baseline and candidate `results.json` files to the report command for
comparison on the same snapshot.

## Campaign operations and publication

- Reusable construction, execution, scoring and export live in
  `src/shapiq_benchmark`. Slurm submission, dependency staging and monitoring are
  optional operations; they do not belong in estimator implementations.
- The active campaign's `WAKEUP.md`, `state.json` and `budget.json` are authoritative
  for jobs and ownership. Admission uses its capacity tree to stay within **1,000
  concurrent CPUs**, while leaving monitor job slots available. The total CPU-hour
  cap was waived; allocation costs remain recorded. No production GPUs.
- Preserve preparation failures and exclusions rather than inventing truth.
  Reuse old outcomes only when their exact inputs match the version 2 design.
- The compact campaign is complete and live-verified. Its watcher and reminder
  schedule are stopped; do not restart historical jobs.
- The published export preserves source overlays and authenticates recovery
  ownership. New runs and future replacements need their own validation.

Edit [site/about.md](site/about.md), then run:

```bash
UV_NO_SYNC=1 uv run python benchmark/render_about.py
UV_NO_SYNC=1 uv run python benchmark/render_about.py --check
```

[site/assets.json](site/assets.json) lists every deployed asset. Related guides:
[partitioned reports](PARTITIONED_REPORTS.md), [publication](MATRIX_PUBLICATION.md),
[operations](OPERATIONS.md), [estimator notes](ESTIMATOR_NOTES.md), and
[handoff checklist](HANDOFF.md). Older campaigns are historical; COMPREHENSIVE.md is the active scope version 2 plan.
