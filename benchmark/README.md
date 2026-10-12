# Shapley estimator benchmark

[Results](https://www.rtealwitter.com/shapiq/) · [About](site/about.md) ·
[Scientific plan](COMPREHENSIVE.md) · [Recipes](suites/comprehensive.csv)

One benchmark compares **22 estimator configurations** on individual predictions,
data valuation and feature selection. Its design spans **24 datasets**, 1,270
recipes and four game-construction seeds: **5,080 intended instances** before
qualification and resource exclusions. Each supported target uses an exact
reference for its frozen game. Intended counts are not successful coverage.

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

For the full design, generate the suite from the CSV, then pass that suite to
`prepare`. This is substantial preparation, not a quick smoke test:

```bash
uv run python -m shapiq_benchmark.comprehensive benchmark/suites/comprehensive.csv benchmark/results/suite.json
```

- **Scientific inputs:** recipe CSV, model profiles, construction seeds, estimator
  parameters, three estimator seeds and nine budgets (0.5–128 × players).
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
- Authenticated memory failures use **128 GiB**, or **512 GiB** for higher-order
  KernelSHAPIQ, InconsistentKernelSHAPIQ, RegressionFSII and RegressionFBII.
  `runtime.recovery_memory` implements this rule without lowering an existing cap.
- Timeout-only retries retain their previous memory cap. Parent allocations need
  enough physical memory and a compatible hard address-space limit.
- Startup/output failures are operational failures, separate from estimator
  timeouts. Failed cells have no invented scores or timings. The generic runner
  does not retry failures automatically; campaign recovery records one scored
  owner per cell and retains previous attempts as evidence.

These are defaults for **new runs**. Historical successes keep their actual
recorded limits; active 600-second and 3,600-second campaign workers are unchanged.

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

The current campaign has separate qualified patches for **SVARMIQ caching**,
**SHAPIQ contribution caching** and **ProxySPEX high-dimensional sampling weights**.
They remain outside deployed `shapiq`. Use the source files and replay adapters
from the corresponding authenticated campaign/release supplement; the generic
snapshot ZIP does not package these patches or the full environment. Final
combined publication and its reproduction supplement are still pending.

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
- Preparation originally allowed two hours per reference; authorized recovery
  allows four. Preserve failures and exclusions rather than inventing truth.
- Event-only monitoring remains active; periodic progress reminders are disabled.
  Publication waits for actual completed outputs and verified coverage/scores,
  with 100% supported success sought for every method except ShaplEIG and SPEX.
- Keep the existing website until the replacement passes. The final combined
  exporter must preserve source overlays and authenticate recovery ownership.

Edit [site/about.md](site/about.md), then run:

```bash
UV_NO_SYNC=1 uv run python benchmark/render_about.py
UV_NO_SYNC=1 uv run python benchmark/render_about.py --check
```

[site/assets.json](site/assets.json) lists every deployed asset. Related guides:
[partitioned reports](PARTITIONED_REPORTS.md), [publication](MATRIX_PUBLICATION.md),
[operations](OPERATIONS.md), [estimator notes](ESTIMATOR_NOTES.md), and
[handoff checklist](HANDOFF.md). FOCUSED.md and ROADMAP.md are historical plans.
