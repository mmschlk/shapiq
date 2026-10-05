# Shapiq estimator benchmark

[Interactive results](https://www.rtealwitter.com/shapiq/) ·
[How it works](https://www.rtealwitter.com/shapiq/about.html) ·
[Implementation PR](https://github.com/mmschlk/shapiq/pull/602) ·
[Discussion](https://github.com/mmschlk/shapiq/issues/601)

Compare 22 public estimator classes against exact answers on frozen games.
Select a target, family, dataset, model, player count or query budget, then inspect
accuracy, coverage and timing. Python prepares games and runs estimators; the
website is plain HTML/CSS/JavaScript with SVG charts and no frontend build step.
You can test a private estimator locally without uploading anything.

## Current status

The selected seven-phase rollout is published: **248 game instances, 1,188 target
definitions and 235,224 recorded cells**, including 90,070 successful evaluations.
Unsupported targets and failures remain visible. This is a declared cohort, not
completed coverage of every dataset/model/construction combination.

- [Phase-seven data and reproduction archives](https://github.com/rtealwitter/shapiq/releases/tag/benchmark-phase7-svd-2026-10-04)
- [Current website data pointer](site/data-source.json): the compact serialization
  of the same audited results; all seven reproduction archives remain unchanged.
- [Roadmap and qualification limits](ROADMAP.md): the full 63-dataset expansion
  is running separately in 38 waves. Its results are not yet the live cohort.
- [Operational and historical reference](OPERATIONS.md): recovery, exact
  preparation, archived campaigns and detailed commands.

The published settings retain their recorded estimator revisions and parameters.
Later estimator fixes or defaults do not silently change existing measurements.

## Try a small local comparison

From the repository root, in your own development environment:

```bash
uv sync --locked --extra benchmark
uv run python -m shapiq_benchmark.prepare --suite benchmark/suites/pilot.json --output benchmark/results/pilot
uv run python -m shapiq_benchmark.runner --snapshot benchmark/results/pilot --output benchmark/results/baselines
```

The pilot uses one California Housing point, a fitted tree, three estimators,
three budgets and three seeds: 27 runs. Exact truth uses all 256 coalitions of an
eight-player game. This checks integration, not general estimator quality. To
try larger structured games, use [suites/structured.json](suites/structured.json)
and a separate output directory.

**On a shared campaign environment, do not run `uv sync` or install packages.**
Use its pinned interpreter and `UV_NO_SYNC=1`; preserve frozen source, package
bytes and cached inputs. Set `PYTHONPATH` to the intended checkout's `src`.

## Evaluate a private estimator

Copy [example_estimator.py](example_estimator.py) and replace its factory:

```bash
uv run python -m shapiq_benchmark.runner --snapshot benchmark/results/pilot --output benchmark/results/candidate --candidate benchmark/example_estimator.py:factory
uv run python -m shapiq_benchmark.report --results benchmark/results/baselines/results.json benchmark/results/candidate/results.json --output benchmark/results/local-report
uv run python -m http.server 8000 --bind 127.0.0.1 --directory benchmark/results/local-report
```

Open `http://localhost:8000`. The factory receives `n`, `index`, `order` and `seed`.
It returns an object whose `approximate(budget, game)` method returns
`InteractionValues`. The game accepts Boolean coalition matrices and charges
**every requested row**, including duplicates. Invalid or over-budget results
fail the cell. Candidate files are trusted Python code.

Reuse saved baseline accuracy only on the identical snapshot; rerun comparisons
on the same hardware for timing. Private candidates do not access the production
duplicate registry or execute historical baseline constructor options. Built-in
reruns still need compatible constructors and the configured registry. Otherwise,
prepare a local suite and run both methods there. Reports reject snapshot,
method-version and duplicate-cell conflicts. Nothing uploads automatically.

For the public cohort, use the [phase-seven reproduction archives](https://github.com/rtealwitter/shapiq/releases/tag/benchmark-phase7-svd-2026-10-04).
Each archive includes frozen games, truth, checksums, provenance and candidate
commands. Use `--games` and `--methods` for a smaller comparison; no model refit
is needed. To inspect exported reports, use **Open local** and select `data.json`
along with all its companion files.

## Read the rankings

- **Median nMSE** is the default: squared attribution error divided by exact
  attribution energy, excluding the empty baseline. Zero is exact; one matches
  predicting all zeros. The weighted mean exposes extreme errors the median can
  hide. Zero or negligible signal is excluded consistently across methods.
- **Weights** balance family → configuration → instance → budget → estimator
  seed. Failed or missing runs are never zero error; partial scores renormalize
  successful weights. Check coverage before comparing methods.
- **Elo** compares paired successful cells and is centered at 1,000. Higher means
  more frequent wins, not necessarily smaller average error. Custom selections
  without a matching published preset retain nMSE but withhold Elo and history.
- **Budget** is an allowance, expressed as queries per player (`B/d`); actual
  queries may be fewer. The nine ratios are 0.5, 1, 2, 4, 8, 16, 32, 64 and 128.
  The budget cap chooses the largest measured budget within each game's limit.
- **Time** distinguishes measured estimator time from estimated uncached cost.
  Cached payoff charges are batch-amortized estimates, not fresh end-to-end
  measurements. Shared-node timings are diagnostic; hardware profiles matter.
- **History** places today's complete-panel results at publication dates. It
  does not reconstruct historical benchmark results.

Targets have separate rankings. The default SV view groups SVARMIQ under SVARM
and KernelSHAPIQ under KernelSHAP because each pair has the same order-one
configuration. **All variants** restores their separate records; scores are not
pooled. Plot-family filters are independent of table filters.

[Full scoring, weighting, uncertainty and display rules](OPERATIONS.md#read-the-rankings) ·
[Estimator findings](ESTIMATOR_NOTES.md) ·
[About the game constructions and exact answers](site/about.html)

## Continuing through the phases

The active full matrix lives at
`/hopper/groups/witterlab/rwitter/shapiq-benchmark-matrix-expansion`.
Read its `state.json` and the coordinator's
`benchmark/results/full-rollout/WAKEUP.md` before acting. The latter directory is
a symlink to `/hopper/groups/witterlab/rwitter/shapiq-benchmark-full-rollout`.
Reviewed recovery journals override superseded job IDs; do not resubmit an
existing campaign from a historical command template.

The reviewed 38-wave plan separates preparation, evaluation and audit lanes:
**48 preparation CPUs + 64 evaluation CPUs + one sequential audit CPU = 113**,
within the approved **128-core ceiling**, with no production GPUs reserved.
The first wave is still in preparation recovery; the replacement runtime is
under review. Original preparation and recovery retain their reviewed limits.
Evaluation waits for its own preparation checks and stays ordered for duplicate
ownership. Later preparation can overlap evaluation; audits gate publication,
not subsequent computation. The scheduling overlay in
`operational/audit-independent-pipeline/` retains original job IDs and science.

The existing watcher runs every five minutes, with an implementation heartbeat
while work continues. It wakes the authorized session; it neither submits jobs
nor publishes results. Acknowledge a wake with:

```bash
python benchmark/watch_campaign.py /shared/coordinator --acknowledge
```

Keep its exact job IDs and next action current. Mark it complete only after the
final audited live release, or pause/cancel at the user's request. Large caches
belong in lab storage. Moving a campaign requires stopped writers, verified
copies and preserved authenticated paths. Keep ignored dataset caches in frozen
worktrees and never change a running campaign's environment.

[Detailed bounded-rollout operations](OPERATIONS.md#bounded-rollout-operations) ·
[Corrected-method replacements](OPERATIONS.md#replacing-a-corrected-estimator) ·
[CPU backend recovery](OPERATIONS.md#recovering-a-stopped-gpu-preparation-on-cpu)

## Export and publish

After a wave's final audit and independent review, use a publication plan that
pins those receipts, source equivalence and the authenticated published history:

```bash
UV_NO_SYNC=1 uv run python benchmark/export_matrix.py /shared/publication-plan.json /shared/release/data --plan-sha256 PLAN_SHA256 --database /shared/release/export.sqlite --cache-dir /shared/publication-cache
```

The command combines authenticated records in temporary disk storage and writes
a new partitioned data directory. The database is removed on exit. It does not
publish anything. Use lab storage; output, database and cache locations must be
separate. The larger matrix export still needs independent output review,
browser/hosting qualification and reproduction archives before publication.
See [partitioned report APIs and limits](PARTITIONED_REPORTS.md).

For smaller cumulative campaigns, the legacy exporter remains available:

```bash
UV_NO_SYNC=1 uv run python benchmark/export_phase.py /shared/campaign/campaign /shared/release/site --through-phase 3
```

Optional `--cache-dir` avoids repeated normalization, while still validating raw
inputs every run. It is acceleration, not publication approval, and does not
remove this older exporter's whole-panel memory requirement. `--supplement`,
`--replacements` and `--backend-supersession` preserve explicit recovery lineage;
see [the operational reference](OPERATIONS.md#bounded-rollout-operations).

Website data omits exact truth, raw coefficient vectors and private paths.
Reproduction ZIPs retain the numerical artifacts, hashes and source provenance;
both public export paths reject private candidates. Keep raw prepared roles and
apply publication-quality overlays through the cumulative exporter.

The live compact layout uses `data.json` plus `records-*.json` target shards.
Upload every declared companion before changing [site/data-source.json](site/data-source.json).
Pages verifies the pinned manifest and companions. The browser restores shared
worker profiles and per-record diagnostics losslessly; original archives remain
unchanged. Local legacy single-file reports still work.

Pages and local report exports share the explicit [site-asset manifest](site/assets.json).
The Pages workflow runs when `BENCHMARK_PAGES=true`
and the repository uses GitHub Actions for Pages. It does not run experiments
or upload local results. For standalone report/ZIP commands, see
[legacy exports](OPERATIONS.md#legacy-report-and-archive-export).

## Code map

Python library modules below live in [`src/shapiq_benchmark/`](../src/shapiq_benchmark/).
Entries starting with `benchmark/` share that directory prefix within their row.

| Responsibility | Entry points |
| --- | --- |
| Declare datasets, recipes, models and budgets | `protocol.py`, `dataset_catalog.py`, `matrix.py`, `models.py`, `planning.py` (shared budget grid) |
| Qualify and freeze games | `qualification.py`, `prepare.py`, `materialize.py`, `families.py`, `media.py`, `games.py`, `structured.py` |
| Exact table answers and game diagnostics | `exact.py`, `spectrum.py`, `payoff_cache.py` |
| Count queries, score and checkpoint cells | `runner.py`, `execution.py`, `results_io.py` |
| Coordinate bounded campaigns and monitoring | `benchmark/queue_phases.py`, `phase_batch.py`, `watch_campaign.py` (all under `benchmark/`) |
| Authenticate cumulative composition | `campaign.py`, `matrix_publication.py`, `published.py`, `publication_cache.py` |
| Store rows and compute exact global summaries | `record_store.py`, `summary.py` |
| Export reports and reproduction archives | `report.py`, `partitioned.py`, `bundle.py`, `benchmark/export_matrix.py`, `benchmark/export_phase.py` |
| Load reports and reduce selected panels | `benchmark/site/records.js`, `partitions.js`, `query.js`, `query-worker.js`, `partition-client.js` (all under `benchmark/site/`) |
| Render tables, charts and method descriptions | `benchmark/site/app.js`, `charts.js`, `methods.js`, `style.css` |
| About, catalogs and downloads | `benchmark/site/about.html`, `about.js`, `protocol.js`, `partition-about.js`, `partition-download.js` |

Start with `runner.py` for estimator execution, `summary.py` for scores and the
appropriate exporter for output. The About page shares catalog rendering through
`protocol.js`; its generated `about.json` contains metadata rather than all rows.
The browser has no build step.

## Further reference

- [Scientific design](DESIGN.md) and [independent audits](AUDITS.md)
- [Current rollout and remaining work](ROADMAP.md)
- [Operational procedures and historical campaigns](OPERATIONS.md)
- [Estimator-specific findings](ESTIMATOR_NOTES.md)
