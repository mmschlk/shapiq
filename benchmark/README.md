# Shapiq estimator benchmark

[Open the live research preview](https://www.rtealwitter.com/shapiq/).

An offline benchmark and static comparison website. Python prepares frozen games,
runs estimators, and writes results; the browser only reads those results. A local
paper implementation uses the same runner without uploading anything.

Implemented in five independently reviewed phases on this branch.
The [design and scientific protocol](DESIGN.md) explain the longer-term scope.
[Issue #601](https://github.com/mmschlk/shapiq/issues/601) is the discussion home;
[PR #602](https://github.com/mmschlk/shapiq/pull/602) contains the implementation.

## First local comparison

From the repository root:

```bash
uv run python -m shapiq_benchmark.prepare --suite benchmark/suites/pilot.json --output benchmark/results/pilot
uv run python -m shapiq_benchmark.runner --snapshot benchmark/results/pilot --output benchmark/results/baselines
```

The pilot uses one held-out California Housing point and a fitted tree, three
Shapley estimators, three budgets, and three seeds: 27 runs. Its exact answers
come from all 256 coalitions of the fixed eight-feature game. It is a small
integration pilot, not evidence of general estimator quality.

The snapshot holds the game, exact coefficients, recipe, and provenance. The
runner counts every requested coalition row, including repeated requests, and
rejects over-budget or invalid outputs. Results contain raw MSE and normalized
MSE (squared error divided by ground-truth coefficient energy, excluding the
baseline). Zero-energy games have no defined nMSE. Failures remain visible.

## Evaluate a local estimator

```bash
uv run python -m shapiq_benchmark.runner --snapshot benchmark/results/pilot --output benchmark/results/candidate --candidate benchmark/example_estimator.py:factory
```

Copy [example_estimator.py](example_estimator.py) and replace its constructor with
your method. The factory receives `n`, `index`, `order`, and `seed`; the returned
object implements `approximate(budget, game)` and returns `InteractionValues`.
`game` accepts a Boolean coalition matrix and charges every row. Candidate files
are trusted Python code, not a security sandbox. Only the candidate runs; saved
baseline accuracy results can be compared later. Runtime comparisons require
rerunning both on the same hardware. Nothing uploads automatically.

## Open the interactive report

```bash
uv run python -m shapiq_benchmark.report --results benchmark/results/baselines/results.json benchmark/results/candidate/results.json --output benchmark/results/local-report
uv run python -m http.server 8000 --bind 127.0.0.1 --directory benchmark/results/local-report
```

Open `http://localhost:8000`. The report compares matching snapshot results and
rejects conflicting method versions or duplicate measurements. It removes truth,
raw coefficient arrays, and local file paths. You can also open `index.html`
directly and select the exported `data.json` through **Open local**.

For a public preview, export baseline results with `--public --output benchmark/site`.
Private candidates are rejected. The Pages workflow deploys only this public site
when the repository variable `BENCHMARK_PAGES` is `true` and Pages uses GitHub
Actions as its source. It does not run experiments or publish local results.
The static page uses ordinary HTML/CSS/JavaScript and SVG charts; no frontend
framework, external chart download, database, or build step is required.

## Larger games and interactions

```bash
uv run python -m shapiq_benchmark.prepare --suite benchmark/suites/structured.json --output benchmark/results/structured
uv run python -m shapiq_benchmark.runner --snapshot benchmark/results/structured --output benchmark/results/structured-baselines
uv run python -m shapiq_benchmark.report --results benchmark/results/structured-baselines/results.json --output benchmark/results/structured-report
```

The structured suite uses existing shapiq games: a real fitted forest with all
30 Breast Cancer features (SV and pairwise k-SII), and KNN data valuation with
128 training examples as players (SV). Each truth route must first agree with
exhaustive enumeration on an eight-player counterpart. Large games use their
structured exact solvers and live coalition calls, not powerset tables.

Model reconstruction is checked against the frozen recipe; arrays are stored
without pickle. Preparation and truth computation happen outside estimator
budgets. Unsupported method/target combinations remain visible separately from
failed runs. Tree probabilities, empirical background rows, and the KNN utility
are explicit in each snapshot. These panels illustrate scale; they do not yet
represent all models, datasets, or applications.

## Run a bounded campaign and resume it

```bash
uv run python -m shapiq_benchmark.runner --list-methods
uv run python -m shapiq_benchmark.runner --snapshot benchmark/results/pilot --output benchmark/results/resumable --max-runs 3
uv run python -m shapiq_benchmark.runner --snapshot benchmark/results/pilot --output benchmark/results/resumable --resume
```

Each cell runs in a separate subprocess. `--timeout` caps the whole worker,
including imports and game reconstruction; `seconds` measures estimator
construction plus approximation only. `--memory-gb` optionally caps worker virtual
address space on POSIX. Workers use one native thread. A failed or timed-out cell
is recorded and does not stop the next cell. Completed failures are retained on
resume; use a new output directory after fixing a method or changing limits.

`--max-runs` and `--max-seconds` bound each campaign invocation. A cell needs its
full time allowance before starting; otherwise it remains pending for resume.
Existing outputs require `--resume`, with matching snapshot, code, methods,
software, hardware, and per-cell limits. Checkpoints are written after each cell.
The [catalog suite](suites/catalog.json) lists all 22 public estimators for a small
coverage probe. Optional backends are not installed automatically; unavailable
implementations and unsupported targets remain visible. The common adapter passes
the requested target/order and seed to supported constructor parameters; other
parameters keep library defaults. In particular, the SV configuration of
`kADDSHAP` uses order one, not its default order-two target. These are explicit
configurations, not a search for each paper's best tuned result.

For a short controlled Hopper run, submit from the repository root:

```bash
mkdir -p benchmark/results
sbatch --output=benchmark/results/timing-%j.log benchmark/hopper.sbatch benchmark/results/structured benchmark/results/hopper-structured
```

The script reserves `himem02` exclusively and binds one physical core. The worker
checks the EPYC 9754 model, affinity, full-node allocation, and thread settings.
Results record actual worker placement, Slurm job ID, and native thread pools.
These checks verify the execution configuration; the preview still labels times
as diagnostic pending a larger timing qualification study. Other machines use
the default diagnostic profile and are not silently pooled with Hopper results.

## Implementation phases

1. Frozen local pilot, measured budgets, nMSE, JSON/CSV, and local candidate adapter.
2. Static results table and charts, filters, downloads, and private local report.
3. Existing tree games for high-player SV/interactions and KNN data valuation for SV.
4. Resumable bounded campaigns, more compatible methods, and Hopper timing metadata.
5. Coverage-aware summaries, paired comparisons, Elo-style ratings, and release history.

Each phase is audited by an agent who did not implement it, with findings addressed
before the next phase. The audit record and runnable commands are updated as work
lands. No new SCM integration or exact algorithm is required for these phases.

## Read the comparisons correctly

Select a target first: Shapley values and pairwise k-SII never share a ranking.
Then choose a family, player range, and measured budget. An optional **budget cap**
selects the largest measured budget at or below your limit for each game, either
in absolute queries or queries per player. Games with no available budget remain
missing, so filtering cannot silently reward an estimator for easier coverage.

Mean nMSE uses equal family → stratum → game → budget → seed weights. The median
is the lower weighted median under those same weights. A method must have valid
results for every selected cell to receive an aggregate rank. Zero-energy games
are excluded for every method and their count is disclosed. Separate failure,
unsupported, and pending counts show why coverage is incomplete.

The Elo calculation compares identical cells with the same weights. Errors
within `1e-12 + 0.01 × max(error_A, error_B)` tie. Fixed target/family/budget presets
also include an **Elo-style rating**: a batch Bradley–Terry fit, centered at 1000
with a 400-point logistic scale and fixed `0.001` L2 regularization. These ratings
summarize pairwise outcomes, not error magnitude, and depend on the method set.
Custom panels retain nMSE rankings but withhold preset-specific Elo/history.
The website shows the Elo ranking rather than individual win/tie/loss pairs.

Release history evaluates current implementations on the current frozen panel
and places their horizontal score lines at verified first-publication dates.
The step line shows the best eligible mean or median so far. This is a
retrospective comparison, not a reconstruction of what was measured historically.
Unverified dates are listed and omitted from that chart, without removing methods
from the accuracy ranking. Every included date links to its primary source.

The exporter supports paired model-cluster/seed bootstrap intervals when every
stratum has at least two independent model clusters. The current preview has
only one fitted model per stratum, so intervals are unavailable. Adding seeds or
explanation points alone does not create independent model replication.

## Reproduce the public preview exactly

Download the ZIP from the [preview release](https://github.com/rtealwitter/shapiq/releases/tag/benchmark-preview-2026-09-29).
It contains the frozen snapshot, exact truth, baseline results, checksums, and
commands for comparing a local candidate. This avoids refitting a subtly
different game. The archive README records the preparation commit and versions.
Use a local report to view private results; publishing is always separate.

To make a new public reproduction archive from your own measured baseline:

```bash
uv run python -m shapiq_benchmark.bundle --snapshot benchmark/results/structured --results benchmark/results/structured-baselines/results.json --output benchmark/results/structured-bundle.zip
```

The exporter verifies hashes and matching provenance and rejects private methods.
The Pages workflow publishes only four small static files; frozen numerical
artifacts belong in a release archive. No server, account, database, or upload API
is involved.

## What is measured, and what remains

All five implementation phases are working; see [independent audits](AUDITS.md).
The public preview is deliberately small: three real-data game/target settings,
six estimator configurations, two budgets, and two seeds (72 planned cells).
The separate catalog probe covers all 22 public estimator classes, with missing
optional backends and failures visible. This is infrastructure and a reproducible
starting panel, not a claim that every shapiq game and estimator has been qualified.

Before a scientific leaderboard release, expand the frozen suite across datasets,
independent fitted models, budgets, and qualified game families; validate optional
estimator backends; and qualify runtime repeatability. The current structured
adapters also support the 64-feature Digits dataset and larger KNN player sets;
64-feature tree truth and 1,024-player KNN truth were checked, but the public
estimator campaign stops at 128 players. SOUM does not contribute to this preview.

Review the implementation in this order: `prepare.py` / `games.py` freeze truth,
`runner.py` / `execution.py` measure cells, `summary.py` computes fixed statistics,
`report.py` exports the website, and `bundle.py` packages reproduction data.
Existing estimator algorithms were not modified.

## Full library-family sweep

The expanded suite exercises one bounded representative for each shipped game
family, including local/conditional explanations, global fidelity, feature/data/
grouped-data valuation, ensembles, uncertainty, clustering, dependence, tree and
product-kernel games, nearest-neighbor variants, text, images, TabPFN, and causal
attribution. Synthetic payoff and causal examples have a separate **Diagnostics**
panel; they never enter the real-game ranking. Dataset-specific wrappers are not
all separate workloads. The coverage drawer identifies the concrete recipe for
each family, including any preparation failure.

```bash
uv sync --locked --extra benchmark --extra sparse --extra shapleig --extra proxy
uv run --no-sync python -m shapiq_benchmark.prepare --suite benchmark/suites/all-families.json --output benchmark/results/all-families
sbatch --output=benchmark/results/sweep-%j.log benchmark/sweep.sbatch benchmark/results/all-families benchmark/results/all-family-runs
uv run --no-sync python -m shapiq_benchmark.report --results benchmark/results/all-family-runs/shard-*/results.json --public --output benchmark/site
```

This uses all 22 estimator classes, six separate targets (SV, k-SII, SII, STII,
FSII, FBII), two seeds, and `B/d = 2, 8, 64`. Each game's absolute budgets are
`ceil(ratio × players)`, stored in `budgets_by_game`. Unsupported combinations,
insufficient budgets, and timeouts remain visible. The sweep runs disjoint game
panels on fixed cores, with a 120-second worker cap and bounded total job time.
Resubmit unchanged to resume; no completed cell is overwritten. Concurrent
accuracy-sweep timings remain diagnostic and carry their actual worker profile.
`--games` and `--methods` also allow a small local subset with the ordinary runner.

Small families use exact enumerated tables. Stochastic imputers/global games are
explicitly frozen in canonical coalition order: their truth is exact for that
saved realization, not an exact population expectation. Table timings measure
estimator work against cached payoffs; the larger tree/KNN panels call live games.
Original zero-energy cases are retained and excluded consistently from nMSE.

The website defaults to relative budgets. Its two main charts show mean nMSE
across seeds on the selected game, against either `B/d` (or absolute `B`) and
measured time. Choose **All methods** to see methods that succeed only at some
budgets; each plotted point still requires every seed. The table can sort by
mean nMSE or Elo. Hover or keyboard-focus a method to highlight it across plots
and rows; color is reinforced by dash patterns and marker shapes. Publication
history appears below, and detailed methodology stays in expandable sections.
