# Shapiq estimator benchmark

[Open the live research preview](https://www.rtealwitter.com/shapiq/).

An offline benchmark and static comparison website. Python prepares frozen games,
runs estimators, and writes results; the browser only reads those results. A local
paper implementation uses the same runner without uploading anything.

Implementation is proceeding in five independently reviewed phases on this branch.
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
directly and select the exported `data.json` through **Open local report**.

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
