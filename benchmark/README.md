# Shapiq estimator benchmark

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

## Implementation phases

1. Frozen local pilot, measured budgets, nMSE, JSON/CSV, and local candidate adapter.
2. Static results table and charts, filters, downloads, and private local report.
3. Existing tree games for high-player SV/interactions and KNN data valuation for SV.
4. Resumable bounded campaigns, more compatible methods, and Hopper timing metadata.
5. Coverage-aware summaries, paired comparisons, Elo-style ratings, and release history.

Each phase is audited by an agent who did not implement it, with findings addressed
before the next phase. The audit record and runnable commands are updated as work
lands. No new SCM integration or exact algorithm is required for these phases.
