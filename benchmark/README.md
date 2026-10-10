# Shapley estimator benchmark

[Results](https://www.rtealwitter.com/shapiq/) ·
[About](site/about.md) · [Scientific plan](FOCUSED.md) ·
[Recipe manifest](suites/focused.csv)

One fixed benchmark compares 22 estimator configurations on three applications:
individual predictions, data valuation and feature selection. The manifest has
112 recipes across twelve datasets and four construction seeds: 448 intended
instances, or at most 424 after the existing model-quality exclusions. Every
accuracy comparison uses an exact reference for its frozen game.

The website continues to serve its previously audited results until this cohort
passes the final audit. Intended counts do not establish measured coverage.

## Implementation

Keep benchmark orchestration in `shapiq_benchmark` and `benchmark/`; estimator
changes belong in their own PRs. The pipeline reuses the same game adapters,
qualification workers, snapshot format, counted-query runner and report exporter.
The main CSV is the single recipe source. See [FOCUSED.md](FOCUSED.md) for sizes,
models, exact references, weighting and resource limits.

Generate a new suite from a checkout containing the approved estimator options:

```bash
UV_NO_SYNC=1 uv run python -m shapiq_benchmark.focused benchmark/suites/focused.csv /shared/suite.json
```

This writes a manifest only. The active campaign reconciles it against already
prepared recipe identities, reuses authenticated identical games and evaluates
only remaining cells. Its `WAKEUP.md`, `state.json` and `budget.json` on lab
storage are authoritative for jobs, accounting and next actions.

Never install into the environment used by active jobs. Use the pinned interpreter,
`UV_NO_SYNC=1` and explicit frozen `PYTHONPATH`. Preserve admitted source, dataset
caches, prepared truth, journal ownership and review hashes. Budget all allocations
under 2,048 CPU-hours and 128 concurrent CPUs; currently qualified recovery has a
64-worker ceiling. No production GPUs are used.

## Small local example

In your own development environment:

```bash
uv sync --locked --extra benchmark
uv run python -m shapiq_benchmark.prepare --suite benchmark/suites/pilot.json --output benchmark/results/pilot
uv run python -m shapiq_benchmark.runner --snapshot benchmark/results/pilot --output benchmark/results/baselines
```

The example has 27 runs on an eight-player tree game. It checks integration,
not general estimator quality. [Structured examples](suites/structured.json)
exercise exact native references.

## Private estimators

Copy [example_estimator.py](example_estimator.py), implement its factory, then:

```bash
uv run python -m shapiq_benchmark.runner --snapshot benchmark/results/pilot --output benchmark/results/candidate --candidate benchmark/example_estimator.py:factory
uv run python -m shapiq_benchmark.report --results benchmark/results/baselines/results.json benchmark/results/candidate/results.json --output benchmark/results/local-report
uv run python -m http.server 8000 --bind 127.0.0.1 --directory benchmark/results/local-report
```

The factory receives `n`, `index`, `order` and `seed`. Its estimator returns
`InteractionValues` from `approximate(budget, game)`. Every requested coalition
row counts, including duplicates. Candidate code stays local and never accesses
the production duplicate registry. Reuse baseline accuracy only for identical
snapshots; shared-node timings are diagnostic.

## About document and publication

Edit [site/about.md](site/about.md), then render its static HTML:

```bash
UV_NO_SYNC=1 uv run python benchmark/render_about.py
UV_NO_SYNC=1 uv run python benchmark/render_about.py --check
```

The committed page includes a table of contents and needs no browser Markdown
library. [site/assets.json](site/assets.json) lists every deployed asset and is
shared by local exports and GitHub Pages.

The campaign's `production/RECOVERY-EXPORT.md` describes current ownership and
numerical audits. Final publication requires independent coverage, score,
weighting, privacy, asset and browser checks. Keep the existing site until the
replacement passes. Never combine the focused cohort with historical campaigns.

The watcher checks hourly and issues routine progress reminders every two hours.
Acknowledge deliveries with the pinned interpreter:

```bash
python benchmark/watch_campaign.py /shared/coordinator --acknowledge
```

Stop it only after verified live completion or an explicit user request.

## Reusable references

- [Partitioned reports](PARTITIONED_REPORTS.md): disk-backed export and browser data.
- [Publication validation](MATRIX_PUBLICATION.md): authenticated export contracts.
- [Audit checklist](AUDITS.md): score, provenance and coverage checks.
- [Operational reference](OPERATIONS.md): recovery and historical reproduction.
- [Estimator notes](ESTIMATOR_NOTES.md): recorded numerical findings.

Historical published artifacts and frozen execution sources remain authenticated
reproduction evidence. They are not additional active benchmark plans.
