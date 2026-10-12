# Compact benchmark operations

The active design is [scope version 2](COMPREHENSIVE.md): 512 recipes, four
construction seeds and 2,048 intended instances. Start with the portable commands
in [README.md](README.md). Superseded campaigns and their counts are historical;
this guide does not authorize restarting them.

## Before running work

Read the campaign's current `WAKEUP.md`, state, accounting and immutable input
manifest. Use CPU-only nodes, no GPUs or GPU-node CPU allocations, and keep all
concurrent work within the 1,000-CPU ceiling. Preparation, evaluation, recovery
and validation share that capacity. Record actual allocation CPU-time even though
the total CPU-hour cap was waived.

Authenticate the recipe subset, frozen source, environment, model settings,
construction seeds, estimator seeds and selected budgets before reuse. Reuse a
historical game or result only when those inputs match. A new scope label does
not change an old attempt's source or identity.

Use a separate development environment. While any job uses a shared environment,
checks must set `UV_NO_SYNC=1`; do not install or rebuild into that environment.
Dataset caches and byte-identical node-local dependencies must be ready before
parallel workers start.

## Prepare and evaluate

Generate the compact suite:

```bash
UV_NO_SYNC=1 uv run python -m shapiq_benchmark.comprehensive benchmark/suites/comprehensive.csv /shared/campaign/suite.json
```

The README's small pilot demonstrates preparation, a counted-query run and a
local report. For production, freeze qualified snapshots before evaluation;
retain exact reference vectors, artifacts, selected feature/row IDs and model
metadata. Generic games enumerate at most 12 players. Native RF tree games use
the matching exact solver, with recorded qualification limits.

Execution settings are explicit in [settings.json](settings.json). Startup,
estimation and output have separate watchdogs. Normal workers use an 8-GiB
address-space cap; any override is recorded with the actual attempt. Historical
larger-memory and source-patched recoveries are not compact-plan defaults.

Use atomic shared claims and durable intents before launching an estimator.
Resume only never-attempted cells or explicitly authorized replacement attempts;
preserve successes. A scheduler exit does not establish complete cell coverage.
Read compact or legacy results through `shapiq_benchmark.results_io.read_results`,
which understands the snapshot, manifest and journal companions.

## Monitor and recover

Inspect exact allocation IDs, terminal accounting, checkpoints and launcher logs.
An absent queue entry is not proof of success. Shared-filesystem checkpoint reads
can transiently return `ESTALE`; retry the read rather than classifying it as a
worker failure. Never scan a live partial journal as a completed export input.

Acknowledge an existing campaign watcher delivery with:

```bash
UV_NO_SYNC=1 uv run python benchmark/watch_campaign.py /shared/campaign --acknowledge
```

Do not start duplicate watchers or submission coordinators. Monitoring is event
based; a user pause or cancellation takes precedence over continuation. Keep
unattempted cells distinct from durable unanswered intents, genuine estimator
failures, unsupported targets and missing references.

The completion goal is 100% successful supported coverage except ShaplEIG and
SPEX. Diagnose other failures without changing the scientific game, parameters
or scores. Authenticate any repair and preserve its original attempt evidence.
Timing on shared CPUs is diagnostic; cached-oracle cost estimates are separate
from measured estimator runtime.

## Validate and publish

Run one consolidated reference, ownership, numerical and coverage check over the
selected compact cohort. Retain one current outcome per cell; never mix unmatched
historical recipes into the report. Check zero-signal eligibility, within-recipe
duplicate ownership and weighting before computing ranks.

[Partitioned reports](PARTITIONED_REPORTS.md) describe bounded report delivery;
[publication](MATRIX_PUBLICATION.md) describes release and deployment. Campaign
export entry points require their actual authenticated inputs and terminal
outputs. Portable `shapiq_benchmark.report` is a local comparison, not a substitute
for those final campaign checks.

Keep the current public report until the replacement passes. Publish report data,
About content and reproduction artifacts together. Verify deployed hashes,
coverage, filters and representative scores against the audited report. Public
reproduction must retain original run mappings and qualified source patches while
removing private paths and raw exception text.

Canonical About content is Markdown; render and verify it with:

```bash
UV_NO_SYNC=1 uv run python benchmark/render_about.py
UV_NO_SYNC=1 uv run python benchmark/render_about.py --check
```

`site/assets.json` is the explicit deployed asset list. Editing `site/` can trigger
Pages deployment, so hold those changes until the matching report is ready.

## Cleanup and handoff

Remove obsolete launchers, plans and rebuildable staging material after checking
imports, registered worktrees and retained-result dependencies. Materialize and
verify any required provenance before deleting its old path. Some comprehensive
staging scripts still reference `production/local-environment` receipts; directory
age alone does not make those files disposable.

Keep one active plan and a concise [handoff checklist](HANDOFF.md). Git history
retains superseded repository narratives. [DESIGN.md](DESIGN.md) and
[AUDITS.md](AUDITS.md) contain historical technical rationale and validation,
not current scientific scope or instructions to resume old jobs.
