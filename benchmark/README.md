# Shapiq estimator benchmark

[Open the interactive benchmark](https://www.rtealwitter.com/shapiq/) ·
[Discuss the plan](https://github.com/mmschlk/shapiq/issues/601) ·
[Review the implementation](https://github.com/mmschlk/shapiq/pull/602)

Compare all 22 public estimator classes across representatives of every shipped
game family. Filter by target, game family, player count, or budget; compare median/mean nMSE
and Elo in the table, then explore family-level median nMSE against **queries per
player (`B/d`)** and time. Publication history appears below. Hover or focus a
method to highlight it across charts and rows; click its name for a description,
paper and implementation. Colors also have distinct markers and line patterns.

Python freezes games, runs estimators, and exports results. The website is plain
HTML/CSS/JavaScript with SVG charts: no frontend framework, database, or build
step. You can evaluate a private estimator with the same runner and view the
comparison locally without uploading anything.

## Code map

| Responsibility | Files |
| --- | --- |
| Construct and qualify games | `prepare.py`, `families.py`, `media.py`, `materialize.py`, `games.py` |
| Run, count queries and checkpoint | `runner.py`, `execution.py` |
| Calculate scores, Elo and history | `summary.py` |
| Export the website and reproduction archive | `report.py`, `bundle.py` |
| Render the interface | `benchmark/site/app.js`, `style.css`, `index.html` |
| Describe estimators and link sources | `benchmark/site/methods.js` |

Python files above live in `src/shapiq_benchmark/`. The browser's `render()`
coordinates separate functions for the leaderboard, performance plots, history,
run issues and hardware details. There is no frontend build step.

## What the expanded suite covers

[suites/all-families.json](suites/all-families.json) defines:

| Dimension | Coverage |
| --- | --- |
| Small games | 31 recipes, covering all shipped game families |
| Targets | SV, k-SII, SII, STII, FSII, FBII; interactions at order two |
| Larger games | Forests with 30/64 features (all six targets); product-kernel games with 30/64 features (SV); KNN with 128/256 training-example players (SV) |
| Estimators | All 22 public classes, with library defaults unless target configuration requires otherwise |
| Budgets | `B/d = 0.5, 1, 2, 4, 8, 16, 32, 64, 128`, rounded up to whole queries per game |
| Repetitions | Four constructed game instances per recipe/dataset; one estimator run per instance and budget |
| Prepared panel | 37 setups × four constructions; 808 game/target instances, including all six targets where supported |
| Planned matrix | 159,984 cells including unsupported combinations |

All four-instance recipes qualified. The expanded suite is being measured on
Hopper (job `360655`, frozen source `1472a003`). The website
continues to show the previous published data until its replacement is audited.

The previous three-budget Hopper sweep recorded **9,357 successful, 14,814 unsupported, and
777 failed cells**, with none pending. Failures comprise 754 SPEX minimum-budget
errors, eight OddSHAP minimum-budget errors, and 15 ShaplEIG timeouts. All 34
preparation entries qualified. Successful zero-energy cases remain excluded
from nMSE rankings for every estimator.

The families include local and conditional explanations, global fidelity,
feature/data/grouped-data valuation, ensembles, uncertainty, clustering,
dependence, tree and product-kernel games, nearest-neighbor variants, text,
images, TabPFN, and causal attribution. These are **family representatives**, not
every dataset-specific wrapper or a representative sample of every application.
The coverage drawer identifies each concrete recipe and its preparation status.
Synthetic payoff and causal examples appear separately under **Diagnostics**;
they never enter the real-game ranking.

Small games use exact enumeration of their saved payoff tables. For stochastic
imputers and global games, preparation freezes one canonical-order realization:
truth is exact for that table, not for the population expectation. Large tree, product-kernel and
KNN games use existing structured exact solvers, checked against enumeration on
an eight-player counterpart. KNN players are training examples, not features.

`game_seeds: [0, 1, 2, 3]` changes game construction: data splits, fitted models,
backgrounds and held-out points as applicable. Text and image games use four
distinct inputs to the same pretrained model. `seeds: [0]` runs each estimator
once on each instance at each budget; it does not repeat the estimator four
times on one frozen game. California Housing and Iris keep their native eight
and four features. The larger settings use naturally larger datasets.
Different constructions can still yield identical or zero-energy payoffs on
easy problems; those outcomes are retained, with zero-energy nMSE excluded.

Table-backed timings measure estimator work against cached payoffs; large-game
timings include live coalition calls. Concurrent sweep timings are diagnostic.
Unsupported targets, insufficient budgets, failures, timeouts, and pending cells
remain visible rather than being silently dropped.

## Try a small local comparison

From the repository root:

```bash
uv sync --locked --extra benchmark
uv run python -m shapiq_benchmark.prepare --suite benchmark/suites/pilot.json --output benchmark/results/pilot
uv run python -m shapiq_benchmark.runner --snapshot benchmark/results/pilot --output benchmark/results/baselines
```

This pilot uses one held-out California Housing point, a fitted tree, three
Shapley estimators, three budgets, and three seeds: 27 runs. Exact truth comes
from all 256 coalitions of the eight-feature game. It is a quick integration
check, not evidence of general estimator quality.

To try the larger games separately, substitute
[suites/structured.json](suites/structured.json) and a new output directory.

## Evaluate a private estimator

Copy [example_estimator.py](example_estimator.py) and replace its constructor:

```bash
uv run python -m shapiq_benchmark.runner --snapshot benchmark/results/pilot --output benchmark/results/candidate --candidate benchmark/example_estimator.py:factory
uv run python -m shapiq_benchmark.report --results benchmark/results/baselines/results.json benchmark/results/candidate/results.json --output benchmark/results/local-report
uv run python -m http.server 8000 --bind 127.0.0.1 --directory benchmark/results/local-report
```

Open `http://localhost:8000`. The factory receives `n`, `index`, `order`, and
`seed`; it returns an object implementing `approximate(budget, game)` that returns
`InteractionValues`. The supplied game accepts Boolean coalition matrices and
charges every requested row, including duplicates. Over-budget or invalid
outputs fail the cell. Candidate files are trusted Python code.

Only the candidate runs in the first command; saved baseline accuracy results
can be reused on the identical snapshot. Rerun both methods on the same hardware
for runtime comparisons. The report rejects mismatched snapshots, conflicting
method versions, and duplicate measurements. Nothing uploads automatically.
You can also open the site's `index.html` and choose an exported `data.json`
through **Open local**.

For the full public panel, download its frozen snapshot and baseline results from
the [reproduction release](https://github.com/rtealwitter/shapiq/releases/tag/benchmark-families-2026-09-29).
The archive includes exact truth, checksums, software provenance, and local
candidate commands, so there is no need to refit the games. Use `--games` and
`--methods` to select a smaller experiment when needed.

## Run the expanded suite on Hopper

```bash
uv sync --locked --extra benchmark --extra sparse --extra shapleig --extra proxy
uv run --no-sync python -m shapiq_benchmark.prepare --suite benchmark/suites/all-families.json --output benchmark/results/all-families
sbatch --partition=main --time=04:00:00 --output=benchmark/results/sweep-%j.log benchmark/sweep.sbatch benchmark/results/all-families benchmark/results/all-family-runs --seconds 14040
```

Preparation may download optional model weights and records unavailable families.
Each game's evaluation counts are `ceil(ratio × players)`, saved in
`budgets_by_game`. The adapter passes target, order, and seed to supported
constructor parameters; other settings retain library defaults. For example,
`kADDSHAP` uses order one for SV. These are reproducible configurations, not a
search for each paper's best tuning.

The sweep reserves `himem02` exclusively, divides games among 128 fixed physical
cores, and uses one native thread per worker. All estimators for a game share
the same core. Each cell runs in its own subprocess with a 120-second timeout
and a 12 GiB virtual-memory limit; the job is bounded to 30 minutes.
Resubmit the same command to resume unfinished shards. Keep the snapshot, source,
installed packages, allocation, and limits unchanged during a campaign.

Results record CPU model, worker affinity, Slurm job, and thread settings. This
concurrent accuracy sweep is **not a qualified runtime leaderboard**. For an
isolated timing study, [hopper.sbatch](hopper.sbatch) reserves the node and runs
on one physical core, checking the EPYC 9754 model and full-node allocation:

```bash
sbatch --output=benchmark/results/timing-%j.log benchmark/hopper.sbatch benchmark/results/all-families benchmark/results/isolated-timing
```

The ordinary runner also supports bounded local campaigns:

```bash
uv run --no-sync python -m shapiq_benchmark.runner --snapshot benchmark/results/pilot --output benchmark/results/resumable --max-runs 3
uv run --no-sync python -m shapiq_benchmark.runner --snapshot benchmark/results/pilot --output benchmark/results/resumable --resume
```

`--timeout` includes worker imports and reconstruction; reported `seconds` covers
estimator construction and approximation. `--memory-gb` caps virtual address
space on POSIX. `--max-runs` and `--max-seconds` bound each invocation. Checkpoints
are saved after every cell, including failures. Resume retains completed cells;
use a new output directory after changing code or limits.

## Read the rankings

Select a target first; different interaction definitions never share a ranking.
Choose a budget in multiples of the player count, or use **budget cap** to select
the largest measured budget within your limit for each game. Missing cells remain missing.
The **Plot family** selector controls both query and time charts independently
of the table. Charts show all measured budgets, with the same weighting rule as
the table; each point reports coverage on hover. Both axes use logarithmic
scales, except historical publication years. Errors at or below `1e-12` share a
labeled lower band; hover retains the actual value. The time chart groups runs
with matching CPU and thread profiles; it may combine cached and live oracles.

nMSE is squared coefficient error divided by ground-truth coefficient energy,
excluding the baseline. Zero-energy games have undefined nMSE and are excluded
for every method. Aggregate means give equal weight at each level:
family → stratum → game instance → budget → estimator seed. At an exact half-weight
boundary, the median is the midpoint of the neighboring values, matching the
ordinary median for equal weights. For example, errors 2 and 100 give median 51.
The default view shows compatible estimator families; the **All variants** dropdown option exposes
the underlying classes without combining their scores. The target column marks
value and interaction support. [Estimator notes](ESTIMATOR_NOTES.md) explain
LeverageSHAP’s low-budget instability and the OddSHAP, SPEX, and ShaplEIG findings.

The table defaults to median nMSE, with complete coverage ahead of incomplete
coverage. Click column headers to sort by a different score, name, or coverage.
Methods with any successful scored runs retain their nMSE scores. Their planned
panel weights are renormalized over those runs; missing or failed results are
never treated as zero error. **Coverage** shows successful versus planned cells,
with failure details in its tooltip. Different coverage means different evidence
behind the scores, so use a common game and budget when making close comparisons.

Overall and family presets also provide **Elo rankings** from cells where both estimators
succeeded. Each pair keeps its original panel weights, so less overlap supplies
less evidence. Global ratings require a connected comparison graph. Errors within `1e-12 + 0.01 × max(error_A, error_B)` tie. Ratings
use a batch Bradley–Terry fit centered at 1000, a 400-point logistic scale, and
`0.001` L2 regularization. They describe head-to-head outcomes rather than error
magnitude and depend on the selected methods. Custom panels retain nMSE rankings
but withhold preset-specific Elo and history.

History places each current implementation's score at its verified first-publication
date and shows the best mean or median among methods with complete coverage.
History keeps a common panel so its frontier remains comparable. This is retrospective
performance on today's frozen panel, not a reconstruction of historical results.
Unverified dates are omitted from history without removing methods from accuracy
rankings; included dates link to primary sources.

Bootstrap intervals require at least two independent model clusters in every
stratum. Four independent fitted models can support these intervals, but four inputs to
one pretrained model still form one model cluster. More explanation points alone
do not create independent models. Partial panels do not receive intervals.

## Export and publish

After the sweep, export its shards into the static site and reproduction ZIP:

```bash
uv run --no-sync python -m shapiq_benchmark.report --results benchmark/results/all-family-runs/shard-*/results.json --public --output benchmark/site
uv run --no-sync python -m shapiq_benchmark.bundle --snapshot benchmark/results/all-families --results benchmark/results/all-family-runs/shard-*/results.json --output benchmark/results/all-families-bundle.zip
```

Both public exports reject private candidates. The website omits exact truth,
raw coefficient arrays, and local paths; the ZIP includes the frozen numerical
artifacts needed for reproduction, with hashes and provenance verified.

Generated `data.json` is a release asset, excluded from Git.
[site/data-source.json](site/data-source.json) pins its URL and SHA-256; Pages
downloads it and verifies its checksum and public-export checks before deploying.
Updating that small manifest publishes a new audited dataset. Local reports
still write their own `data.json` directly. To preview the current public data
from a fresh checkout, download the pinned asset into `benchmark/site/data.json`
or open it with **Open local**.

GitHub Pages deploys the six static site assets when `BENCHMARK_PAGES=true` and
Pages uses GitHub Actions. It does not run experiments or upload local results.
Large numerical artifacts belong in the separate release archive. Hosting needs
no server, account system, database, or upload API.

## Implementation and next steps

The five initial phases are implemented and independently reviewed:

1. Frozen local pilot, counted budgets, nMSE, and private candidate adapter.
2. Static table, charts, filters, downloads, and local reports.
3. Structured exact truth for larger tree and KNN games.
4. Bounded, resumable campaigns and Hopper placement metadata.
5. Coverage-aware summaries, Elo, and publication history.

The expanded family sweep and redesigned website build on these phases. See
[AUDITS.md](AUDITS.md) for independent reviews and [DESIGN.md](DESIGN.md) for the
scientific protocol. The next research steps are independent fitted models,
more datasets and budgets, estimator configuration studies, and runtime
repeatability qualification. Broad family coverage alone is not enough for
claims of universal estimator quality.

For code review, start with `prepare.py`, `families.py`, `media.py`, and
`materialize.py` for frozen tables; `games.py` for structured games;
`runner.py`/`execution.py` for measured cells; `summary.py` for statistics; and
`report.py`/`bundle.py` for exports. Existing estimator algorithms are unchanged.
