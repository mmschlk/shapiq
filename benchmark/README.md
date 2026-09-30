# Shapiq estimator benchmark

[Open the interactive benchmark](https://www.rtealwitter.com/shapiq/) ·
[Discuss the plan](https://github.com/mmschlk/shapiq/issues/601) ·
[Review the implementation](https://github.com/mmschlk/shapiq/pull/602)

Compare all 22 public estimator classes across representatives of every shipped
game family. Filter by target, game family, player count, or budget; compare median/mean nMSE
and Elo in the table, then explore family-level median nMSE against **queries per
player (`B/d`)** and time. Publication history appears below. Hover or focus a
method to highlight it across charts and rows; click its name to expand a
description, paper and implementation inline. Colors also have distinct markers
and line patterns.
History labels sit in endpoint-error order on the right, spaced for readability.
On narrow screens, an ordered list below the plot shows their nMSE.

Python freezes games, runs estimators, and exports results. The website is plain
HTML/CSS/JavaScript with SVG charts: no frontend framework, database, or build
step. You can evaluate a private estimator with the same runner and view the
comparison locally without uploading anything.

## Code map

| Responsibility | Files |
| --- | --- |
| Select compatible dataset/recipe/player combinations | `matrix.py`, `benchmark/suites/matrix.json` |
| Prepare a matrix in parallel and resume checkpoints | `benchmark/prepare_matrix.py` |
| Construct game recipes | `families.py`, `media.py`, `games.py` |
| Freeze and qualify game snapshots | `prepare.py`, `materialize.py` |
| Compute exact values and interactions from a table | `exact.py` |
| Authenticate payoff checkpoints and record evaluation costs | `payoff_cache.py` |
| Run, count queries and checkpoint | `runner.py`, `execution.py` |
| Calculate scores, Elo and history | `summary.py` |
| Export the website and reproduction archive | `report.py`, `bundle.py` |
| Render the interface | `benchmark/site/app.js`, `style.css`, `index.html` |
| Draw performance and history charts | `benchmark/site/charts.js` |
| Describe estimators and link sources | `benchmark/site/methods.js` |

Python files above live in `src/shapiq_benchmark/`. Start with `prepare.py` for
snapshot construction and `runner.py` for estimator execution. The parallel
preparation script separates task planning, worker execution and final assembly.
In the browser, `app.js` manages data, filters and tables; `charts.js` draws the
charts. Both are ordinary scripts with no frontend build step.

## What the expanded suite covers

[suites/all-families.json](suites/all-families.json) defines:

| Dimension | Coverage |
| --- | --- |
| Enumerated games | 31 game kinds, with 58 settings at 11/12 players |
| Targets | SV, k-SII, SII, STII, FSII, FBII; interactions at order two |
| Larger games | Forests with 30/64 features (all six targets); product-kernel games with 30/64 features (SV); KNN with 16/32/64/128/256/512 training-example players (SV) |
| Estimators | All 22 public classes, with library defaults unless target configuration requires otherwise |
| Budgets | `B/d = 0.5, 1, 2, 4, 8, 16, 32, 64, 128`, rounded up to whole queries per game |
| Repetitions | Four constructed game instances per recipe/dataset; one estimator run per instance and budget |
| Minimum players | Every game must have at least 11 players; all nine budgets remain below the full coalition count |
| Planned panel | 71 settings × four constructions; 1,484 game/target definitions if all qualify |
| Planned matrix | 293,832 cells including unsupported combinations |

The replacement suite requires **at least 11 players in every game**, rather
than merely filtering small games from the display. Regression-feature games
use actual Bike Sharing features; Gaussian imputers, uncertainty and TabPFN use
Wine classification games. Valuation and ensemble games use Diabetes or Bike
Sharing with 11/12 row, group or model players. Neighbor games use 11/12 Wine
training rows, plus larger exact KNN games on Breast Cancer and Digits. Text and
image games use longer inputs and finer nonempty segmentations. Four constructed
instances remain required for each setting. Real-game dimensions come from those
inputs or model components; synthetic diagnostics can deliberately include dummy
players, and not every declared feature must affect a fitted model's output.

At 11 players, the maximum budget is `128 × 11 = 1,408 < 2,048` coalitions.
This prevents full enumeration within the budget grid; easy games can still have
near-zero estimation error. Preparation rejects any constructed game below the
minimum. New results are not yet on the website.

The earlier 74-setting preparation finished in job `360843`; its queued sweep
`360844` was cancelled before execution to replace it with this minimum-player
suite. The completed frozen preparation remains archived separately. Publication
requires an audited export and a data-manifest update; a completion watcher will
wake the agent to carry out that work for the replacement campaign.

The selected-pairing preparation uses frozen source `218d1390` with the
LeverageSHAP and OddSHAP fixes. Job **360853** saved every payoff table, but two
exact-reference calculations hit the worker's memory limit. Recovery **361008**
qualified those saved tables in fresh processes, preserving all original bytes;
the complete snapshot has **1,484 game/target definitions and 324 artifacts**.
Replacement sweep **361009** is queued for the exclusive benchmark node, and its
completion watcher is armed. The cancelled sweep **360873** never evaluated cells.

### Broader dataset × game matrix

[suites/matrix.json](suites/matrix.json) crosses 23 tabular recipes with seven
datasets: California Housing, Diabetes, Bike Sharing, Iris, Wine, Breast Cancer
and Digits. Players can mean features, training rows, groups or models depending
on the recipe. Each selected setting gets **four independently constructed games**
and one estimator evaluation per game and budget.

The filter checks the task type, available features and exact-reference support.
Generic recipes use exhaustive coalition tables through **20 players**. The grid
includes 11, 12, 16 and 20, plus compatible native widths such as Wine’s 13.
Larger candidates are excluded unless an existing, qualified structured adapter
provides exact truth for that particular game and explanation target. This is an
implementation limit, not a claim that larger exact games are impossible. The
13 retained structured settings are distinct tree, KNN and product-kernel games;
they do not silently replace excluded recipes. Feature counts are never padded.
Digits clustering uses pixels that vary on its training rows so singleton
coalitions have defined scores; the original feature IDs and selection rule are
recorded.

The expansion selects **356 of 1,524 tabular candidates**, documenting a reason
for each of the 1,168 exclusions. With 12 non-tabular and 13 structured settings,
that is **381 settings, 8,924 game/target definitions and 1,766,952 planned cells**.
Unsupported estimator targets remain visible as coverage, separate from failures.
Selection is a plan: preparation must still qualify every constructed game.

```bash
uv run python -m shapiq_benchmark.matrix \
  --config benchmark/suites/matrix.json \
  --output benchmark/results/matrix20-campaign/suite.json
```

The generated suite records selected combinations and exclusion reasons in
`matrix_coverage`. The frozen snapshot, public export and reproduction archive
retain this inventory. Generated suites and results stay outside Git. The broader
campaign follows the selected-pairing run; each has its own completion watcher,
audit and publication. An older campaign must never overwrite a newer live dataset.
Preparation uses 64 pinned single-thread workers on an exclusive Hopper node,
with 384 GiB allocated and a 12-GiB address-space limit per worker. Jobs have a
72-hour window; completed chunks and tables survive resubmission. Evaluation
retains the same 120-second/12-GiB per-cell limits and a 24-hour job window.

The earlier pending matrix jobs 360871/360872 were cancelled before execution
when the enumeration cap increased. Replacement preparation **360900** follows
the selected-pairing sweep; evaluation **360901** starts after successful
preparation. Both use frozen source
[`0762a2e5`](https://github.com/rtealwitter/shapiq/commit/0762a2e503ad0d5852308b06427feabedd2a534d),
including both estimator fixes. Separate watchers are armed for both campaigns
to wake this session for audit, resumption if necessary, and verified publication.
New results are still pending.

### Evaluate once, reuse the table

A 20-player game has 1,048,576 coalitions; its float64 payoff array is **8 MiB**.
Preparation saves every payoff in canonical bitmask order, along with the recipe,
construction seed, data identities, selected rows/features and source provenance.
All estimators use this same table, with the same query limits. No model is
retrained just to repeat an estimator evaluation.

Above 12 players, preparation checkpoints blocks of 4,096 coalitions. Each block
reconstructs the same seeded recipe and resets its random streams. For sampled
games, this defines a documented frozen realization; shared random draws across
blocks are not independent Monte Carlo samples. Smaller games keep their existing
full-batch protocol. Checksums and recipe/source identities prevent mixing caches.

Exact SV and order-two interactions are combined from the saved table using
vectorized first/second differences, without another oracle call or a huge
regression matrix. Independent checks agree with the existing definitions and
preserve both exactly zero and genuinely tiny attributions. The six targets take
about four seconds to combine at 20 players on the qualification machine; game
evaluation can take much longer. Hopper TabPFN probes suggest roughly eight
CPU-days per 20-player table, motivating parallel checkpoints. These are cost
estimates, not guaranteed completion times.

The new matrix excludes an enumerated game/target from rankings when
`RMS(truth) / std(all coalition payoffs) < min_signal_ratio` (currently `1e-6`).
This scale-independent check applies to the whole coefficient vector, not each
player. Exclusions are uniform across methods and retain raw truth/results for
inspection. Larger structured games retain the zero-energy check; their full
payoff variation is not enumerated. The policy is part of the frozen suite.

### Cached runtime and estimated evaluation cost

The **measured runtime** includes cheap table lookups. New tables also record
original evaluation time, divided equally among coalitions in each preparation
batch. Each accepted estimator query—including repeated queries—is charged that
saved cost without sleeping. **Estimated uncached time** is measured runtime,
minus actual lookup time, plus those recorded costs. It excludes recipe/model
construction, just as the estimator timer does.

Both timings remain available. Estimates are diagnostic: batching, warm caches
and concurrent preparation affect cost, so this is not a fresh end-to-end timing
measurement. The plot labels the estimate and uses only games with recorded
costs and matching timing profiles. Older cached tables have no inferred costs;
structured live-oracle runs keep their measured timings.

The **published dataset** still contains 37 settings × four constructions,
808 game/target definitions and 159,984 cells. All its recipes qualified.
The full rerun completed on Hopper
(job `360683`) on September 29 at 21:48 PDT, after 3 hours 42 minutes. It recorded
**58,517 successful, 94,284 unsupported and 7,183 failed cells**, with none pending.
Failures include insufficient-budget errors and 303 worker timeouts; the website
keeps each method's actual coverage visible. All nine relative budgets and all
four game constructions are included. Zero-energy games remain excluded from
nMSE rankings for every estimator.

Execution is frozen at
[`d4ac18e6`](https://github.com/rtealwitter/shapiq/commit/d4ac18e674841f79c1ca25d8cfbf550e84dc21a7)
on `benchmark-ridge-run`, including corrected LeverageSHAP from PR #603. The frozen
games retain preparation source `1472a003` and their original snapshot identity.
The superseded run and provisional release remain separate; no earlier scores
are reused. Published datasets are static snapshots, updated after export and
independent audit, rather than live feeds from the compute cluster.

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
The coverage drawer describes each game kind and links its implementation and
data sources. Known unsupported explanation types are compatibility metadata;
under-budget runs are listed separately from timeouts and other failed attempts.
Both remain unscored and reduce coverage; reporting does not change estimator behavior.
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
times on one frozen game. Legacy recipes retain their original defaults for
reproduction; the new public suite explicitly selects larger datasets or
recorded feature subsets selected by seed without using labels. Each dataset
and player-count setting has its own stratum, so adding instances to a family
does not increase that family's overall weight.
New neighbor settings select training rows by stratified sampling. New threshold
neighbor settings use the median positive pairwise distance of standardized
training inputs as their radius; held-out points and benchmark scores do not
choose it. Legacy settings retain their original row selection and radius.
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
the [reproduction release](https://github.com/rtealwitter/shapiq/releases/tag/benchmark-budget-report-2026-09-30).
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
and a 12 GiB virtual-memory limit. The template defaults to 30 minutes; the
command above requests four hours with a shorter runner deadline for checkpoints.
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
The default SV view groups KernelSHAPIQ under KernelSHAP and SVARMIQ under SVARM:
each pair uses the same estimator configuration at order one. The **All variants**
dropdown restores their individual records; interaction views retain their own
methods. Scores are never pooled, and the representative keeps its own publication
date. ProxySHAP and RegressionMSR remain separate because their default residual
adjustment and sampling differ. Hiding variants does not change the full-panel Elo
fit. The target column marks
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

History shows one horizontal mean or median nMSE line per method with complete
coverage, beginning at its verified first-publication date. Each method uses the
same frozen panel. Names and publication years align with line endpoints on the
right; narrow screens show an ordered list with endpoint scores below the plot.
This is retrospective performance on today's frozen panel, not a reconstruction
of historical results.
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
