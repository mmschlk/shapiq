# Shapiq estimator benchmark

[Open the interactive benchmark](https://www.rtealwitter.com/shapiq/) ·
[How the benchmark works](https://www.rtealwitter.com/shapiq/about.html) ·
[Discuss the plan](https://github.com/mmschlk/shapiq/issues/601) ·
[Review the implementation](https://github.com/mmschlk/shapiq/pull/602)

Compare all 22 public estimator classes on a declared cohort of frozen games.
The current release covers 248 game instances across several explanation and
valuation families; the [roadmap](ROADMAP.md) expands coverage further. Filter by target, family,
model, dataset, player count, or budget; compare median/mean nMSE
and Elo in the table, then explore family-level median nMSE against **queries per
player (`B/d`)** and time. Publication history appears below. Hover or focus a
method to highlight it across charts and rows; click its name to expand a
description, paper and implementation inline. Colors also have distinct markers
and line patterns.
History labels sit in endpoint-error order on the right, spaced for readability.
On narrow screens, an ordered list below the plot shows their nMSE.

The separate [About page](site/about.html) explains the benchmark from game
construction through exact ground truth and scoring. It includes a coalition-count
explorer and the loaded report’s game/dataset/model tables, with frozen-source links.
The main page keeps the rankings, charts and result-specific coverage details.

Python freezes games, runs estimators, and exports results. The website is plain
HTML/CSS/JavaScript with SVG charts: no frontend framework, database, or build
step. You can evaluate a private estimator with the same runner and view the
comparison locally without uploading anything.

## Continuing through the phases

The bounded seven-phase rollout is published. The larger matrix is now running
from `/hopper/groups/witterlab/rwitter/shapiq-benchmark-matrix-expansion`.
Its 38 waves are already queued: prepare → evaluate → audit → next wave.
The ceiling is **128 shared CPU cores**, with no production GPUs reserved.
Read the coordinator's `WAKEUP.md` and the matrix `state.json` before acting;
audit-job recovery journals override the corresponding original job IDs.
New matrix results require independent review before publication. The exporter
and browser still need further scaling work for this larger dataset.

The following records the earlier bounded rollout and its reproduction commands.

The corrected bounded rollout keeps its plan and progress in lab storage at
`/hopper/groups/witterlab/rwitter/shapiq-benchmark-quality-rollout`:
`campaign/campaign.json` is the immutable scientific plan and `state.json` records
current progress. Job recovery overrides are explicit; for phases 4–7 use
`operational/future-runtime-jobs.json` and its resource-replacement journal
`operational/shared-cpu-v1/jobs.json`, not the superseded IDs in
`campaign/jobs.json`. The [roadmap status](ROADMAP.md) distinguishes this rollout
from the original phase-three campaign, whose twenty batches are complete and
remain separate from the corrected-source results.

The existing coordinator remains at `benchmark/results/full-rollout/`, a symlink
to `/hopper/groups/witterlab/rwitter/shapiq-benchmark-full-rollout`.
Its `WAKEUP.md` gives the next action and `watch.json` records current-stage jobs.
Large payoff caches belong in lab storage. Moving a campaign requires stopping
its writers, verifying copied hashes and preserving recorded paths.
The watcher runs every five minutes and queues a message to the authorized
session when a job finishes or fails. During
implementation, or after the jobs stop, a thirty-minute idle heartbeat also wakes
the session. The next action is always explicit: **qualify → run → independently
audit → publish → verify live → start the next phase**.

`benchmark/queue_phases.py` creates disjoint batches from the cumulative phase
manifests, retaining all nine budgets and four game seeds. Its job journal makes
resubmission reviewable. Each evaluation array task depends on its corresponding
preparation task. CPU batches use pinned, single-threaded workers on EPYC 9754
CPU-only nodes. The corrected rollout requests 16 CPUs for preparation and 32
for evaluation, without reserving the whole node. Timings remain diagnostic;
shared-node contention can affect wall time. The explicit `phase_batch.py
--workers N` override records this policy separately from frozen scientific inputs.
Explicit CUDA batches use one L40S per preparation worker. Later phases are queued on hold, and the resumed agent releases them
after the preceding audit. A
completed scheduler job is never enough to declare a phase complete.

GPU preparation is wrapped by `benchmark/gpu_utilization.py`: after two minutes
of startup, each allocated GPU must sustain an average utilization of at least
80% over a five-minute rolling window. The guard stops the job on sustained
underuse or missing telemetry and saves a receipt for the continuation audit.
Short runs are marked insufficient observation. This measures device occupancy,
not speedup; a CPU fallback needs a separately qualified backend and cache identity.
Run the guard inside the Slurm step so job cleanup also stops descendant workers.

The original `queue_phases.py` submission defaults below retain their legacy
whole-node policy. For this rollout, use the audited shared-CPU operational
wrappers and journal above; do not resubmit it with those legacy defaults.

From a **clean frozen checkout containing the approved estimator corrections**:

```bash
export PYTHONPATH="$PWD/src"
UV_NO_SYNC=1 uv run python benchmark/queue_phases.py /shared/campaign
# Review the generated inventories and batches, then queue them:
UV_NO_SYNC=1 uv run python benchmark/queue_phases.py /shared/campaign --submit
```

For the corrected expansion, pass **`--bounded-core`** to both commands. This
versions recipes as `quality-v2` and selects at most sixteen new recipes per
phase, round-robin across constructions, before expanding the full matrix.
`phase-N-core.json` records every selected and deferred recipe and aggregate
payoff, estimator-cell, reference-size, CPU-hour and storage bounds. These are
conservative planning bounds, not promised runtimes. The complete inventory stays
available; deferred recipes are not failed runs. Per-game qualification then
checks all four seeds, validation against a dummy model, sampled-imputation
stability, and actual requested-size structured-solver cost.

The quality protocol stratifies classification row players and scales SVR training
targets while retaining original payoff units. Frozen tables identify inactive
players, constant nonempty utility and dominant empty-coalition jumps. These
remain inspectable **control games**, excluded from the default representative
panel. Monte Carlo stability is a separate diagnostic from the Fourier spectrum.
No game is selected using estimator performance.

The corrected core explicitly requests `OddSHAP(ridge=0.001)` through the suite's
`method_parameters`; its library default remains unregularized. The penalty acts
only at budgets at most `3d` before full enumeration. A paired low-budget pilot
found small average improvements, not a fix for omitted singleton terms. The
frozen checkout must contain PR609 as well as the earlier approved corrections.
Constructor overrides are saved with method provenance, so differently configured
results cannot be silently merged under one method name.

Exact duplicate payoff tables are registered once across batches before estimator
evaluation. Public reports remove duplicate games and record their canonical
aliases; historical snapshots remain reproducible. Checkpoints now share one
immutable snapshot and append individual cells to `records.jsonl`, avoiding a
full metadata rewrite for every evaluation. Use `results_io.read_results()` to
read either this format or older self-contained results.

Interaction reports retain whole-vector nMSE and add singleton/pair-only scores
with a near-zero signal guard. Confidence intervals resample shared dataset
instances together across constructions. Common-panel Elo is a sensitivity check
alongside the full available comparisons. Cached oracle charges remain estimates;
measured runtime is labelled separately.

Every wake-up is acknowledged with
`python benchmark/watch_campaign.py /shared/campaign --acknowledge`.
The acknowledgement also refreshes the implementation heartbeat. Update watched
job IDs when advancing; do not retire the watcher after one successful phase.
Set its status to `complete` only after the final audited live release, or to
`paused`/`cancelled` at the user's request. The watcher submits no experiments and
publishes nothing itself: it wakes the agent to inspect evidence and continue.

After every batch in a phase finishes, export the cumulative comparison:

```bash
UV_NO_SYNC=1 uv run python benchmark/export_phase.py /shared/campaign/campaign /shared/release/site --through-phase 3
```

The exporter checks frozen inputs, estimator revisions and complete cell panels,
then computes statistics across the combined games. It refuses incomplete or
actively written batches. Independent scientific review still precedes publication.
Keep each batch's reproduction archive and the exported composition manifest so
the combined report can be traced back to its original snapshots.

For repeated exports, add `--cache-dir /shared/publication-cache`. This private
cache reuses each batch's normalized scores and game diagnostics. Every export
still validates the original shards and their provenance; a hit avoids the
second parse and normalization pass. Changed inputs or normalization code get a
new cache entry, and damaged entries are rejected. Keep this cache out of the
website and reproduction archive. It grants no publication approval and does
not remove the exporter's whole-panel memory requirement.

For a separate retry of a transient dataset-download failure, add
`--supplement /shared/recovery/campaign`. The exporter authenticates both runs,
requires unchanged recipes and settings, and resolves duplicates across them.
It preserves the original failure records and rejects retries of scientific
exclusions or unrelated games.

### Replacing a corrected estimator

Keep the frozen games and original results. Run only the affected built-in methods
from a clean, separately recorded revision, using `runner --methods` and new
output directories. Preparation and estimator execution retain their own provenance.

Add `--replacements /shared/replacements.json` to the cumulative export command.
The manifest declares the corrected source, method names, original snapshot IDs
and replacement result paths. The exporter requires every cell for those methods
across every included batch, including supplements. It replaces their complete
panels, preserves other methods, and rejects changed games, settings, unsupported
or duplicate decisions. A later phase also needs its complete replacement panel.
Regenerate reproduction archives with the same corrected baseline measurements.

The completed repair for [PR #610](https://github.com/mmschlk/shapiq/pull/610) replaces
KernelSHAP, KernelSHAPIQ, InconsistentKernelSHAPIQ, RegressionFSII, RegressionFBII
and kADDSHAP at all nine budgets. The phase-seven release includes the complete
corrected panels through phase seven; games and exact answers are reused.
LeverageSHAP already uses SVD.
On Hopper, `operational/regression-svd-rerun/` contains the frozen execution plan,
job records, completed audits and `replacements-through-phase7.json`. Results,
website data and reproduction files passed independent audits; original raw
measurements stay archived. The separately authorized full matrix expansion is
tracked independently and does not change these frozen measurements.

### Recovering a stopped GPU preparation on CPU

Use `--backend-supersession /shared/backend-supersession.json` when the utilization
guard stopped a GPU preparation and a separate CPU campaign replaces it. The
manifest pins the original campaign and suite, guard report, terminal failure and
dependent-job cancellation receipts, and the new campaign and source. Its recipe
mapping may change only IDs and `device: cuda` to `device: cpu`; budgets, seeds,
models, datasets and qualification gates stay fixed. Saved scheduler receipts
make the export portable; Slurm is not needed to read them.

The original batch remains recorded as superseded, never as a fabricated empty
qualification. Every CPU recipe must finish qualification and every retained
evaluation must be complete. With `--replacements`, corrected panels cover the
original components; the new CPU campaign already runs the corrected source.
Records keep their actual run provenance. Methods executed under multiple package
revisions have explicit `source_versions` metadata, without relabeling older runs.

Large reports split lossless evaluation records into one file per explanation
target. The browser fetches only the selected target; filters and scores retain
full precision. Upload `data.json` **and every `records-*.json` companion** to the
same release before updating `site/data-source.json`. Pages verifies their hashes.
For a local comparison, select the manifest and its companions together; legacy
single-file reports still work. Static worker profiles are shared; per-run memory
and CPU counters stay in scalar columns and are restored in JSON downloads.
Preset objects use the lossless `columns-v2` dictionary encoding; records retain
`columns-v1`. The browser accepts older preset arrays too. Reproduction archives
keep their original full records.

## Code map

| Responsibility | Files |
| --- | --- |
| Declare rollout, budgets, datasets and model/construction pairings | `protocol.py` |
| Fit and reuse stronger models with held-out quality checks | `models.py` |
| Describe frozen game complexity | `spectrum.py` |
| Select compatible dataset/recipe/player combinations | `matrix.py`, `benchmark/suites/matrix.json` |
| Prepare a matrix in parallel and resume checkpoints | `benchmark/prepare_matrix.py` |
| Pilot preparation cost and record exclusions | `qualification.py` |
| Queue phases, run batches and wake the continuation session | `benchmark/queue_phases.py`, `phase_batch.py`, `watch_campaign.py` |
| Construct game recipes | `families.py`, `media.py`, `games.py` |
| Freeze structured exact games above twenty players | `structured.py` |
| Freeze and qualify game snapshots | `prepare.py`, `materialize.py` |
| Compute exact values and interactions from a table | `exact.py` |
| Authenticate payoff checkpoints and record evaluation costs | `payoff_cache.py` |
| Run, count queries and checkpoint | `runner.py`, `execution.py` |
| Calculate scores, Elo and history | `summary.py` |
| Authenticate and combine completed phase batches | `campaign.py`, `benchmark/export_phase.py` |
| Export the website and reproduction archive | `report.py`, `bundle.py` |
| Load and verify compact report records | `benchmark/site/records.js` |
| Render the interface | `benchmark/site/app.js`, `style.css`, `index.html` |
| Draw performance and history charts | `benchmark/site/charts.js` |
| Explain the benchmark | `benchmark/site/about.html`, `about.css`, `about.js` |
| Share report labels and render provenance tables | `benchmark/site/protocol.js` |
| Describe estimators and link sources | `benchmark/site/methods.js` |

Python files above live in `src/shapiq_benchmark/`. Start with `prepare.py` for
snapshot construction and `runner.py` for estimator execution. The parallel
preparation script separates task planning, worker execution and final assembly.
In the browser, `app.js` manages data, filters and tables; `charts.js` draws the
charts. Both are ordinary scripts with no frontend build step. The About page reuses
`protocol.js` and loads the generated `about.json`, which contains only snapshot,
suite and public game metadata rather than the full evaluation records. Both
Python exports and the Pages workflow produce this file; it is not tracked in Git.

## Archived all-family suite

This earlier design is retained for reference. The current website cohort and
stronger-model rollout are described below; these counts are not live coverage.
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
minimum. This archived suite is not the current phase-two release.

## Current rollout

The old preparations and sweeps **360900, 360901, 361328 and 361353 were
cancelled at the user's request**. Their completion watchers are disabled and
those outputs will not be published. The current release replaces the earlier
preview with the audited phase-two cohort below.

The **[roadmap](ROADMAP.md)** lists all 63 datasets, model profiles and
construction mappings. The first replacement cohort is **64 games**: four
datasets × two models × two local constructions × four seeds, with 12 real
features, six explanation targets and nine relative budgets. Random forests
have 100 trees without a depth cap; XGBoost uses up to 200 depth-eight trees
with validation-based early stopping. Fit, validation and held-out partitions
are disjoint; fitting is capped at 5,000 rows. Each fitted model is authenticated
and reused across its game constructions.

Generate the executable manifest with:

```bash
UV_NO_SYNC=1 uv run python -m shapiq_benchmark.protocol --phase 2 --output /tmp/phase2.json
```

Phases 2–7 produce compatible executable configurations. Missing exact solvers
and incompatible pairings remain explicit in the inventory. Selection is a plan:
each actual game must still qualify before evaluation, and each phase needs an
independent audit before publication. Device choices and model parameters are part
of the recorded recipe, and the website shows actual dataset/model provenance.

Preparation **361441** and sweep **361442** completed. The release uses frozen
source [`893a5da7`](https://github.com/rtealwitter/shapiq/commit/893a5da7757f5896fff1bddfffa1a82427781f01),
including the approved LeverageSHAP, OddSHAP, ProxySPEX and SVARM fixes.
All **76,032 planned cells** are accounted for: **27,919 successful evaluations**,
2,432 under-budget runs (SPEX/ShaplEIG), 177 ShaplEIG timeouts, and 45,504
unsupported target combinations. Unsupported targets are excluded from coverage;
no evaluations are pending. All other methods have complete supported coverage.
The sweep took 2 h 45 min on 128 pinned single-thread workers on an exclusive
AMD EPYC 9754 node, with a 600-second per-cell limit. Concurrent timings remain
labeled diagnostic. All games have **12 players** and four construction seeds.

Exact references and fitted-model quality passed independent preparation audit;
the completed records and public exports passed separate release audits.
All 32 fitted models outperform their held-out dummy baseline. The games are
still mostly low order: first-order Fourier energy spans roughly 60–99%, and
energy above order three is at most 2.07%. Greater model depth alone does not
establish difficult high-order games. Later phases expand constructions and
models without silently selecting games based on estimator results.

[Release data and reproduction archive](https://github.com/rtealwitter/shapiq/releases/tag/benchmark-phase2-2026-10-01)
retain the frozen source, exact game tables and per-run provenance.

Use GPUs for game preparation when measured faster: the first fixed-TabPFN
pilot strongly favored the L40S. RF and the initial XGBoost cohort stay on CPUs;
GP and other workloads need their own measurements. Estimator evaluation keeps
its standardized single-CPU profile. Cached-query charges retain their actual
preparation hardware; they are estimates, not measured uncached runtime.

For the shipped TabPFN contextualization game, a GPU recipe explicitly adds
`"device": "cuda"` to its family specification, for example:

```json
{"id": "tabpfn-adult-d12-cuda", "family": "tabpfn", "dataset": "adult_census", "n_players": 12, "device": "cuda"}
```

Submit its suite from the frozen checkout with
`sbatch benchmark/prepare_gpu.sbatch SUITE STAGE SNAPSHOT`. The launcher allocates
one L40S and runs one preparation worker. CPU remains the default; unavailable
CUDA fails explicitly. GPU recipes use float32, separate authenticated payoff
caches and recorded backend versions. This enables preparation, not automatic
publication: the full saved table still needs its usual exact-reference audit.

Cached payoff tables also yield a Boolean Fourier spectrum without additional
queries. It describes game complexity alongside predictive validation scores;
it is not an automatic exclusion rule or a Shapley interaction index.

The inventory below describes the previous shallow recipes, retained for
reproduction. It is not the configuration of the stronger-model rollout.

### Expanded matrix using shapiq's datasets

[suites/matrix.json](suites/matrix.json) now selects **63 datasets**: the six
retained small datasets, shapiq's **Wine Quality**, **Adult Census**, **Mushroom**,
**Ionosphere**, **NHANES I**, **Communities and Crime**, and **all 51 TabArena
loaders** shipped in `shapiq_games.datasets`. Wine Quality is a regression dataset
with 12 columns (including wine type); it replaces sklearn's 13-column Wine
classification dataset in this new matrix. The old `wine` key remains available
only for explicitly selected historical recipes.

The implementation stays close to the library: [datasets.py](../src/shapiq_benchmark/datasets.py)
invokes the declared shapiq loader, and [families.py](../src/shapiq_benchmark/families.py)
passes its data to shipped game constructors. The explicit
[dataset catalog](../src/shapiq_benchmark/dataset_catalog.py) records loader names,
source links, dimensions, task types and categorical columns. TabArena task types
were verified against OpenML tasks; numeric target codes alone do not determine
whether a task is classification or regression.

Each compatible dataset × recipe × player count gets **four construction seeds**.
Player counts remain at least **11**, with enumeration capped at **20**. The
expansion currently selects **3,730 tabular settings** from **13,866 candidates**,
plus the existing 12 special and 13 structured settings: **89,900 game/target
definitions** before runtime qualification. This is a preparation plan, not
completed results or a change to the cancelled frozen campaigns.

Dataset size and player count mean different things. A larger dataset supplies
more rows and columns, but these bounded recipes still use at most **512 training
rows and 128 held-out rows**; some games use smaller documented backgrounds.
Feature games select original columns, row games select training examples, and
ensemble games select models. We do not train on a million rows for every coalition.

Models are fixed by recipe, not another Cartesian-product axis:

| Recipe | Model |
| --- | --- |
| Ordinary local/global explanations; feature/data/group valuation | Decision tree, depth 3, minimum leaf size 5 |
| Forest local explanation | 8-tree random forest, depth 4 |
| Path-dependent / interventional tree | Decision tree, depth 3 / 4 |
| Uncertainty | 8-tree random forest classifier, depth 3 |
| Product kernel | RBF SVC/SVR with `gamma="scale"` |
| Heterogeneous ensemble | Logistic regression/Ridge, SVC/SVR, 3-NN, then trees of varying depth |
| Forest ensemble | One tree per player, depth 3 |
| Neighbor games | 3-NN, distance-weighted 3-NN, or radius neighbors |
| Clustering / unsupervised dependence | 3-cluster K-means / no fitted prediction model |
| TabPFN | CPU classifier with one ensemble member |

The dataset task chooses classifier versus regressor where supported. These are
bounded benchmark choices; the shipped game constructors accept other models.
Shallow trees may use fewer features than the declared player count. Broader model
choices are a separate expansion, not implied by adding datasets.

Compatibility and preprocessing are explicit:

- Use the loader's encoding and target unchanged. Shapiq may already preprocess
  using the full dataset. Remaining missing inputs (notably Mushroom and NHANES)
  use medians computed only from the recorded training rows.
- Gaussian/Gaussian-copula imputation selects noncategorical columns with more
  than two observed values. New clustering recipes use `cluster_continuous_v1`:
  exclude declared categorical columns and require more than three distinct
  values on the actual clustering rows, even when categorical metadata is absent.
  Legacy `cluster` recipes retain their original selection and cache identity.
  Combinations with too few eligible columns are excluded or
  fail qualification, rather than padding features or clipping scores.
- Classification-only games exclude regression targets. Binary product-kernel
  games require two classes. Neighbor games need enough players for the classes.
- NHANES's signed survival labels are retained as a **regression surrogate**.
  This does not claim to fit a censoring-aware survival model.
- Constants, missing data, rare classes and near-zero truth are still checked at
  construction/qualification. Selection does not guarantee a valid measured game.

Clustering tables also receive a score-independent numerical check. A
Calinski-Harabasz payoff at least `(N - 2) / float64_eps` implies within-cluster
variance at most machine precision relative to between-cluster variance, for
any actual cluster count. Such games remain available as controls, with original
payoffs and estimator results unchanged. Cumulative campaign exports apply this
versioned check to older tables too and record the previous quality role.
Raw reports and reproduction snapshots retain their recorded preparation roles;
the website's exported `game_quality.clustering_numerics` documents the additional
publication check. Reproducing the headline cohort requires the campaign exporter,
not just merging raw reports. Large Fourier mass caused
by these numerical spikes is not evidence of useful interaction diversity.

TabArena downloads use the shipped OpenML loaders and their CSV cache. Install
`openml` and a parquet engine such as `pyarrow` in the **preparation environment**
before the first uncached load (`pip install "shapiq[benchmark]"`). The matrix
preparation driver warms these caches before starting parallel workers, and games
always reload the CSV representation so first-download rounding cannot differ.
Copy the warmed caches into a new frozen checkout before an offline run. Never change the shared environment of running
jobs. Cached CSVs are ignored by Git; no dataset or result payload is committed.

Generate a separate suite; do not overwrite the running campaign's frozen suite:

```bash
uv run python -m shapiq_benchmark.matrix \
  --config benchmark/suites/matrix.json \
  --output benchmark/results/dataset-expansion/suite.json
```

Full enumeration requires a new immutable source checkout, its own staging/cache
identity and completion watcher. Existing matrix20 chunks are not automatically
compatible with this expansion. The seven-dataset campaign below is archived;
it was cancelled in favor of the phased stronger-model rollout.

### Archived seven-dataset matrix (cancelled)

The [frozen v2 configuration](https://github.com/rtealwitter/shapiq/blob/0762a2e503ad0d5852308b06427feabedd2a534d/benchmark/suites/matrix.json) crosses 23 tabular recipes with seven
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

The generated suite records selected combinations and exclusion reasons in
`matrix_coverage`. Its source and incomplete outputs are retained for diagnosis,
but jobs **360900/360901** and their watchers were cancelled. These historical
counts are not the current website cohort and must not be published over it.
The [phased roadmap](ROADMAP.md) supersedes this execution plan.

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
808 game/target definitions. All its recipes qualified. The current filtered
view removes OddSHAP's superseded measurements pending its corrected rerun;
it contains **152,712 cells across 21 methods: 57,449 successful, 88,344 unsupported
and 6,919 failed**, with none pending. Other measurements, including corrected
LeverageSHAP, are unchanged. Rankings, Elo and history were recomputed.
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
can be reused on the identical snapshot. Private candidates do not access the
production duplicate registry or execute historical baseline constructor options.
Built-in reruns require compatible constructors and access to the suite's registry;
otherwise prepare a local suite and rerun both methods there. Rerun both methods
on the same hardware
for runtime comparisons. The report rejects mismatched snapshots, conflicting
method versions, and duplicate measurements. Nothing uploads automatically.
You can also open the site's `index.html` and choose an exported `data.json`
through **Open local**.

For the full public panel, download its frozen snapshot and baseline results from
the [reproduction release](https://github.com/rtealwitter/shapiq/releases/tag/benchmark-odd-withdrawal-2026-10-01).
The archive includes exact truth, checksums, software provenance, and local
candidate commands, so there is no need to refit the games. Use `--games` and
`--methods` to select a smaller experiment when needed.

## Legacy expanded-suite commands

These older templates reserve a whole node. The active phased rollout uses
16-CPU preparation and 32-CPU evaluation instead; follow the
[continuation procedure](#continuing-through-the-phases) and its existing job
journal. Do not submit these commands to restart an active campaign.

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
