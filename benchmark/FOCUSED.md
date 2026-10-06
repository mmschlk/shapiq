# Focused Shapley estimator benchmark

One fixed benchmark, one final release. The target is a reproducible run within
one day on 128 CPUs, subject to measured preparation and execution costs. The
campaign retains its 2,048 allocated CPU-hour cap, including checks, recovery
and export. No production GPUs are used.

## Scope

The canonical [recipe CSV](suites/focused.csv) defines 112 recipes and four
construction seeds: 448 intended instances across twelve datasets. Six original
recipes failed model-quality checks on at least one seed, leaving at most 424
qualified instances. The new recipes still require those same checks.

| Application | Recipes | Intended instances | Maximum qualified instances | Players |
|---|---:|---:|---:|---|
| Individual predictions | 32 | 128 | 124 | 12–512 features |
| Data valuation | 40 | 160 | 152 | 12/14 groups; 32–512 neighbor examples |
| Feature selection | 40 | 160 | 148 | 12/14 features |

The 360 original qualified instances and their exact references are already
prepared. Reuse their authenticated artifacts and preserved successful results;
never regenerate them merely because the full manifest now has more rows. The
original recipe definitions and estimator settings remain unchanged.

Original datasets: Adult Census, Wine Quality, Mushroom, Ionosphere,
Communities and Crime, NHANES, Bioresponse and QSAR-TID11. Models are primarily
random forests, XGBoost and LightGBM, with selected SVM, MLP, GP and TabPFN
prediction/feature-selection comparisons.

The remaining matched recipes are:

| Dataset | Model | Valuation group players | Valuation input features | Feature-selection players |
|---|---|---|---|---|
| Breast Cancer | Random forest | 12, 14 | 24 | 12, 14 |
| Digits | Linear classifier | 12, 14 | 64 | 12, 14 |
| Miami Housing | LightGBM | 12, 14 | 15 | 12, 14 |
| Superconductivity | Linear regression | 12, 14 | 64 | 12, 14 |

Each pair shares its seeded row splits and model-selection protocol. Feature
subsets are nested. Valuation changes only the number of groups while retaining
the declared input columns; input width is not the game's player count.

## Ground truth

| Game | Reference |
|---|---|
| Small fixed-model explanations | Every coalition evaluated and saved |
| Feature selection | Retraining payoff for every feature subset |
| Grouped data valuation | Retraining payoff for every group subset |
| Large fixed tree explanations | Specialized exact solver for the declared explanation game |
| KNN/threshold-NN valuation | Specialized exact SV solver |

At 12/14 players, complete enumeration requires 4,096/16,384 payoff evaluations.
A tree predictor does not make a retraining game eligible for TreeSHAP. Exact
references describe the frozen game, including its recorded randomness, up to
documented numerical error. Approximate high-budget estimates are not truth.

Enumerated games request SV and order-two SII, k-SII, STII, FSII and FBII.
Native path-dependent trees request SV, SII and k-SII; interventional trees
request all six; neighbors request SV only. Validate native references against
small exhaustive games and qualify the actual requested sizes and all seeds.

## Estimators and scoring

Use the same 22 estimator configurations, four construction seeds, one estimator
seed and nine relative budgets: 0.5, 1, 2, 4, 8, 16, 32, 64 and 128 times the
player count. LeverageSHAP/OddSHAP retain the recorded approved two-query fallback.
Worker limits are 30 seconds, or 120 seconds when both d >= 128 and budget >= 32d;
startup counts. Failures and unsupported combinations remain explicit.

Python and browser summaries weight applications equally, then subtypes, recipes
and instances. Data-valuation SV balances retraining groups and neighbor examples.
Zero/near-zero reference signal is excluded consistently, before estimator
results are inspected. Preserve original games, truth, seeds and result ownership.

## Running the benchmark

Generate the suite from a frozen checkout with the approved estimator fixes:

```bash
UV_NO_SYNC=1 uv run python -m shapiq_benchmark.focused benchmark/suites/focused.csv /path/on/lab/storage/suite.json
```

The command only validates and writes a new manifest. Preparation, evaluation and
export use the existing modules below. Reconcile the manifest with authenticated
prepared recipe identities and run only missing work. Select resource exclusions
from model-quality and measured cost rules, never estimator accuracy.

Current operational state lives in the campaign's WAKEUP.md, state.json and
budget.json on lab storage. The original selective recovery continues with
production/selective-recovery/submit_slice_v3.py, selecting never-attempted cells.
Each admission reserves its worst-case allocation before release and settles
actual Slurm CPUTimeRAW after terminal, nonlive accounting. Recovery is currently
qualified for 64 workers; 128-worker execution needs separate concurrency evidence.

The working remaining-cost envelopes are 320 CPU-hours for original recovery and
final work and 400 CPU-hours for the 64 newly specified instances, including
qualification, preparation, evaluation and checks. They are admission bounds,
not measured runtime promises. Every submission must still fit the live ledger.

Publish the single focused cohort only after independent ownership, numerical,
coverage and browser audits. Keep the existing website until the replacement
passes. Watcher checks are hourly, with routine progress reminders every two hours.

## Code map

| Responsibility | Reusable implementation |
|---|---|
| Dataset/model/player choices | [suites/focused.csv](suites/focused.csv) |
| Manifest validation | [focused.py](../src/shapiq_benchmark/focused.py) |
| Cost qualification and CPU ledger | [focused_campaign.py](focused_campaign.py) |
| Exact game preparation | [prepare_matrix.py](prepare_matrix.py) |
| Games and retraining adapters | [families.py](../src/shapiq_benchmark/families.py) |
| Evaluation with counted queries | [runner.py](../src/shapiq_benchmark/runner.py) |
| Static report export | [report.py](../src/shapiq_benchmark/report.py) |
| Public explanation | [site/about.md](site/about.md) |

Benchmark code stays separate from the deployed shapiq package. Estimator fixes
are reviewed independently in PR #611; the Monte Carlo speedup is PR #612.
