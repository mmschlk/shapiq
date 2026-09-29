# An interactive benchmark for Shapley estimators

**Status: proposal for discussion; this branch adds only this plan.** No benchmark
results, website, or deployment have been created. Inventory checked on 2026-09-29
against shapiq commit `6321cbdaefee7af7a4bea58ca15e7cf2450f4691`.

[Discuss the proposal in issue #601](https://github.com/mmschlk/shapiq/issues/601)
or [review the document in draft PR #602](https://github.com/mmschlk/shapiq/pull/602).

## The idea in two minutes

Build a public website that answers **“Which estimator works best for games like
mine, with the evaluations or time I can afford?”** Cover Shapley values and
interaction estimators, with reproducible experiments behind every chart.

The recommended design is deliberately small:

1. **Python runs experiments offline**, using shapiq's existing estimators, games,
   and exact-value computers.
2. **Versioned JSON files hold the results.** Each result links back to its game,
   parameters, seed, ground truth, and code version.
3. **One static HTML/CSS/JavaScript site** filters those files and draws charts with
   Plotly.js. GitHub Pages hosts it; no running Python service or database.

The first release should show a leaderboard, error versus evaluation budget,
error versus runtime, direct head-to-head comparisons, and a historical progress
chart. Researchers should also be able to add an estimator **locally**, evaluate
it on the same frozen suite, and open a private comparison report without editing
shapiq's source or publishing anything. Start with a small, trustworthy suite, then expand through an explicit
coverage checklist to every supported estimator and constructible game family.

“All games at all budgets” needs a reproducible interpretation: games have
unbounded parameter choices, exact enumeration costs grow exponentially, and some
methods only work on particular targets or models. We will publish a **finite,
versioned experiment matrix**, list every method/family, and show whether each
combination is measured, unsupported, blocked, or still planned. Arbitrary budget
inputs select available measurements; they do not invent new results.

Use a **feature-request issue as the discussion home**, linked to a draft PR for
this document. This follows the repository's [contribution guidance](../.github/CONTRIBUTING.md)
and gives decisions a durable, reviewable home. Implementation follows in small PRs.

### Reading guide

- [What users will see](#what-users-will-see)
- [What shapiq already contains](#what-shapiq-already-contains)
- [Rules for a fair comparison](#rules-for-a-fair-comparison)
- [Release history and the progress chart](#release-history-and-the-progress-chart)
- [The code to write](#the-code-to-write)
- [Evaluate a paper or a new estimator locally](#evaluate-a-paper-or-a-new-estimator-locally)
- [What to run and how much it costs](#what-to-run-and-how-much-it-costs)
- [Hosting and publication](#hosting-and-publication)
- [Small implementation steps](#small-implementation-steps)
- [Decisions for discussion](#decisions-for-discussion)

## What users will see

Take inspiration from [Artificial Analysis](https://artificialanalysis.ai/): a
readable comparison table, shared filters, and charts connecting accuracy with
resource cost. Use our own visual design and scientific definitions.

```text
Shapiq Benchmark                         Suite v1 | Methodology | Download

Target: Shapley values       Games: tabular explanation     Players: 8–16
Budget: 1,024 evaluations    [absolute / fraction of all coalitions]
Methods: all eligible       Hardware: CPU reference        Reset | Copy link

Leaderboard     Error vs budget     Error vs seconds     Head-to-head     History
Method | mean nMSE ↓ | median nMSE ↓ | seconds | coverage | failure rate

Every chart: selected games, target, budget rule, sample count, data version
```

Filters should include game family, dataset, model family, player count, target
index, maximum order, scored interaction order, absolute budget, relative budget,
and timing mode. “Similar games” initially means these understandable filters and
curated presets (for example, small tabular games or large sparse synthetic games).
An automatic similarity model would add complexity before we know it is useful.

| View | What it answers | Required behavior |
| --- | --- | --- |
| Leaderboard | Which methods work well in this selected setting? | Sortable mean/median nMSE, uncertainty for published presets, runtime, coverage, failures; expandable method details. |
| Budget curve | What accuracy does more evaluation budget buy? | Measured points, optional log axes, actual budgets in tooltips; gaps for missing points. |
| Runtime curve | What accuracy can I obtain in a given time? | Separate cached-oracle and live-game timing, one hardware profile at a time. |
| Overall budget summary | Which method works well across the selected budget regime? | Average over a fixed declared budget grid; show the grid and target. |
| Head-to-head | How often does A beat B on the same tasks? | Win/tie/loss matrix, paired counts, plus Elo-scale ratings for published presets. |
| History | When did methods capable of lower error become available? | Method release markers, horizontal score lines, and a best-so-far step curve. |
| Coverage | What has actually been tested? | Every estimator/family/target with status and exclusion reason. |

Show “lower is better” beside error metrics. Display exact-zero errors explicitly
on log charts instead of quietly replacing them with positive values. Keep an
accessible HTML table alongside charts, keyboard-operable controls, responsive
layout, CSV download of the selected data, and filters stored in the URL. Empty
selections should explain what is missing. Keep the suite version in shared URLs.

## What shapiq already contains

### Estimators

There are **22 concrete public estimator classes** exported by
[`shapiq.approximator`](../src/shapiq/approximator/__init__.py), excluding the base
class and optional-dependency placeholders. The table below groups all of them.
These are source-level capabilities; each selected configuration still needs a
small smoke run before being admitted to a published suite.

| Family and source | Classes | Target notes |
| --- | --- | --- |
| [Marginal sampling](../src/shapiq/approximator/marginals) | `OwenSamplingSV`, `StratifiedSamplingSV` | SV. |
| [Permutation sampling](../src/shapiq/approximator/permutation) | `PermutationSamplingSV`, `PermutationSamplingSII`, `PermutationSamplingSTII` | Respectively SV; SII/k-SII; STII. |
| [Monte Carlo](../src/shapiq/approximator/montecarlo) | `UnbiasedKernelSHAP`, `SVARM`, `SHAPIQ`, `SVARMIQ` | UnbiasedKernelSHAP: SV; SVARM: SV/BV. SHAPIQ/SVARMIQ declare SV, SII, k-SII, STII, FSII, FBII, CHII, BII, BV. |
| [Regression](../src/shapiq/approximator/regression) | `KernelSHAP`, `LeverageSHAP`, `OddSHAP`, `kADDSHAP` | First three: SV. kADDSHAP advertises SV compatibility but constructs a kADD-SHAP target; validate its SV extraction before admitting it to that panel. Its representation is not a standard interaction index. |
| [Interaction regression](../src/shapiq/approximator/regression) | `KernelSHAPIQ`, `InconsistentKernelSHAPIQ`, `RegressionFSII`, `RegressionFBII` | Kernel variants: SII/k-SII, with order-one SV specialization. Faithful variants: FSII/SV and FBII/BV respectively. |
| [Proxy models](../src/shapiq/approximator/proxy) | `ProxySHAP`, `RegressionMSR`, `ProxySPEX` | ProxySHAP and ProxySPEX route SV, SII, k-SII, STII, FSII, FBII, BII, BV, subject to backend support. RegressionMSR estimates SV/BV using regression adjustment with a chosen proxy; pass its index explicitly. |
| [Sparse recovery](../src/shapiq/approximator/sparse) | `SPEX` | Sparse transform and Möbius conversion to SV, BV, SII, BII, k-SII, STII, FSII, FBII; budget feasibility depends on configuration. |
| [Active learning](../src/shapiq/approximator/shapleig) | `ShaplEIG` | SV; requires the optional PyTorch/GP stack. |

SV means Shapley value. SII, k-SII, STII, and FSII are different definitions of
Shapley interactions; they need separate ground truths and leaderboards. BV/BII/
FBII are Banzhaf-related targets and belong on separate, later tracks. CHII and
other converter-supported targets can be added with the same registration process.
See [`game_theory`](../src/shapiq/game_theory) for definitions and conversions.

Do not build coverage by blindly concatenating `SV_APPROXIMATORS` and
`SI_APPROXIMATORS`: they overlap and omit valid routes, including ProxySHAP's SV
route and RegressionFBII from the general interaction list. Use a short explicit
registry with constructor adapters, tested output targets, supported orders,
minimum budgets, optional dependencies, and default parameters. Do not count
equivalent order-one specializations as separate independent discoveries; mark
aliases/variants in method metadata.

[`pyproject.toml`](../pyproject.toml) already defines `sparse`, `proxy`, `shapleig`,
`tree`, and `benchmark` extras. Some game modalities need additional ML packages.
Record a missing dependency as such, rather than silently losing a method.

### Exact methods and model-specific explainers

Reuse [`ExactComputer`](../src/shapiq/game_theory/exact.py),
[`MoebiusConverter`](../src/shapiq/game_theory/moebius_converter.py), and
[`shapiq_benchmark.computers`](../src/shapiq_benchmark/computers.py) for truth where
their assumptions match the game. [`Game`](../src/shapiq/game.py) supports
precomputation and stored coalition values.

Tree, product-kernel, nearest-neighbor, and other structured explainers in
[`src/shapiq`](../src/shapiq) have access to model structure that a generic oracle
estimator does not. Include them as exact references or in a **separate structured
track**, with their supported models, semantics, and preprocessing costs stated.
They must not win a black-box evaluation-budget leaderboard by receiving free
access to the fitted model. Likewise, a proxy method may fit a model using its
budgeted queries, but may not receive the original model or truth table for free.
For trees, record the explicit backend (`TreeSHAPIQ`, `LinearTreeSHAP`,
`QuadratureTreeSHAP`, `InterventionalTreeSHAPIQ`, or Woodelf) instead of relying
on automatic selection, which can change with dependencies and input sizes.

### Games and datasets

The reusable code has three layers; they are not three independent benchmark suites.

| Layer | Existing code | How to use it |
| --- | --- | --- |
| Current game implementations | [`shapiq_games/synthetic`](../src/shapiq_games/synthetic), [`shapiq_games/tabular`](../src/shapiq_games/tabular), and domain APIs in `shapiq` | Build reproducible game factories. SOUM and unanimity games are especially useful for known truth. |
| Current benchmark helpers | [`shapiq_benchmark`](../src/shapiq_benchmark) | Reuse `LocalXAIBench`, `InterventionalBench`, `PathdependentBench`, `TabPFNBench`, `ImageBench`, and their ground-truth computers after capability checks. |
| Deprecated game collection | [`shapiq_games/benchmark`](../src/shapiq_games/benchmark) | Inventory its families and adapt useful factories incrementally. Its `__init__.py` explicitly marks it deprecated; some current helpers still depend on it. |

The legacy collection contains local explanations (tabular, language, image),
global explanations, feature selection, data valuation, dataset valuation,
ensemble selection including random-forest variants, uncertainty, unsupervised
clustering, unsupervised data, tree explanations, interventional tree explanations,
and causal explanations. The `product_kernel` files are placeholders, not runnable
games. There is a working core
[`ProductKernelGame`](../src/shapiq/explainer/product_kernel/game.py), plus
[`InterventionalGame`](../src/shapiq/tree/interventional/game.py) and
[nearest-neighbor games](../src/shapiq/explainer/nn/games), to catalog separately.
The latter include `KNNExplainerGame`, `TNNExplainerGame`,
`WeightedKNNExplainerGame`, and `BinaryWeightedKNNExplainerGame`, with training
rows as players. Register every family, including blocked ones, so broad coverage
is visible.

Dataset discovery starts in
[`shapiq_games/datasets`](../src/shapiq_games/datasets/__init__.py): **83 exported
loaders**, comprising 32 non-TabArena loaders and 51 TabArena loaders. Examples
include Adult Census, California Housing, Communities and Crime, Bike Sharing,
Breast Cancer, Wine Quality, and the TabArena collection. Some loaders overlap or
generate synthetic data; 83 loaders does not mean 83 independent datasets. Loader existence
does not establish that a game constructor, model, or exact truth is available.

The newer [`setup.py`](../src/shapiq_benchmark/setup.py) explicitly lists **54
dataset identifiers** (Adult Census, California Housing, Communities and Crime,
plus 51 TabArena identifiers) and model choices spanning decision trees, random
forests, XGBoost, LightGBM, MLP, TabPFN, ViT, and ResNet. The supported combinations
vary by wrapper. Legacy tabular local-XAI classes cover a broader set than their
package exports, so inspect the implementation modules as well as `__all__`.

<details>
<summary>Complete dataset-loader inventory and legacy game variants</summary>

The 32 non-TabArena exports are Adult Census, Annealing, Arrhythmia, Bike Sharing,
Breast Cancer, California Housing, Hepatitis, Ionosphere, Mushroom, Nursery,
Soybean, Thyroid, Zoo, Forest Fires, Amazon, Bioresponse, Communities and Crime,
NHANES I, Real Estate, Microresponse, Leukemia, Wine Quality, independentlinear60,
corrgroups60, condind, xor, group, cross, chess, sphere, disjunct, and
Curth–van der Schaar synthetic. `_all.py` also defines the unexported `load_random`.

The 51 `load_tabarena_` suffixes are:

```text
airfoil_self_noise, amazon_employee_access, anneal, fiat_500, aps_failure,
bank_marketing, bank_customer_churn, bioresponse, blood_transfusion, churn,
coil2000, concrete_strength, credit_g, credit_card_default, airline_satisfaction,
diabetes, diabetes130us, diamonds, ecommerce_shipping, fitness_club,
food_delivery, give_me_credit, hazelnut, health_insurance, heloc,
hiva_agnostic, houses, hr_analytics, coupon_recommendation, good_customer,
kddcup09, marketing_campaign, maternal_health, miami_housing, online_shoppers,
protein, bankruptcy, qsar_biodeg, qsar_tid11, qsar_fish_toxicity,
sdss17, seismic_bumps, splice, students_dropout, superconductivity,
taiwanese_bankruptcy, website_phishing, wine_quality, naticusdroid, jm1, mic
```

| Legacy family | Existing variants |
| --- | --- |
| Local tabular explanation | 82 concrete classes: 31 older variants plus 51 TabArena variants. |
| Local image / language | `ImageClassifier` (ViT/ResNet); `SentimentAnalysis`. |
| Global explanation, feature selection, data valuation, dataset valuation, ensemble selection, random-forest ensemble selection, clustering, unsupervised data | Adult Census, Bike Sharing, California Housing variants in each family. |
| Uncertainty | Adult Census. |
| Path-dependent tree explanation | Adult Census, Bike Sharing, California Housing, synthetic. |
| Interventional tree explanation | Those four plus Annealing, Arrhythmia, Breast Cancer, Hepatitis, Ionosphere, Mushroom, Nursery, Soybean, Thyroid, Zoo. |
| Causal/confounding explanation | `GlobalConfoundingXAI`, `LocalConfoundingXAI`, `CurthVDS`; not top-level exports. |

Some CSVs ship with the package; other loaders fetch upstream data. TabArena
loaders fetch explicit OpenML IDs, preprocess, and cache into the package data
directory. Audit classification/regression labels explicitly rather than relying
on setup heuristics. TabArena requires `openml`, and some UCI loaders need
`ucimlrepo`; neither is currently declared in `pyproject.toml` or `uv.lock`.
Qualify dependencies and add needed extras in an implementation PR; installing
`shapiq[benchmark]` alone does not enable every loader.
`TabularLocalExplanation` supports baseline, marginal, and generative conditional
imputation, but its string `"tabpfn"` option is unimplemented; use the separate
`TabPFNBench`/`TabPFNImputer` pathway when qualified.

</details>

The future generated catalog should enumerate every loader and concrete game
factory with: family, dataset source/version/license, model and parameters,
player definition, preprocessing, missing-feature semantics, background sample,
prediction output/class, explanation point, seeds, supported truth routes, and
availability status. For data valuation a player may be a row or group of rows;
for images it may be a patch. These are different games even on the same dataset.

### Benchmarks, metrics, and data already present

The current benchmark package has game/model setup, exact-value computers,
Optuna setup and a small set of saved tuning results, metric functions, and JSON
serialization. It does **not** contain a complete estimator × game × budget sweep
runner, a public leaderboard dataset, a website, or an Elo implementation.

[`metrics.py`](../src/shapiq_benchmark/metrics.py) includes MSE/MAE/SSE/SAE,
Kendall/Spearman correlations, precision at k, and a faithfulness measure. It has
no normalized MSE. Before reuse, test coordinate alignment and baseline removal:
the MSE denominator can still include order zero, the Spearman `k` argument is
unused, and the Kendall implementation compares sorted coordinate indices.
These are reasons to validate metric semantics before publishing a ranking.

Existing [`tests/shapiq_benchmark`](../tests/shapiq_benchmark) and
[`tests/shapiq/data`](../tests/shapiq/data) provide useful setup and exact-value
fixtures, but are not leaderboard observations. The older
[benchmark study issue #67](https://github.com/mmschlk/shapiq/issues/67) and
[migration issue #459](https://github.com/mmschlk/shapiq/issues/459) are relevant
history. The separate [shapiq-benchmark repository](https://github.com/mmschlk/shapiq-benchmark)
is a small prototype with tree benchmarks and metrics, old import paths, and no
general sweep runner or published result assets found in its default branch.
Build on the current in-repository helpers; confirm repository ownership in the
discussion rather than creating a second framework.

## Rules for a fair comparison

### 1. Freeze the game and its ground truth

A task is one immutable game instance, target index, maximum interaction order,
and scored coordinate set. A run adds an estimator configuration, budget, and
replicate seed. Store content hashes so these identities are independent of filenames.

Use analytic truth for SOUM where supported; otherwise enumerate all `2^n`
coalitions when feasible, or use a validated model-specific exact computer with
the **same game semantics**. Cross-check each structured truth route against
enumeration on small examples before trusting larger cases. Marginal,
interventional, conditional, and tree-path-dependent games are not interchangeable.

Freeze data splits, trained models, imputations/background samples, stochastic
game randomness, and explanation points before running estimators. Truth must
refer to that exact frozen game. An approximate large-budget answer is a
**reference estimate**, never labelled ground truth; keep that exploratory track
out of exact-truth rankings. Truth/precomputation time is recorded separately and
never charged to an estimator or exposed through its oracle.

Raw `RandomGame` is unsuitable as a deterministic oracle: its implementation
draws values by batch position rather than coalition identity. Either create one
fixed coalition table and expose only lookup, or exclude it with a reason.
Game qualification must check repeated calls, permuted batches, and batch-size
independence. Use a fresh estimator for every run; do not leak truth caches.
Also note that partial `Game.precompute()` creates lookup-only behavior; it is
not a transparent cache that evaluates previously unseen coalitions on demand.

### 2. Define normalized MSE once

For the same set of **nonempty** coordinates `J`, ground truth `phi`, and estimate
`phi_hat`, define:

```text
MSE  = sum((phi_hat[j] - phi[j])² for j in J) / len(J)
nMSE = sum((phi_hat[j] - phi[j])² for j in J) / sum(phi[j]² for j in J)
```

Thus nMSE = 0 is perfect and nMSE = 1 is the error of predicting zero everywhere.
Example: truth `[1, -1]` and prediction `[0, 0]` give MSE = 1 and nMSE = 1.
Scaling both truth and prediction leaves nMSE unchanged. Do not normalize by the
sum of attributions: positive and negative effects can cancel.
Show reference lines at 1 (zero prediction) and 0 (exact answer), outside the
estimator inventory, Elo competitors, and release-history frontier.
`Game(normalize=True)` centers game outputs; it does not perform nMSE normalization.

Exclude the empty-coalition/baseline coefficient. Align values by coalition key,
not array position. Validate index, player identities, orders, and expected
coordinates before scoring. Implicit zeros are allowed only when the estimator's
documented sparse representation defines them as zeros; truncated or malformed
output is a failure, not a free zero prediction.

For zero ground-truth energy, nMSE is undefined: report `null`, raw MSE, and the
exclusion count. Flag numerically near-zero energy with a documented tolerance
relative to the game's output scale; do not hide a denominator epsilon inside
the metric or cap large errors. All-zero and cancellation fixtures must be tested.

For interactions, provide both an individual-order view and an explicitly named
orders-1-through-k view. Keep `index`, `max_order`, and scored order distinct in
the data: changing maximum order can change the quantity being estimated. Never
mix different indices or coordinate sets into one headline rank.

### 3. Measure budgets instead of trusting labels

The primary budget is **the number of coalition evaluations requested by the
estimator**, including repeats, empty/full coalitions, initialization, and tuning.
A vectorized call with 100 coalition rows costs 100, not one. Also record unique
coalitions, oracle calls/batches, cache hits, and actual model evaluations when
available. Show the requested cap and measured usage separately; some existing
`estimation_budget` fields report a requested budget rather than observed usage.

Instrument the game boundary with one small wrapper. Do not give different
methods different free caches. For cached games, repeats still consume the
primary query budget; a distinct-coalition view can be a labelled secondary view.
Count proxy fitting/validation queries and adaptive searches in the same run.
Hyperparameters are fixed on a separate development suite, never selected by
test nMSE. Publish defaults and any tuned variant as separate configurations.

Some methods need a minimum budget or consume whole batches. Preflight those
constraints. A run exceeding its declared cap is `over_budget`, not an eligible
measurement; do not silently grant extra evaluations or truncate its output.
Set process time/memory limits and preserve timeout, OOM, numerical-error,
missing-dependency, and unsupported statuses separately.

Concrete adapter checks include SPEX's transform-specific minimum; ShaplEIG's
budget above its initial design (default `n+1`) and finite candidate pool;
LeverageSHAP's default even-budget rounding; OddSHAP's minimum budget and documented
low-budget implementation differences; and permutation methods' indivisible
iterations. Capture warnings and actual backend/proxy parameters alongside results.

Offer absolute budgets and `B / 2^n` (fraction of the coalition space). Use named
low/medium/high regimes with explicit numeric boundaries in the suite manifest.
For a user cap between measured points, choose the largest **requested grid cap**
no greater than the input that is common to the eligible methods for that task;
show the selected cap. Never choose a run based on its observed error. Below the
smallest common cap, show insufficient data. Relative grids may produce different
absolute caps for different player counts; tooltips must say so.

### 4. Separate runtime measurements

Report two modes:

- **Cached oracle:** coalition values are precomputed. Measures estimator overhead
  plus lookup cost; useful for reproducible algorithm comparisons.
- **Live game:** time includes oracle evaluation and estimator setup, fitting,
  tuning, and postprocessing. The task's trained predictive model/data can be
  prepared once, but disclose and separately report that preparation cost.

Measure wall time with a monotonic clock; exclude ground-truth computation,
metric calculation, and writing results. Record estimator initialization and
execution components plus their total. Use an idle dedicated machine, fixed
thread counts, pinned software, hardware/OS/CPU/GPU metadata, and GPU synchronization
where relevant. Warm-up policy must be identical and explicit. Do not parallelize
competing timed runs on one device or combine hardware profiles in a time ranking.

Initially plot nMSE against measured time at the fixed budget grid. A time-cap
selector uses the median runtime across all planned successful repeats for each
method/task/budget point, requiring a complete repeat set for official eligibility.
Choose the largest tested budget meeting that statistic and use all its accuracy
repeats; never select individual unusually fast runs or inspect error to choose
a point. Show the fraction of repeats exceeding the cap and label the result
**observed time**, not a deadline guarantee. Below the smallest available point,
show insufficient data. Real deadline-controlled adaptive runs are a later feature.

### 5. Make aggregation and missing data visible

Use equal weights for game families, then equal weights for declared task strata
within a family (dataset/model/player-count configuration), then game instances,
then replicate seeds. This prevents a large dataset, many explanation points,
or many similar synthetic variants from dominating. Freeze the family/stratum
definitions in the suite manifest.

Compute both the weighted arithmetic mean and weighted median of run nMSE under
those weights. Label this median as the **typical run**; it is not the median of
per-game means. Define weighted median as the smallest sorted value whose
cumulative normalized weight is at least 0.5, identically in Python and JavaScript.
For “overall across budgets”, add equal weight to a fixed log-spaced
budget grid; display the included grid. This is an average over the chosen
regime, not a universal measure of estimator quality. Keep SV and each interaction
target/order separate. Report runtime and accuracy as separate columns.

For an official ranked preset, freeze a task panel for methods qualified as
compatible before results are known. Require complete finite results on its
nondegenerate tasks for a headline accuracy rank. Incomplete methods remain
visible as provisional with successful-run summaries, coverage, and failure
rates, but cannot win by dropping difficult tasks. Unsupported methods have
coverage status, not a loss. An exploratory common-task view may compare a
smaller intersection, but must display the reduced panel and exclusions.

Use 95% paired hierarchical bootstrap intervals for published presets to describe
variation across games and estimator randomness: resample independent game/model
clusters within strata, then replicate slots within clusters, keeping methods and
budgets paired at both levels. Explanation points sharing a trained model/dataset
split are clustered together. State when there
are too few independent units for meaningful intervals. Arbitrary browser filter
combinations can show descriptive mean/median/counts without claiming preset
intervals apply to them. This keeps statistical computation in Python. See
[Agarwal et al.](https://arxiv.org/abs/2108.13264) for motivation for uncertainty
reporting; the weighting and resampling protocol here are project design choices.

### 6. Add head-to-head results and Elo carefully

A match compares two estimators on the same game, target, order, budget cap, and
replicate slot. Equal seed numbers pair experiments but do not imply identical
random samples across algorithms. Lower nMSE wins. Declare ties when
`abs(a-b) <= 1e-12 + 0.01 * max(a,b)`; freeze that proposed 1% tolerance before
the production run and publish sensitivity to 0%/5% on the development suite.
Zero-energy tasks are excluded from nMSE matches. Report paired sample counts.

Show the win/tie/loss matrix first. For a stable secondary leaderboard, fit a
**batch Bradley–Terry model on an Elo scale** to weighted paired outcomes:

```text
P(A beats B) = 1 / (1 + 10^((rating_B - rating_A) / 400))
```

Treat ties as half wins; center ratings at 1,000. Use the same game weights as
the error metrics. Fit in Python using SciPy, with a fixed documented weak
regularizer to handle all-win/all-loss cases. Report bootstrap intervals and
connected comparison groups; never infer a shared rank for disconnected groups.
This is an Elo-style summary of objective error comparisons, not human votes.
It loses information about the size of an accuracy difference, so nMSE remains
the primary metric. The [Chatbot Arena paper](https://arxiv.org/abs/2403.04132)
is a useful precedent for pairwise ratings and uncertainty, not our task protocol.

Publish Elo only for fixed, versioned presets computed offline. Free-form filters
update the direct matrix and descriptive nMSE; show Elo as unavailable when no
matching preset exists, rather than leaving stale ratings on screen. Sequential
Elo would require an arbitrary match order; the batch fit avoids that dependency.

Fit official ratings using complete eligible methods only. Keep any provisional
fit involving incomplete methods separate, so their selectively successful matches
cannot change official ratings. Unsupported cells never become losses. Report
failures and omitted pairs separately; provisional comparisons cannot determine
the official winner.

## Release history and the progress chart

Each method has `first_public_method_date`, a primary-source URL, date precision,
conference/publication year, and a separate shapiq implementation commit/release.
Use the earliest verified public description of that method by default, not the
conference year or file modification date. Unknown dates stay unknown; do not
invent January 1 for a year-only source. Implementation changes can become named
variants with their own provenance.

These first-public dates were checked against the linked arXiv submission histories;
they seed the catalog rather than complete it:

| Method | First public date | Source |
| --- | --- | --- |
| StratifiedSamplingSV | 2013-06-18 | [Maleki et al.](https://arxiv.org/abs/1306.4265) |
| KernelSHAP | 2017-05-22 | [Lundberg and Lee](https://arxiv.org/abs/1705.07874) |
| OwenSamplingSV | 2020-10-22 | [Okhrati and Lipani](https://arxiv.org/abs/2010.12082) |
| UnbiasedKernelSHAP | 2020-12-02 | [Covert and Lee](https://arxiv.org/abs/2012.01536) |
| Faithful regression / Faith-Shap | 2022-03-02 | [Faith-Shap paper](https://arxiv.org/abs/2203.00870) |
| kADD-SHAP | 2022-11-03 | [kADD-SHAP paper](https://arxiv.org/abs/2211.02166) |
| SVARM | 2023-02-01 | [Kolpaczki et al.](https://arxiv.org/abs/2302.00736) |
| SHAPIQ | 2023-03-02 | [Fumagalli et al.](https://arxiv.org/abs/2303.01179) |
| SVARMIQ | 2024-01-24 | [Kolpaczki et al.](https://arxiv.org/abs/2401.13371) |
| KernelSHAPIQ | 2024-05-17 | [Fumagalli et al.](https://arxiv.org/abs/2405.10852) |
| LeverageSHAP | 2024-10-02 | [Musco and Witter](https://arxiv.org/abs/2410.01917) |
| SPEX | 2025-02-19 | [Kang et al.](https://arxiv.org/abs/2502.13870) |
| ProxySPEX | 2025-05-23 | [Butler et al.](https://arxiv.org/abs/2505.17495) |
| RegressionMSR | 2025-06-13 | [Witter et al.](https://arxiv.org/abs/2506.11849) |
| OddSHAP | 2026-02-01 | [OddSHAP paper](https://arxiv.org/abs/2602.01399) |
| ShaplEIG | 2026-06-01 | [ShaplEIG paper](https://arxiv.org/abs/2606.02247) |

Audit the remaining methods, aliases, and implementation dates before including
them in history. Earlier sampling methods may predate KernelSHAP; include them
when verified, while offering “since KernelSHAP” as a display range.

For a fixed suite version, target/order, budget rule, and hardware/timing mode:

1. Compute each complete eligible method's mean or median nMSE on the
   **same frozen task panel**. Provisional methods cannot set the frontier.
2. Draw a horizontal line from its release date to the present at that score.
3. Draw a step function `frontier(t) = min(score[m] for release[m] <= t)`.
   A new release lowers the frontier only when it improves that score.
4. Label the method responsible for each step, link its paper, and expose its
   coverage and uncertainty. Mean and median have separate frontiers.

This delivers the requested staircase. Label it **“Retrospective performance of
current implementations, grouped by method release date.”** It is not a claim
that these datasets, implementations, or timings existed then. Never recompute
older steps on different game subsets or pool unmatched targets. Freeze the
panel across the whole timeline; a coverage change requires a new suite version.
Missing dates/measurements cannot create a frontier step.

## The code to write

Keep the experiment code in `src/shapiq_benchmark` and the site in `benchmark/site`.
Extend existing helpers with explicit functions and small data records; avoid a
new experiment framework, plugin system, web server, database, or frontend build
tool until an observed need justifies one. Use NumPy/pandas/SciPy already in the
project. Use a pinned local Plotly.js bundle for charts; its
[official setup guide](https://plotly.com/javascript/getting-started/) supports
direct script loading. No React or Node build is required for the site.

```mermaid
flowchart LR
    A[Versioned suite and method catalog] --> B[Prepare frozen games and truth]
    B --> C[Run one isolated experiment at a time]
    C --> D[Validate and aggregate in Python]
    D --> E[Compact public JSON and raw result archive]
    E --> F[Static filters, tables, and charts on GitHub Pages]
```

Proposed files below **do not exist yet**. Names are an implementation guide,
not a request to create scaffolding in this planning PR.

| File or area | Responsibility | Reuse |
| --- | --- | --- |
| `benchmark/suites/{smoke,pilot,core,extended}.json` | Explicit games, methods, targets, budgets, seeds, resource limits, weights. | Existing dataset/model setup and game constructors. |
| `src/shapiq_benchmark/catalog.py` | Method/game records, constructor adapters, capabilities, dates, sources. | Public estimator classes; audited target routes. |
| `src/shapiq_benchmark/prepare.py` | Deterministic game artifacts, truth, hashes, qualification checks. | `Game`, exact computers, SOUM/Möbius conversion. |
| `src/shapiq_benchmark/run.py` | CLI, local adapter loading, expansion/dry-run, isolated jobs, budget counter, timers, resume. | Existing `Benchmark` wrappers; standard library processes. |
| `src/shapiq_benchmark/metrics.py` | Add validated nMSE and repair/test reused metric semantics. | Existing metrics where correct. |
| `src/shapiq_benchmark/report.py` | Weights, paired matches, batch ratings, bootstrap, public export or private local overlay. | NumPy/pandas/SciPy and the same site assets. Split ratings into a module only if needed. |
| `benchmark/site/{index.html,styles.css,app.js,stats.js}` | Controls, table, charts, URL state; small pure browser reductions. | Pinned Plotly.js. |
| `benchmark/site/vendor/` | Pinned chart bundle plus license and checksum. | Upstream distribution. |
| `benchmark/examples/local_estimator.py` | One documented adapter example for a paper's implementation. | Existing `approximate(budget, game)` interface. |
| `tests/shapiq_benchmark/` | Scientific invariants, runner recovery, schema and export checks. | Existing truth/setup fixtures. |
| `.github/workflows/benchmark-pages.yml` | Validate and deploy already-generated static content. | Official Pages actions. |

### Results contract

Use ordinary JSON with a schema version and strict finite-number validation;
undefined metrics are `null` with a reason, never JSON `NaN` or `Infinity`.
Write one raw result atomically per deterministic run ID. Resume only if the
configuration, code, environment, game, and truth hashes match. Record errors as
results so an interrupted sweep has an auditable completion map.

| Record | Minimum fields |
| --- | --- |
| Suite manifest | schema/suite version, code commit, lockfile/environment hash, task panel, weighting, budget grids, seeds, metric/tie definitions, artifact checksums, hardware profiles. |
| Method | stable ID, display name, class/parameters, target/order capabilities, dependencies, variant/alias, paper dates and links, implementation commit. |
| Game/task | stable IDs, family/stratum, player count and meaning, dataset/model/semantics, all construction seeds, truth method/hash, output coordinates and norm. |
| Run | task/method/config IDs, seed, requested/measured/unique queries, cache mode, timing components, hardware, status/reason, MSE/nMSE, artifact links. |
| Published summary | exact filter/profile ID, eligible panel and exclusions, weights, counts, means/medians, confidence intervals, pairwise counts/ratings, snapshot hash. |

Store raw attribution vectors, coalition tables, models, logs, and environment
details in a compressed release archive or durable research archive. The browser
loads a small catalog and per-target/order metric shards, not those large artifacts.
Keep enough per-run metric rows in shards for honest custom filtering and matching.
Preset uncertainty/ratings are separate Python-generated files.

There is deliberately a little shared arithmetic between Python export and
browser filtering. Verify both against the same tiny fixtures for weights,
mean/median, cap selection, ties, and coverage; otherwise the table and downloads
could disagree. Do not implement optimization or bootstrap twice.

## Evaluate a paper or a new estimator locally

This should be a first-class use case: **“I am reviewing a paper; where would
its estimator land on this benchmark?”** It should require a small adapter and
ordinary CLI commands, with no library-source edits, public registration, GitHub
account, issue, or website update.

### A small adapter, not a plugin framework

Accept a trusted local Python file and factory name:
`--estimator local_benchmark/paper_adapter.py:create_estimator`. The factory
receives `n`, `index`, `max_order`, `random_state`, and explicit JSON parameters,
and returns an object with the existing `approximate(budget, game)` interface.
Loading this file is ordinary local Python execution, not a sandbox.

For an estimator already implementing shapiq's interface, the adapter is just
the constructor call. Otherwise a small wrapper translates the paper's arguments
and converts its output to `InteractionValues`, declaring coalition-coordinate
keys, player count, target, and orders. Ship a short working example and explain
these two adaptation steps. No inheritance, entry-point registration, package
installation into shapiq, or changes to `__init__.py` should be necessary.

Pair it with a local JSON record containing a stable candidate ID, name,
supported targets/orders, minimum budget, method parameters, dependency versions,
paper/code reference if appropriate, and source provenance. Record both the
adapter hash and the external implementation's pinned source
commit/package version; hashing the adapter alone cannot identify the estimator.
Adapters must access coalition values through the supplied counted oracle, so initialization,
proxy fitting, and tuning obey the same budget rules. The truth object is only
used by the evaluator after estimation. Methods needing model internals use the
structured track instead. Validate the adapter on tiny known games first.

### Reuse the published benchmark snapshot

Release a **reproduction bundle** alongside every public dataset: suite manifest,
frozen game tables where distributable, truth, baseline per-run metrics, checksums,
and code/environment versions. For games that cannot be redistributed, include
construction instructions and source IDs; reconstructed games qualify for direct
comparison only when the required hashes match. An unavailable game remains
unavailable locally instead of silently becoming a different task.

The local workflow has five steps:

1. Download a chosen snapshot once, verify it, and install the candidate's
   dependencies in a local environment. Cached-oracle evaluation can then run
   offline; live games or a candidate with its own remote dependencies may differ.
2. Run adapter qualification and a dry run on a small selected subset.
3. Run only the candidate on that snapshot's tasks, targets, budgets, and replicate
   slots. Save its source/environment hashes and all failures privately.
4. Join candidate results to the frozen baseline run metrics, enforcing matching
   suite, task/truth hashes, target/order, metric protocol, budget cap, and timing
   mode. Different candidate dependencies are recorded explicitly; if they change
   game values or semantics, create a separate exploratory suite instead.
5. Generate the same static site into a local output directory, marked
   **“Local comparison — unpublished”**, with the candidate highlighted. Reuse
   the filters, charts, downloads, and methodology instead of maintaining a
   separate notebook/dashboard implementation.

Proposed commands, to implement with the rest of the runner:

```bash
# The downloaded, checksum-verified release bundle lives in local_benchmark/snapshot.
uv run python -m shapiq_benchmark.run --suite local_benchmark/snapshot/suite.json --artifacts local_benchmark/snapshot/artifacts --estimator local_benchmark/paper_adapter.py:create_estimator --estimator-config local_benchmark/paper.json --output local_benchmark/results --qualify --dry-run
uv run python -m shapiq_benchmark.run --suite local_benchmark/snapshot/suite.json --artifacts local_benchmark/snapshot/artifacts --estimator local_benchmark/paper_adapter.py:create_estimator --estimator-config local_benchmark/paper.json --output local_benchmark/results --resume
uv run python -m shapiq_benchmark.report --suite local_benchmark/snapshot/suite.json --input local_benchmark/results --baseline local_benchmark/snapshot/results --output local_benchmark/report --local --validate
uv run python -m http.server 8000 --bind 127.0.0.1 --directory local_benchmark/report
```

With `--estimator`, the runner schedules the candidate only; named built-in
methods can be selected explicitly for reruns. `report --local` copies the shared
site assets and writes its data under the supplied output path, never into the
public site. Add `local_benchmark/`, raw run outputs, and artifacts to `.gitignore`
in the implementation PR. No upload, telemetry, deployment, or automatic public
catalog insertion is part of these commands. A later public submission is a
separate, deliberate data-review PR. This matters for confidential paper reviews.

### What can be compared without rerunning baselines?

**Accuracy and query efficiency:** reuse baseline per-run results when the frozen
tasks and protocol match. Both the candidate and its subset coverage remain
clearly labelled. A partial paper-review run can show paired comparisons but is
not a full-suite rank. Use the same predeclared budget selection and weighting.

**Runtime:** public timings from another machine are not a fair local ranking.
Hide cross-hardware runtime ranks, retain separately labelled measurements, and
offer a runner option to rerun selected built-in baselines on the local machine
with the candidate's thread/timing policy. Record any changed software environment.

**Elo:** recompute the local batch fit from the paired outcomes for the candidate
and eligible incumbents on the selected fixed panel. Do not append a new score
to published ratings or treat public and local centered ratings as numerically
comparable. Local ratings carry their own panel/participant IDs and intervals;
only complete candidates enter the eligible local fit. Incomplete candidates
use a separate provisional fit or direct pairwise view. If release provenance is unknown or
confidential, omit the candidate from history while showing its measured results.

Acceptance test: a contributor outside the project can point to a small adapter,
evaluate a frozen smoke snapshot, reproduce baseline fixture scores, and open a
local candidate comparison. Mismatched truth/targets are rejected, interrupted
runs resume, runtime profiles stay separate, and public files remain untouched.

## What to run and how much it costs

### Stage the experimental work

| Suite | Proposed scope | Purpose |
| --- | --- | --- |
| Smoke | Tiny deterministic games, all registered methods/targets when dependencies permit, a few budgets, 2 seeds. | Catch wrong targets, hidden query use, malformed outputs, and broken adapters. Not published as performance evidence. |
| Pilot | SOUM/unanimity plus a few small tabular games, 3–5 budgets, 3 seeds. | Measure truth cost, per-method runtime/memory, dependency viability, and storage. |
| Core v1 | Qualified SV plus pairwise k-SII/SII/STII/FSII panels; synthetic and feasible tabular families; every compatible estimator qualified for these panels. | First public comparison, all five main views, explicit coverage for methods still blocked. |
| Extended releases | More game families, all datasets that pass qualification, higher orders, large structured games, expensive image/language/TabPFN/valuation cases, other indices. | Expand toward full catalog coverage without weakening ground-truth rules. |

Suggested pilot grids: `n ∈ {8, 12, 16}` for synthetic games; budgets
`{32, 128, 512, 2048}` intersected with valid caps for each task. Use `n=8` for
the first smoke cases, not as a claim that all defaults work at 32 queries.
For the core suite propose powers-of-two absolute caps and relative caps
`{1%, 2%, 5%, 10%, 20%, 50%, 100%}` where feasible. Deduplicate rounded caps.
Do not automatically run `2^n` for a large game; full enumeration is an optional
exact reference when affordable, and need not make every estimator exact.

Start production with 10 independent estimator seeds, then increase if pilot
uncertainty warrants it. Include multiple independent game/model seeds; many
estimator seeds on one model do not demonstrate generalization across games.
Fix all counts and panels before inspecting production rankings. Preserve an
independent development suite for adapters and tuning.

For feasibility, `2^16 = 65,536` coalitions but `2^24 = 16,777,216`. Even a table
of float64 values alone grows from about 0.5 MiB to 128 MiB, before coalition
masks and exact-computation intermediates. Some truth algorithms need much more
time/memory than the table. Use measured limits, not an unconditional n cutoff.

Estimate work **before launching a sweep**:

```text
number of runs = sum(valid task × method × target/order × budget × seed cells)
estimated wall time = preparation + truth + sum(pilot seconds for planned runs)
```

For illustration only, 100 tasks × 15 methods × 8 budgets × 10 seeds = 120,000
runs per target/order panel. At 1 second each that is about 33 serial hours;
at 30 seconds each it is about 1,000. These are arithmetic examples, not measured
cost forecasts. The dry-run report must show planned/skipped counts, estimated
CPU/GPU hours, memory, archive size, and the most expensive cells. Set an explicit
campaign resource cap after the pilot; budget-limited unfinished cells stay visible.

### Proposed commands after the implementation exists

These are the CLI contract to implement; **they do not run today**. Run from the
project root with Python 3.12+ and the committed lockfile. Install expensive extras
only for the corresponding method batches; capture their environment hashes.

```bash
uv sync --locked --extra benchmark
uv run python -m shapiq_benchmark.run --suite benchmark/suites/smoke.json --dry-run
uv run python -m shapiq_benchmark.prepare --suite benchmark/suites/smoke.json --output benchmark/artifacts
uv run python -m shapiq_benchmark.run --suite benchmark/suites/smoke.json --artifacts benchmark/artifacts --output benchmark/results --resume
uv run python -m shapiq_benchmark.report --input benchmark/results --suite benchmark/suites/smoke.json --output benchmark/site/data --validate
uv run python -m http.server 8000 --directory benchmark/site
```

Then replace the smoke suite with pilot/core after qualification. Install
`--extra sparse`, `--extra proxy`, `--extra shapleig`, or `--extra tree` in pinned
environments as needed. A full campaign runs on controlled research hardware;
GitHub Actions runs tiny validation checks and publishes finished data. Preserve
the existing C-extension build precautions in [AGENTS.md](../AGENTS.md) when
benchmarking source changes; stale native objects can invalidate timing/results.

## Hosting and publication

**GitHub Pages is a good first host** because it serves static HTML, CSS, and
JavaScript directly. The expected project URL is
`https://mmschlk.github.io/shapiq/benchmark/` if the site is deployed under a
`benchmark/` folder in that repository's Pages artifact. A separate repository
would have its own project URL. GitHub permits one Pages site per repository, so
check existing Pages settings and preserve any existing content before choosing
the artifact root. The current checkout has no Pages workflow; the API check
returned 404, which does not prove settings are available to this account.
[GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)
describes the hosting model and project URL conventions.

Proposed release procedure:

1. Run/validate the campaign, generate compact data, and attach immutable raw
   artifacts and checksums to a versioned benchmark release.
2. Open a data-update PR with the manifest diff, coverage changes, failures,
   validation report, and any changes to the headline rankings.
3. After review, the workflow stages only approved static assets, runs schema/link
   checks, uploads the Pages artifact, and deploys from the approved branch.
4. Keep old data snapshots addressable, show the suite/commit/date on the site,
   and roll back by deploying a previous approved snapshot.

Use the official `configure-pages`, `upload-pages-artifact`, and `deploy-pages`
actions, pinned to reviewed versions/commits. Deployment needs `pages: write`,
`id-token: write`, and the `github-pages` environment; forks/PRs validate without
deploying. A maintainer enables Pages with GitHub Actions as the source. See
[GitHub's workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
No settings or deployment are changed by this planning branch.

Use relative asset paths and URL query parameters so project subpaths work.
Target an initial compressed data payload below 2 MB, lazy-load target/order
shards, and test responsiveness at the projected core-suite size. Keep heavy raw
artifacts off Pages. Current published-site limits include 1 GB site size and a
100 GB/month soft bandwidth limit; these are ceilings, not payload targets.
See [GitHub Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits).
Archive only redistributable data/models; provenance links can point to upstream
sources when redistribution is restricted. Publish metrics independently of raw data.

## Small implementation steps

| Step / reviewable PR | Deliverable | Done when |
| --- | --- | --- |
| 1. Catalog and protocol | Explicit 22-method catalog, game/dataset discovery report, target adapters, date metadata, frozen smoke/pilot manifests. | Every discovered method/family has a status; aliases, dependencies, truth routes, and unsupported cells are documented. |
| 2. Truth and metrics | Game qualification, exact artifacts, nMSE, budget counter, validated metric semantics. | Tiny hand-computed games pass; structured truth agrees with enumeration; batching/zero/baseline/coordinate tests pass. |
| 3. Runner and pilot | Resumable jobs, accurate accounting, time/memory limits, failure records, pilot cost report. | Stop/resume does not duplicate or mix runs; actual budget and timing are auditable; core campaign fits an explicit resource budget. |
| 4. Aggregation and comparisons | Weighting, means/medians, coverage policy, paired matrix, preset batch ratings/intervals, historical frontier. | Known synthetic fixtures reproduce expected scores; shuffled result order changes nothing; missing/disconnected/degenerate cases are explicit. |
| 5. Static website | Filters, leaderboard, budget/time/history charts, head-to-head matrix, methodology, CSV/share links. | Browser results match Python fixtures; no stale ratings; keyboard/mobile/empty-state/base-path checks pass on a realistic data shard. |
| 5a. Local evaluation | Adapter example, reproduction bundle, candidate-only runs, private report using the shared site assets. | An external estimator works without source edits; matched baseline accuracy is reusable; incompatible tasks/timings are flagged; no public output is changed. |
| 6. Core campaign and Pages | Reviewed result release, publication workflow, public site. | All published numbers trace to raw runs; coverage and runtime provenance are visible; another person can reproduce a small published slice. |
| 7. Expansion | Remaining game families, datasets, targets, orders, optional methods, structured track. | Each addition passes the same qualification process and gets a versioned coverage update. |

Each implementation PR should have one purpose; separate metric/library fixes
from website changes. Use focused scientific tests, not snapshots of incidental
implementation details. Run the repository's required
`uv run pre-commit run --all-files` for each PR. The planning PR needs documentation
validation only; it must not claim the proposed experiment code has been tested.

### Project pitfalls to carry into AGENTS.md during implementation

This investigation found several issues that future agents should know: the
legacy game collection is deprecated but still imported by newer helpers;
estimator convenience registries are not exhaustive; reported estimator budgets
need independent measurement; cached games change the meaning of runtime; raw
`RandomGame` is batch-dependent; legacy product-kernel games are placeholders;
and existing metric semantics need validation. They are recorded here to respect
the requested one-file, plan-only branch. Copy the verified guidance into
`AGENTS.md` with the relevant implementation/fix PRs.

## Decisions for discussion

The proposal is concrete enough to implement, but these decisions deserve
agreement in the feature-request issue before committing substantial compute:

1. **Home and ownership:** keep the runner with the current in-repository helpers
   and host under shapiq Pages, or revive the separate benchmark repository? The
   recommendation is the current repository initially, with one dataset owner.
2. **First official panels:** SV plus pairwise Shapley interaction panels on
   synthetic and feasible tabular games; list the wider catalog immediately and
   expand systematically. Which game families should receive equal headline weight?
3. **Metric and failures:** accept energy-normalized MSE, mean plus median,
   family/stratum weighting, and provisional status for incomplete methods?
4. **Head-to-head:** accept direct win rates plus batch Elo-scale ratings for
   published presets, and the proposed tie tolerance?
5. **Compute and maintenance:** identify a reference machine, campaign resource
   cap after the pilot, and maintainers for result review and refreshes.
6. **History:** accept first public method dates and the retrospective label;
   audit the remaining methods and implementation dates before publication.

The initial site succeeds when someone can select a realistic task and budget,
understand why a method ranks well, see the evidence and its limits, and reproduce
the underlying comparison without learning a new experiment framework.
