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

Start with a working local comparison of three estimators, then publish a small
website with a leaderboard and error-versus-budget chart. Use real games from the
start; add interactions, controlled runtime comparisons, head-to-head ratings, and history in later
working increments. Researchers should also be able to add an estimator **locally**, evaluate
it on the same frozen suite, and open a private comparison report without editing
shapiq's source or publishing anything. Start with a small, trustworthy suite, then expand through an explicit
coverage checklist to every supported estimator and constructible game family.

**Initial scope: benchmark existing games and solvers.** Use trees for large-feature
SV and interaction panels, and unweighted KNN data valuation for large-training-set
SV panels. Product kernels are optional existing-game coverage. New SCM integration,
new game definitions, and new exact interaction algorithms are deferred research,
not requirements for this website. The longer-term designs below preserve the
research without making it a dependency of the first release.

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
- [Choosing representative games](#choosing-representative-games)
- [High-player games with exact ground truth](#high-player-games-with-exact-ground-truth)
- [Rules for a fair comparison](#rules-for-a-fair-comparison)
- [Release history and the progress chart](#release-history-and-the-progress-chart)
- [The code to write](#the-code-to-write)
- [Evaluate a paper or a new estimator locally](#evaluate-a-paper-or-a-new-estimator-locally)
- [What to run and how much it costs](#what-to-run-and-how-much-it-costs)
- [Standardizing CPU measurements on Hopper](#standardizing-cpu-measurements-on-hopper)
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
The **games themselves** can still be in the main generic-estimator benchmark:
only the evaluator gets model structure for truth; estimators get the counted
coalition-value callable. A structured solver's privileged access is what requires
a separate competitor track.

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

## Choosing representative games

**Use real workloads for the headline results.** SOUM, unanimity, additive, dummy,
and known-interaction games belong in diagnostics and receive no weight in the
default real-workload score. Choose game mechanisms as well as datasets: changing
from features to training-data groups or ensemble members changes the task an
estimator must solve. The original [shapiq study](https://arxiv.org/abs/2410.01649)
provides a useful cross-application starting point.

| Panel | Players and coalition value | Starting design |
| --- | --- | --- |
| Local prediction explanation | Raw features; average prediction with present features fixed to the explained point and missing features supplied by fixed background rows. | California Housing (8 features), Adult Census (14), then Bike Sharing (12); boosted trees and a small MLP; held-out explanation points. |
| Grouped-data valuation | Substantial groups of training rows; held-out utility after a clean model fit on their union. | Two real datasets, initially 8 groups; inexpensive regularized learners, then nonlinear learners; multiple frozen partitions. |
| Ensemble utility | Frozen fitted models; held-out utility of the coalition's combined predictions. | Eight varied predictors on each of two real datasets; explicitly choose averaging/soft probabilities or hard voting. |
| Global predictive importance and feature selection | Features; labelled predictive loss after either masking a fixed model or retraining on selected features. | Add as separate games after defining loss, empty-coalition predictor, and deterministic fitting/evaluation. |
| Text and image explanation | Tokens or image regions; a specified prediction under a specified masking/removal rule. | Early expansion with actual inputs, pinned pretrained models, explicit grouping and output scale. |

A possible later representative pilot has **28 game instances**: 20 local
explanations (two datasets × two model families × five held-out points), four
grouped-data games (two datasets × two partitions), and four ensemble games
(two datasets × two constructions). This is a scoped pilot, not evidence covering
all applications, and is not a release prerequisite. The first release prioritizes
existing local/tree and KNN games; grouped-data and ensemble panels can follow.
Use the same frozen games for SV and qualified interaction
targets. Add the high-player panels below before claiming dimensional scalability.

Fit models and preprocessing on training data, select hyperparameters using
predictive validation, and explain held-out points chosen by a predeclared rule.
Do not select cases after seeing estimator rankings. Keep a raw categorical
feature's encoded columns together as one player, with a matching truth method.
Report predictive quality and actual player counts. Preserve full native feature
sets where feasible rather than trimming every problem to fit enumeration.

Freeze background **rows** jointly, not independently shuffled columns, for the
initial marginal game. Truth is exact for that empirical background distribution.
Conditional replacement, path-dependent trees, and causal interventions are
separate semantic panels. Probability and margin outputs are also distinct games.

Grouped-data games should contain meaningful amounts of training data; training
on subsets of eight individual rows is a small-data diagnostic, not a substitute.
Label random partitions as simulated sources; use source/time/geography groups
only when supported by the data. Define empty/single-class fits and fresh model
initialization explicitly. Each eight-group game needs 256 fits for enumeration.

Two existing helpers need care: `GlobalExplanation` compares masked predictions
with the model's own full predictions, samples rows as calls advance, and is not
a deterministic supervised SAGE game. A labelled-loss adapter should follow an
explicit [predictive-power definition](https://arxiv.org/abs/2004.00668).
The legacy ensemble classifier uses hard voting; soft-probability/log-loss utility
needs a distinct adapter. Preserve those differences in names and metadata.

Report per-family results first. An overall view is an explicit mixture of these
panels, not a claim about every real-world use of Shapley values. Many explanation
points from one fitted model must not outweigh an entire other application.

## High-player games with exact ground truth

**Reuse existing games and structured truth solvers, while keeping estimator access
generic.** Evaluate the same fitted model through two interfaces:
`game(coalitions)` for counted estimator queries and a private exact-truth adapter
for the evaluator. Do not enumerate or store `2^n` coalition values for these games.
Persist the model/game recipe and exact coefficients instead. No estimator gets
the source model, graph, truth, or a table of equivalence classes for free.

### Existing routes and deferred research

| Route | Available foundation | First admitted targets | Scaling gate |
| --- | --- | --- | --- |
| Real fitted trees | Current `InterventionalTreeSHAPIQ` and path-dependent `QuadratureTreeSHAP`. | SV and pairwise SII/k-SII first. Interventional STII/FSII and further supported targets after separate checks. | Tree count/depth, distinct path features, background size, and output support. |
| KNN training-data valuation | Current `KNNExplainerGame` and `KNNExplainer`. | SV only; include in the initial high-player scope. | Training-set size, feature dimension, distance computation, and estimator runtime. |
| Real fitted product-kernel models | Current `ProductKernelGame` and `ProductKernelExplainer`. | SV now; interactions require a qualified extension. | Player count, support-vector count, numerical stability, and truth runtime. |
| Causal intervention games — deferred | External [`exactdoshap`](https://github.com/rtealwitter/exactdoshap) class-based solver; integration is outside initial scope. | Future SV, then a separately implemented and validated SII/k-SII extension. | Number of intervention-equivalence classes, graph/oracle cost, and output size. |

### KNN: existing high-player data valuation

Reuse [`KNNExplainerGame`](../src/shapiq/explainer/nn/games/knn.py) with
[`KNNExplainer`](../src/shapiq/explainer/nn/knn.py). Players are training examples,
not input features. For a fixed held-out point and class, the coalition utility
is the number of matching labels among its at most `k` nearest members, divided
by the fixed `k`; preserve this rule even for coalitions smaller than `k`.
The exact SV algorithm uses a distance ordering and recurrence, with `O(n log n)`
sorting/recurrence cost plus the distance computation. It does not enumerate `2^n`.

Start with one real classification dataset and a small enumerated counterpart.
Then pilot frozen training subsets of 128, 512, and 1,024 rows before considering
larger sets. These counts are proposed qualification points, not measured capacity.
Freeze preprocessing, distance metric, `k`, training rows and ordering, held-out
points, class indices, and tie behavior. Check game/solver agreement coordinatewise
on the small case. Keep this data-valuation panel separate from feature attribution;
the current nearest-neighbor explainers support SV only, not interaction truth.

Existing threshold-neighbor and weighted-neighbor games are optional later coverage.
Qualify their precise utilities, empty-coalition baselines, numerical behavior,
and, for weighted KNN, discretized weights before including them. Unweighted KNN
is sufficient for the first large-training-set panel; no new valuation game or
exact algorithm is needed.

### Trees: the first high-player implementation

Train tree ensembles on naturally wider real datasets. Candidates already have
loaders: Breast Cancer, Ionosphere, Communities and Crime, Arrhythmia, Bioresponse,
MicroMass, and suitable TabArena tasks. Audit actual dimensions after preprocessing;
for example, Ionosphere drops constant columns. Select datasets and predictive
models before seeing estimation errors. Use moderate and deeper fitted ensembles,
not only shallow trees chosen because their interactions are easy to compute.

For the primary interventional panel use
`v(S) = mean_b f(x_S, b_notS)` with frozen background rows. Reuse
[`InterventionalGame`](../src/shapiq/tree/interventional/game.py) only after its
output agrees with the exact solver. Regressors are the simplest first route;
sklearn random-forest probabilities and explicit XGBoost/LightGBM margins are
additional qualified choices. Taking a sigmoid of raw-score Shapley values does
not produce probability Shapley values. Do not assume converters and game wrappers
use the same classification output automatically.

The separate path-dependent panel averages missing branches by training cover.
Its exact computer supports SV/SII/k-SII/BV/BII; it must not supply truth for the
empirical-background game. Current `PathdependentComputer` passes target/order
through an explanation call whose path-dependent handler ignores those arguments.
Construct the solver with target/order and validate the returned metadata instead;
fix that helper in its own implementation PR.

Pilot native player counts in **17–64, 65–128, and 129–256** bands, then a
**257–1,024** stretch band. These are qualification targets, not promised coverage.
Record total players, features actually used by the model, path lengths, and
nonzero interaction counts. Naturally unused features remain visible; do not pad
small games with dummy features to advertise high dimension. Keep a shallow/sparse
case's scope visible even when its nominal player count is large.

### Product kernels: different models and a different removal rule

Start with fitted RBF SVR or binary SVC on real wide datasets. The current game is
`v(S) = intercept + sum_i alpha_i * product_{j in S} k_j(X_ij, x_j)`:
removing a feature replaces its kernel factor with 1. This is a useful, explicit
product-kernel task, not marginal feature imputation. Binary SVC uses its decision
score. Use the same feature preprocessing and fixed kernel parameters in the
callable and truth solver. See the [PKeX-Shapley work](https://arxiv.org/abs/2505.16516).

The checked implementation is SV-only and recomputes symmetric polynomials per
feature, with roughly `O(m*n^3)` work for `m` support/training points. Begin with
moderate native dimensions and vary `m` as a separate resource axis. High dimension
is not automatically cheap just because the algorithm is polynomial.
[PR #597](https://github.com/mmschlk/shapiq/pull/597), open at this audit, proposes
a faster quadrature implementation and SII/k-SII/BII/BV support. Qualify that
implementation after review, or implement the narrowly needed adapter separately;
do not depend on unmerged code silently. For truth, use the quadrature rule's
degree-based exact node count, never its optional reduced-node approximation.

Audit coordinatewise numerical accuracy at the intended dimensions, including
underflow/cancellation and factorial weights. Efficiency alone can pass with wrong
individual values. Gaussian-process support can follow after checking conversion
restrictions; the current bare-RBF converter needs particular care with anisotropic
kernels and target normalization. SVR/SVC keeps the first implementation smaller.

### SCMs: deferred research, outside the first release

The following design is retained for future work. It is not an implementation
phase or prerequisite for the benchmark website.

An analytic SCM coalition function does **not** by itself remove the exponential
Shapley summation. The relevant reuse is
[`exactdoshap` at the audited commit](https://github.com/rtealwitter/exactdoshap/tree/7fdd7c20737136df6ded3f74d150f2dddc14a688):
its `AllClasses` solver groups interventions into equivalence classes. Its
[paper](https://arxiv.org/html/2602.07203v1) gives cost `O(r*(n+e+T))`, with `r`
classes, `e` graph edges, and coalition-query cost `T`. The class count can still
be exponential; sparse edges alone are not a guarantee of affordable truth.

Implement a small adapter that freezes graph, structural mechanisms, target,
intervention point, player ordering, and exogenous-noise distribution. The existing
learned nonlinear `realDoGame` is a useful foundation, but currently limits games
to 5–20 players and 40 edges. Replace its random selection/retry behavior with
explicit manifests and new high-player factories. Do not just remove the limits
and launch an unbounded class enumeration.
Qualify a pinned optional causal environment, including causal-learn and the
mechanism-model dependencies. Persist the fitted graph/mechanisms and noise bank
so local candidate evaluation does not need to rerun causal discovery.

Qualify two clearly labelled sources: domain-motivated/learned graphs with fitted
mechanisms, and controlled structural stress graphs. Target 32, 64, then 128
active ancestor players where viable. Record excluded non-ancestors, graph origin,
depth and branching, and measured class count. Stop truth preparation at a fixed
time/memory/class cap and report the blocked cases; truncated enumeration never
becomes ground truth. Selecting tractable graphs is a disclosed coverage limit.

Distinguish **exact population-SCM truth**, requiring analytic expectations or
exact discrete inference, from **exact empirical-SCM truth**, where a frozen finite
noise bank defines the game. The current nonlinear learned-game path uses the
latter. Reuse identical noise samples for every coalition and estimator, and
verify equivalence-class invariance numerically. This yields exact attributions
of the specified empirical game, not of an unknown population causal process.
For learned graphs, exact computation also does not certify the fitted graph as
the true causal structure of the real data.

Current `AllClasses` returns singleton values. The paper derives interaction
extensions, but its experiment code still uses exhaustive computation for them.
Implement SII class-weight contraction and k-SII aggregation, validate against
small enumerated graphs, then admit pairwise SCM interactions. STII/FSII require
their own derivation/implementation before being claimed. Linear causal models
can already have blocking/redundancy interactions; they are useful validation
cases, but add nonlinear mechanisms for substantive workload coverage.

The existing shapiq confounding games are a different family.
[PR #600](https://github.com/mmschlk/shapiq/pull/600), also open at this audit,
adds analytic linear-Gaussian coalition values, not a scalable exact Shapley solver.
Neither should be mistaken for an already integrated high-player do-Shapley route.

### Shared admission tests, budgets, and implementation sequence

Each new truth adapter must pass deterministic batch/order checks, empty/full-game
prediction checks, and **coordinatewise** comparison with enumeration on small
counterparts. Check target/order, semantic/output scale, player mapping, structural
zeros, and efficiency where the chosen index requires it. At large dimensions,
check finite values, numerical residuals, and selected cases against an independent
solver or higher-precision calculation when available. Save truth diagnostics,
method/version, preparation time, peak memory, and hashes with the artifact.

Start with SV and pairwise interactions. At 256 players there are 32,640 pairs;
at 1,024 there are 523,776. Sparse tree truth does not make a competing estimator's
dense output small. Score every logical coordinate, including predictions outside
the nonzero truth support; do not evaluate only a convenient top-k set. Use sparse
norm/dot-product reductions where valid, and admit higher orders only after sizing
both truth and estimator outputs.

For high-player games use absolute query caps and **queries per player (`B/n`)**,
with an initial candidate grid such as `B/n ∈ {2, 4, 8, 16, 32, 64}` plus shared
absolute caps. Enforce method-specific minima and a measured campaign resource
cap. Fraction-of-all-coalitions is a secondary annotation here, not a practical
default slider. Keep results separate by player band, game semantics, target,
and model family; do not let cheap small games determine the large-game rank.

Start with explicit tree and unweighted-KNN truth adapters in the existing benchmark
machinery; add product-kernel SV only as optional coverage. `prepare.py` calls them without `Game.precompute()`
on the large game, and writes ordinary exact-value artifacts. The runner and
local-candidate workflow stay the same. Game/truth objects must remain separate
so caches containing truth-preparation evaluations are not available to estimators.
Count coalition requests consistently even when the oracle shares work internally.

Deliver in this order: **(1)** one real high-player tree game plus its small
validation counterpart; **(2)** a small tree panel spanning player bands and SV/
pairwise targets; **(3)** unweighted-KNN SV on increasing training-set sizes;
**(4)** optionally product-kernel SV using the existing solver. Run a cost pilot for each before
expanding. Keep synthetic structural stress tests separately weighted from real
fitted-model or learned/domain-based SCM panels. No full high-player campaign is
executed by this planning document.

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
The proposed [Hopper reference profile](#standardizing-cpu-measurements-on-hopper)
makes these rules concrete. Parallel accuracy runs may record diagnostic runtime,
but that runtime does not enter the controlled timing leaderboard.

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
| `benchmark/hardware/hopper-epyc9754-1c-v1.json` and a small Slurm launcher | Reference CPU, allocation/thread policy, environment and runtime checks. | Slurm binding, system metadata, existing Python environment. |
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
| Smoke | Begin with three SV methods and tiny deterministic games; extend qualification cases as each method/target is added. | Catch wrong targets, hidden query use, malformed outputs, and broken adapters. Not published as performance evidence. |
| Pilot | One small real local game, then one high-player tree and one unweighted-KNN candidate; 3 budgets and 3 seeds initially. Keep synthetic correctness checks separate. | Measure truth cost, per-method runtime/memory, dependency viability, and storage. |
| Core v1 | Local/tree SV and pairwise k-SII, plus high-player KNN SV. Product-kernel SV is optional. | First substantive comparison after the small public preview; explicit measured/planned/blocked coverage. No requirement to finish all views or methods before publishing. |
| Extended releases | Grouped-data and ensemble panels, other existing families/datasets/targets as exact truth permits. New SCM integration and exact algorithms require separate research scope. | Expand coverage without making new game development a website dependency. |

Use native feature counts for the real-data pilot, and the declared group/member
counts for valuation/ensemble games. Start with budget caps
`{32, 128, 512, 2048}` intersected with valid caps for each small task. Tiny
synthetic smoke fixtures do not contribute to headline performance. Use the
absolute/`B/n` grids above for high-player games.
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

## Standardizing CPU measurements on Hopper

**Recommendation: use an AMD EPYC 9754 compute node, one physical core and one
software thread per estimator, as the initial runtime reference.** Call the
proposed profile `hopper-epyc9754-1c-v1`. This is a profile to implement and qualify,
not a claim that performance measurements have already been taken.

### Hardware actually inspected

On 2026-09-29, `lscpu` on the current `hopper.cluster` host reported two AMD EPYC
9124 sockets, 16 cores per socket, and one hardware thread per core. That host
is absent from the Slurm compute-node list, and the current shell has no Slurm
allocation. Do not use its CPU identity as the identity of cluster jobs.

Three brief, single-CPU Slurm metadata tasks verified the following compute nodes:

| Node | CPU | Topology reported inside the allocation | Other observations |
| --- | --- | --- | --- |
| `gpu04` | AMD EPYC 9754 128-Core Processor | 1 socket, 128 cores, 1 hardware thread/core, 1 NUMA node | The metadata task used only a CPU, with affinity restricted to one core. |
| `himem01` | AMD EPYC 9754 128-Core Processor | 1 socket, 128 cores, 1 hardware thread/core, 1 NUMA node | CPU-only node; `performance` governor, boost enabled; one-core task affinity verified. |
| `himem02` | AMD EPYC 9754 128-Core Processor | 1 socket, 128 cores, 1 hardware thread/core, 1 NUMA node | CPU-only node; `performance` governor, boost enabled; one-core task affinity verified. |

These were hardware inspections, not benchmarks. Other cluster nodes were not
hardware-qualified. Slurm currently advertises no CPU feature labels on these
nodes, so do not invent an `--constraint=epyc9754` selector. Initially choose one
verified node explicitly and recheck the model inside every job. Prefer
`himem01` for the reference campaign; qualify `himem02` separately before pooling
its timings. The same CPU model alone does not establish equivalent performance.

### The reference profile

| Setting | Proposed rule |
| --- | --- |
| CPU/device | AMD EPYC 9754; CPU execution only; record actual hostname, family/model/stepping and microcode. |
| Resources used by each estimator | One process using one pinned physical core; one software thread in numerical libraries and model backends. |
| Memory | Start the pilot with a 16 GiB job limit; record peak memory. Increase only with a new declared profile if the pilot demonstrates a need. |
| Isolation | Official timings use an exclusive CPU-only node allocation or an administrator-arranged equivalent quiet window, one timed run at a time. Shared-node timings remain exploratory. |
| Frequency policy | Record governor and boost state; retain the observed `performance` governor with boost enabled. Do not claim a fixed GHz or change host-wide settings. |
| Environment | Pin shapiq commit, Python/dependency versions, native build, BLAS/OpenMP implementation, OS/kernel, and thread settings. |
| Timing | Exclude queue/environment setup/import/data-load/truth costs from estimation time; report relevant setup separately. Include estimator initialization and execution, including proxy fitting. |
| Repeats | Fixed warm-up policy, reproducibly shuffled method order, repeated measured runs and median/spread; keep all planned repetitions and failures. |
| Modes | Cached oracle first; live-game timings are a separate result group under the same hardware profile. |

Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`BLIS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to 1 before importing numerical
packages. Set estimator/backend options such as `n_jobs`, `nthread`, and PyTorch
thread counts explicitly as applicable. Inspect loaded native pools with
[`threadpoolctl`](https://github.com/joblib/threadpoolctl); environment variables
alone do not establish that every library obeyed the policy. Verify process
affinity inside the worker. Record effective settings, not just requested ones.

The shell used for this inspection already had several thread variables set to
1, which is why `nproc` printed 1 despite affinity covering all 32 host CPUs.
Thread configuration is not a CPU reservation. Use Slurm allocation and binding
for resource ownership; record `os.sched_getaffinity(0)` and hardware topology
rather than treating `nproc` as the machine's physical core count.

The eventual Slurm launcher should request one node, one task, one CPU per task,
and bind the worker with `srun --cpu-bind=cores`; its worker preflight checks the
CPU model, effective affinity, thread pools, and environment hash. Official
timing requests additionally use `sbatch --exclusive` on the selected CPU-only
node when cluster policy permits. **Exclusive allocation reserves the whole
node**, even though the estimator uses one core, so reserve short timing campaigns
after the pilot rather than doing all development this way. See Slurm's
[CPU binding guide](https://slurm.schedmd.com/cpu_management.html) and
[exclusive-allocation documentation](https://slurm.schedmd.com/sbatch.html#OPT_exclusive).

Use normal shared Slurm allocations to prepare truth and run independent accuracy
jobs in parallel, subject to a campaign concurrency cap. Rerun the selected
budget points sequentially in the controlled timing allocation; do not reuse
contended sweep runtimes as official seconds. Time-curve errors must come from
those same timed runs, not be joined to whichever accuracy repeat looked best.

Run a short fixed calibration workload before and after timing batches to detect
load/environment drift. Establish a tolerance in the pilot, then freeze it before
production. Quarantine an entire failed batch under that rule, record the reason,
and rerun it; never remove individual slow observations after seeing rankings.
Calibration is a diagnostic, not a multiplier that converts another machine's
seconds into “Hopper-equivalent” seconds.

Start with this one-core profile because it is easy to audit. A later eight-core
or GPU profile can capture methods that benefit from parallelism, but has its own
rankings. Accuracy comparisons remain usable on a researcher's local hardware
when task/protocol hashes match; their runtime ranks require locally rerun baselines.

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

Build in **working increments**. The complete catalog and all advanced statistics
are the destination; they are not prerequisites for the first useful comparison.
Implement only the corresponding parts of the proposed file layout at each phase.

| Phase | Smallest useful deliverable | Done when |
| --- | --- | --- |
| 1. Working local benchmark | Three SV estimators on real fitted-model games, exact truth, measured budgets, nMSE, and JSON/CSV results. Include a minimal local-estimator adapter and separate synthetic correctness fixtures. | One command produces reproducible real-workload comparisons; another researcher can add a candidate without editing shapiq source. |
| 2. Minimal website and private report | One static page with a table, error-versus-budget chart, method/budget/game-instance filters, and download. Deploy a small public preview on Pages; use the same assets for a local candidate report. | Every displayed number traces to a real run; the page works locally and under the Pages subpath. It visibly identifies the small preview suite. |
| 3. Existing high-player games and interactions | Add qualified high-player tree SV/pairwise k-SII and unweighted-KNN SV, repeat/coverage summaries, and controlled Hopper timing. | Small truth counterparts match enumeration; large-game exact truth avoids powerset enumeration; per-family/player-band budget and time curves are reproducible. |
| 4. Broader coverage and reliable campaigns | Add remaining compatible estimators and existing game families/targets, optionally product-kernel SV, resumable campaigns, and resource limits in small batches. | Each addition passes qualification and gets a versioned coverage update; interrupted campaigns resume without mixing snapshots. No new exact algorithm is required. |
| 5. Rich comparisons and history | Add paired win rates first, then preset Elo/uncertainty and release-date frontiers, followed by richer presets and overall regime summaries. | Shared-panel fixtures validate every summary; dates have sources; incomplete methods cannot improve official ranks. |

**Concrete Phase 1 proposal:** `KernelSHAP`, `PermutationSamplingSV`, and `SVARM`;
SV only; California Housing and Adult Census, two qualified model families
(boosted trees and MLP), and five frozen held-out points per model: 20 games.
With three budget caps and three estimator seeds this gives **540 planned runs**,
plus separate tiny synthetic correctness checks. The first acceptance milestone is
one California model and point, three budgets, and three seeds: **27 planned runs**.
Expand to the 20-game manifest only after this works; it does not block Phase 2.
Obtain truth by enumeration of the
fixed empirical-background game, cross-checking a matched tree route where possible.
Save the manifest, code/source hashes, measured query counts, per-run statuses,
and coefficient-aligned MSE/nMSE. This supersedes the earlier SOUM-first proposal.
Use simple serial execution and a handful of explicit constructor adapters. Log
runtime as diagnostic data initially; do not present it as qualified CPU timing.
The count is a proposed test matrix, not an assertion that these runs have executed.

Phase 1's local adapter is deliberately small: an external factory plus the same
counted oracle and result contract. Phase 2 adds the downloadable reproduction
bundle and private interactive report. This makes the paper-review workflow
available early instead of waiting for the whole benchmark catalog.

In Phase 3, begin interactions with `KernelSHAPIQ`, `SHAPIQ`, and `SVARMIQ` on the
same pairwise k-SII tasks. Introduce the other interaction definitions as separate
panels later. Qualify the first high-player tree and KNN panels after measuring
truth/estimation cost; the 28-game pilot is an optional subsequent expansion.
Set the campaign cap before expanding;
do not advertise dimensional scalability from the small enumerable games alone.

Phase 5 can start after Phase 3 while Phase 4 grows the catalog: history and
head-to-head views do not require every game to be finished. Each phase leaves
a usable artifact. No “all estimators/all games” milestone blocks an initial
local tool or public preview, and preview results are never labelled comprehensive.

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
and existing metric semantics need validation. `GlobalExplanation` is a stochastic
model-fidelity game rather than supervised held-out loss. `PathdependentComputer`
can ignore requested interaction settings through its explanation-call route;
initialize the exact solver with the target/order instead. Product-kernel exact
SV does not imply implemented interactions, and an analytic SCM value function
does not imply scalable exact Shapley computation. Hopper's current shell host and
Slurm compute nodes have different CPU models, and thread-limited `nproc` output
does not establish an allocation. They are recorded here to respect
the requested one-file, plan-only branch. Copy the verified guidance into
`AGENTS.md` with the relevant implementation/fix PRs.

## Decisions for discussion

The proposal is concrete enough to implement, but these decisions deserve
agreement in the feature-request issue before committing substantial compute:

1. **Home and ownership:** keep the runner with the current in-repository helpers
   and host under shapiq Pages, or revive the separate benchmark repository? The
   recommendation is the current repository initially, with one dataset owner.
2. **First official panels:** existing local/tree explanation games, including
   high-player trees, plus unweighted-KNN data valuation. Product-kernel SV and
   other existing families can follow; SCM integration and new algorithms are
   deferred. Synthetic diagnostics have no default headline weight.
   Which real-workload families and player bands should receive equal weight?
3. **Metric and failures:** accept energy-normalized MSE, mean plus median,
   family/stratum weighting, and provisional status for incomplete methods?
4. **Head-to-head:** accept direct win rates plus batch Elo-scale ratings for
   published presets, and the proposed tie tolerance?
5. **Compute and maintenance:** qualify the proposed EPYC 9754 one-core Hopper
   profile and a short exclusive timing window; set the campaign resource cap
   after the pilot, and name maintainers for result review and refreshes.
6. **History:** accept first public method dates and the retrospective label;
   audit the remaining methods and implementation dates before publication.

The initial site succeeds when someone can select a realistic task and budget,
understand why a method ranks well, see the evidence and its limits, and reproduce
the underlying comparison without learning a new experiment framework.
