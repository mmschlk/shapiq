# Compact Shapley estimator benchmark — scope version 2

This is the active scientific plan. Version 2 replaces the historical broad
cohort with one compact comparison across **all 24 datasets**, **22 estimator
configurations**, three applications, four game-construction seeds and three
estimator seeds. The executable recipe list is
[suites/comprehensive.csv](suites/comprehensive.csv), generated and validated by
`shapiq_benchmark.comprehensive`.

**512 recipes × four construction seeds = 2,048 prepared instances**: 477
core recipes plus 35 KNN/TNN recipes with 32 or 64 training-row players. Metadata
checks verify existing exact references for all selected instances; this is not
100% successful estimator coverage or a published replacement report. No new
preparation is required.

Thirty-five core recipes were excluded in full because at least one of their four
construction seeds failed the model-quality check against the dummy predictor.
All four seeds of each retained recipe remain together. Neighbor replacements
were selected for balanced coverage without consulting estimator accuracy.
The public [selection manifest](suites/compact-selection.json) records the exact
retained and excluded recipes and the deterministic balancing rule.

Historical broader results outside this selection do not count toward version 2
coverage. Keep the existing published report until the replacement passes
validation; deploy its About page and data together.

## Fixed scope

| Application | Construction and models | Retained player grids | Recipes |
| --- | --- | --- | ---: |
| Individual prediction | Baseline masking; linear/logistic regression and random forest | Up to 12 features | 127 |
| Individual prediction | Path-dependent and interventional random-forest trees | 8, 16, 32 features | 88 |
| Data valuation | Training-row groups; linear/logistic regression and random forest | 4, 8, 12 groups | 135 |
| Data valuation | Selected KNN and TNN games | 32, 64 training rows | 35 |
| Feature selection | Retraining linear/logistic regression and random forest | Up to 12 features | 127 |

Valuation models use **exactly `min(12, dataset width)` input features**. Input
width is distinct from the number of training-row groups. There is no width
sweep or special-model contrast. Selected native KNN/TNN games provide exact SV
on classification datasets. Nested feature selections, training/held-out
splits, model profiles and construction seeds remain explicit recipe inputs.

The datasets are Adult Census, Mushroom, Ionosphere, Bioresponse, Breast Cancer,
Digits, Wine classification, Amazon Employee Access, APS Failure, Anneal,
Splice, Credit Card Default, Wine Quality, Communities and Crime, NHANES I,
QSAR-TID11, Miami Housing, Superconductivity, California Housing, Diabetes,
Bike Sharing, Airfoil Self Noise, Concrete Strength and Protein. Generic grids start at 4, 8 and 12 features where feasible, with natural full
widths below 12; whole-recipe quality exclusions then determine retained grids.
Actual shipped loader widths determine feasibility. Wine classification and Wine Quality
regression are different datasets; NHANES I uses its supplied survival label as
a regression surrogate. [About](site/about.md) lists each dataset's player grids.

## References and evaluation

- Generic games enumerate all coalitions, with **at most 12 players**. Native
  random-forest tree and KNN/TNN games use their matching exact solvers. Preserve recorded
  floating-point qualification and finite-endpoint FSII reference-floor policies;
  never replace a missing exact reference with an approximate estimate.
- Targets are **SV and order-two k-SII, SII, STII, FSII and FBII**, where supported.
  KNN/TNN supply SV only. Constructor-supported targets determine the capability catalog; unsupported
  method/target combinations are not failures.
- Relative budgets are **0.5, 1, 2, 4, 8, 16, 32 × players**, with the existing
  query-counting and finite-game budget rules. No 64× or 128× budget belongs to
  version 2. Retain all 22 estimator configurations and their explicit parameters.
- Use four construction seeds and three estimator seeds. Evaluate all methods on
  the same frozen game artifacts and references; do not reconstruct or retune a
  game for an estimator. Preserve random-state and isolated global-RNG policies.
- Score nonempty coefficients using the frozen reference and recorded normalized
  squared error. Preserve per-order eligibility, zero-signal and model-quality
  exclusions. Report reference coverage separately from estimator success.
- Weight applications equally, then subtype and recipe. This is not equal weight
  per dataset. Collapse exact table aliases only within the same application,
  subtype, recipe and quality role, using deterministic construction order;
  retain distinct weighted recipes even when their payoff tables coincide.

## Completion and publication

The goal is **100% successful supported coverage for every estimator except
ShaplEIG and SPEX**. ProxySPEX is included. Unexpected failures in other methods
require diagnosis and recovery; an explicit failed outcome alone is not success.
Show failures, exclusions and unattempted cells honestly. Do not alter games,
settings, truth or denominators to make coverage appear complete.

Reuse historical results only when the exact version 2 recipe, artifacts,
settings, budget, seed, source and ownership match. Historical outcomes outside
this scope must not enter the replacement report. Obsolete artifacts may be
deleted after required retained-result provenance is safely materialized and
verified. One consolidated reference, ownership, numerical and coverage check precedes
publication of the report and its reproducibility artifacts.

## Estimator-combination analysis

Fit signed linear weights without an intercept separately for each target and
budget, using complete same-game/budget/seed panels of supported nonexempt methods.
Dataset-grouped cross-validation keeps all games from a dataset together.
Held-out normalized R² and nMSE compare the fitted combination with equal weights;
PCA of normalized estimator errors describes shared error patterns. Coefficients
are not importance scores. Publish this analysis only with its authenticated,
report-matched artifact; the method description makes no performance claim.

## Execution

Use CPU-only nodes and at most **1,000 concurrent CPUs**, including preparation,
recovery and validation. The total CPU-hour cap was waived; record actual
allocation costs. Campaign `WAKEUP.md`, state and accounting remain the operational
record. This plan does not authorize reviving superseded jobs.

Freeze source, dependencies and dataset caches before parallel work. Use durable
attempt intents and atomic shared claims. Preserve successful attempts and
explicitly authenticate replacements. Scheduler completion alone does not prove
that a worker finished its planned cells.

[settings.json](settings.json) separates startup, estimation and output limits.
Memory policies and qualified source overlays are recorded operational inputs;
never install into an environment used by running jobs. See [README.md](README.md)
for portable preparation, execution and replay commands. Timings on shared CPUs
are diagnostic, not isolated hardware comparisons.
