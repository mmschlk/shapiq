# Compact Shapley estimator benchmark — scope version 2

This is the active scientific plan. Version 2 replaces the historical broad
cohort with one compact comparison across **all 24 datasets**, **22 estimator
configurations**, three applications, four game-construction seeds and three
estimator seeds. The executable recipe list is
[suites/comprehensive.csv](suites/comprehensive.csv), generated and validated by
`shapiq_benchmark.comprehensive`.

**512 recipes × four construction seeds = 2,048 intended instances**, before
qualification. These are planned counts, not measured coverage. The historical
5,080-instance design and its recovery campaigns are not version 2 coverage.
Keep the existing published report until the replacement passes validation;
deploy its About page and data together.

## Fixed scope

| Application | Construction and models | Players | Recipes |
| --- | --- | --- | ---: |
| Individual prediction | Baseline masking; linear/logistic regression and random forest | 4, 8, 12 where feasible, plus full widths below 12 | 138 |
| Individual prediction | Path-dependent and interventional random-forest tree games | 8, 16, 32 where feasible | 92 |
| Data valuation | Groups of training rows; linear/logistic regression and random forest | 4, 8, 12 groups | 144 |
| Feature selection | Retraining linear/logistic regression and random forest | 4, 8, 12 where feasible, plus full widths below 12 | 138 |

Valuation models use **exactly `min(12, dataset width)` input features**. Input
width is distinct from the number of training-row groups. There is no width
sweep, neighbor-game branch or special-model contrast. Nested feature selections, training/held-out
splits, model profiles and construction seeds remain explicit recipe inputs.

The datasets are Adult Census, Mushroom, Ionosphere, Bioresponse, Breast Cancer,
Digits, Wine classification, Amazon Employee Access, APS Failure, Anneal,
Splice, Credit Card Default, Wine Quality, Communities and Crime, NHANES I,
QSAR-TID11, Miami Housing, Superconductivity, California Housing, Diabetes,
Bike Sharing, Airfoil Self Noise, Concrete Strength and Protein. Actual shipped
loader widths determine feasible grids. Wine classification and Wine Quality
regression are different datasets; NHANES I uses its supplied survival label as
a regression surrogate. [About](site/about.md) lists each dataset's player grids.

## References and evaluation

- Generic games enumerate all coalitions, with **at most 12 players**. Native
  random-forest tree games use their matching exact solvers. Preserve recorded
  floating-point qualification and finite-endpoint FSII reference-floor policies;
  never replace a missing exact reference with an approximate estimate.
- Targets are **SV and order-two k-SII, SII, STII, FSII and FBII**, where supported.
  Constructor-supported targets determine the capability catalog; unsupported
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
verified. One
consolidated reference, ownership, numerical and coverage check precedes
publication of the report and its reproducibility artifacts.

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
