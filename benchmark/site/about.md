# About the benchmark

We compare Shapley estimators on games built from real datasets. Each method gets
an allowance of coalition queries, and its answer is compared with an exact
reference for the same frozen game.

**This guide describes the accepted focused replacement plan.** The dashboard
currently displays an earlier audited cohort, whose games, estimator settings and
weighting remain those recorded in its report. The focused replacement has not
yet been published.

[TOC]

## Scope

The accepted design has **448 intended game instances across 12 datasets**:
112 recipes, each with four construction seeds. Twenty-four instances are already
excluded by recorded model-quality checks, leaving **up to 424 qualified
instances**. Further qualification can reduce that number. These are design
counts, not a claim that every estimator run has finished.

| Application | Intended instances | Maximum after existing exclusions | Players |
| --- | ---: | ---: | --- |
| Individual predictions | 128 | 124 | 12–512 input features |
| Data valuation | 160 | 152 | 12 or 14 training-data groups; 32–512 individual examples for neighbor games |
| Feature selection | 160 | 148 | 12 or 14 input features |

This is one unified benchmark and one final publication. The currently published
report remains available until the replacement passes its numerical and
publication audits. Its displayed coverage and downloadable metadata identify
what has actually been measured; the intended design above is not a substitute
for that evidence.

## Games and datasets

A game assigns a payoff to each subset of its players. What a player represents
and how its payoff is computed depend on the application:

- **Individual predictions:** players are input features. A coalition retains
  selected features while a fixed predictor's missing inputs are handled by the
  recorded baseline, marginal or tree construction.
- **Data valuation:** players are groups of training rows, and the model is
  refitted using the selected groups. Neighbor games instead treat individual
  training examples as players and use their specified KNN or threshold-neighbor
  utility.
- **Feature selection:** players are input features. The model is refitted on
  the selected columns and scored on held-out examples.

The original eight datasets are Adult Census, Wine Quality, Mushroom, Ionosphere,
Communities and Crime, NHANES I, Bioresponse and QSAR-TID11. Their 96 recipes and
frozen results are preserved. Models include random forests, XGBoost and
LightGBM, with selected SVM, MLP, Gaussian-process and TabPFN comparisons.

Four additional datasets broaden data valuation and feature selection. Each row
below contributes both applications at 12 and 14 players, with four construction
seeds: 16 new recipes and 64 intended instances in total.

| Dataset | Model | Input columns for data valuation | Feature-selection players |
| --- | --- | ---: | --- |
| Breast Cancer | Random forest | 24 | 12 and 14 |
| Digits | Linear model | 64 | 12 and 14 |
| Miami Housing | LightGBM | 15 | 12 and 14 |
| Superconductivity | Linear model | 64 | 12 and 14 |

**Input columns and players are different axes in data valuation.** A 14-player
Digits game has 14 groups of training examples and 64 input columns. Its
coalitions select groups, not columns. Matched feature-selection recipes use
nested columns: the 12 selected columns are contained in the 14-column selection,
with the same seeded split. Dataset repair and model fitting use their recorded
training rows.

## Exact references

Generic 12- and 14-player games enumerate every coalition: respectively 4,096
and 16,384 payoffs. Each retraining payoff can require another model fit. The
saved table fixes the game realization, and exact coefficients are computed from
that table with recorded numerical checks.

Larger fixed-tree games use specialized exact tree solvers. A tree used inside a
retraining game does not make that game eligible for TreeSHAP: the predictor
changes between coalitions. Neighbor games use their specialized exact Shapley
value solvers.

The six target panels are **SV, SII, k-SII, STII, FSII and FBII**, with interaction
order two. Enumerated and interventional-tree games support all six targets;
native path-dependent trees support SV, SII and k-SII; neighbor games support SV
only. An estimator is compared only on targets it supports. Exact references
refer to the stated frozen game, subject to its documented numerical tolerance;
a high-budget approximation is not used as exact truth.

## Estimators and query budgets

The focused protocol includes **22 estimators**, four game-construction seeds
and one estimator seed. The results page describes each method and exposes its supported
targets. Frozen estimator settings, seeds and earlier measurements are retained.
In the focused campaign, LeverageSHAP and OddSHAP explicitly use the two-query
equal-allocation fallback at eligible low budgets; this can consume less than the
allowance.

For a game with `d` players, the nine query allowances are:

**0.5d, 1d, 2d, 4d, 8d, 16d, 32d, 64d and 128d.**

A coalition evaluation counts as a query, including repeated evaluations. For
example, 8d allows 96 queries in a 12-player game. Charts show the requested
allowance; recorded usage can be smaller. Timeouts and failed attempts are
explicit outcomes, never invented estimates or zero-error results.

## Scores

**Normalized mean squared error (nMSE)** is the sum of squared coefficient errors
divided by the sum of squared exact coefficients. The empty-coalition baseline is
excluded. Lower is better: zero is exact agreement, one has the error of
predicting all scored coefficients as zero, and values above one are worse than
that baseline.

Zero or negligible truth signal is excluded from accuracy summaries for every
method. Interaction-order filters apply the same signal rule to the selected
order. The exact signal reference and threshold are recorded in the report.

The focused replacement will balance applications equally, then subtypes,
recipes, instances, budgets and estimator seeds. Its data-valuation SV gives
equal weight to group retraining and neighbor-example games. Scores for partial coverage renormalize
successful observations; failures remain in the coverage denominator. The earlier
published cohort balances game families and configurations under its recorded
protocol. Check coverage alongside accuracy, because methods with missing runs
can be measured on different subsets.

Elo compares errors on shared game/target/budget/seed cells and summarizes paired
wins and ties. It is a relative ranking, not an error magnitude. “Same cells for
all” restricts comparisons to common successful cells. Bootstrap intervals
preserve related construction seeds and are descriptive; four constructions and
one estimator seed do not isolate estimator randomness.

## Timing and reproducibility

The campaign uses CPU workers only. Preparation, evaluation, diagnostics and
export share a 2,048 allocated CPU-hour cap. Concurrent estimator timings are
diagnostic, and estimated uncached time combines estimator work with recorded
payoff costs; it is not a fresh end-to-end timing.

Every published cell must have an explicit outcome and authenticated ownership,
source, hardware and score metadata. Successful earlier results are retained;
recovery attempts cannot silently replace them. Independent audits check the
final cohort, numerical scores, browser filters and deployed report before the
replacement is accepted.

Use the results page's downloads for the published games, exact answers,
settings and reproduction files. The design and executable recipes live in the
repository:

- [Accepted focused plan](https://github.com/rtealwitter/shapiq/blob/benchmark/benchmark/FOCUSED.md)
- [Reviewed recipe table](https://github.com/rtealwitter/shapiq/blob/benchmark/benchmark/suites/focused.csv)
- [Run or reproduce the benchmark](https://github.com/rtealwitter/shapiq/blob/benchmark/benchmark/README.md)
- [Markdown source of this guide](about.md)
