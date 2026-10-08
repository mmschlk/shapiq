# About the benchmark

**How accurately can we estimate Shapley values with a limited number of model
queries?** We compare 22 estimators across three applications, 24 datasets, and
small to large player sets.

A **player** is a feature or training example. A **coalition** is a subset of
players. Shapley values measure each player's contribution; interactions measure
contributions from players working together.

**The comprehensive results are published.** Coverage is shown alongside scores,
including comparisons that failed or were not reached.

[TOC]

## Explaining an individual prediction

**Why did a model make this prediction?** Train the model once, then vary which
features it can see for one held-out example.

- **Players:** input features.
- **Payoff:** a fixed class's predicted probability, or a regression prediction.
  Missing features are replaced with their training-data means.
- **Player counts:** 4, 8, 12, 14 and 16, where available.
- **Datasets and models:** all 24 datasets with logistic/linear regression and
  random forests. On the eight comparison datasets below, add XGBoost, LightGBM,
  RBF support-vector machines and multilayer perceptrons at 12 players.
- **Ground truth:** evaluate every coalition—4,096 at 12 players; 65,536 at 16.

**Larger tree games: 8, 16, 32, 64, 128, 256 and 512 features**, where available,
using random forests, XGBoost and LightGBM.

- **Interventional:** fill missing features from fixed background examples and
  average predictions.
- **Path-dependent:** average missing branches using training-path weights.
- **Ground truth:** matching exact tree solvers. These explain their respective
  missing-feature rules; boosted classifiers use the recorded model margin.

## Valuing training data

**Which training data helps a model learn?** Train on selected groups of rows and
measure performance on the same held-out test set.

- **Players:** 4, 8, 12, 14 or 16 groups of training rows.
- **Payoff:** classification accuracy or negative mean squared prediction error.
  Higher is better; the empty coalition has payoff zero.
- **Datasets and models:** all 24 datasets with logistic/linear regression and
  random forests. Add XGBoost, LightGBM and multilayer perceptrons at 12 groups
  on the eight comparison datasets.
- **Ground truth:** retrain and score every coalition, keeping model settings fixed.
- **Feature diversity:** the core uses up to 12 input features. At 12 groups,
  the eight comparison datasets also vary input width through 8, 16, 32, 64,
  128 and full width, where available. Groups are players; columns are inputs.

**Larger neighbor games: 32, 64, 128, 256, 512 or 1,024 individual training rows.**

- Use all 12 classification datasets, subject to available rows.
- Utility depends on nearby labels around a held-out example: a fixed number of
  neighbors (**KNN**) or a distance threshold (**TNN**).
- Specialized formulas give **exact Shapley values** without enumerating subsets.
  These games do not benchmark interactions.

## Selecting useful features

**Which measurements should a future model use?** Fit a fresh model for each
selected set of columns, then evaluate its held-out predictions.

- **Players:** 4, 8, 12, 14 or 16 features, where available.
- **Payoff:** accuracy or negative mean squared prediction error; zero for the
  empty coalition.
- **Datasets and models:** all 24 datasets with logistic/linear regression and
  random forests. Add XGBoost, LightGBM, RBF support-vector machines and
  multilayer perceptrons at 12 features on the eight comparison datasets.
- **Ground truth:** fit and score every coalition. These games stay small enough
  for exhaustive evaluation.

## Datasets and variation

**12 classification + 12 regression datasets.** Counts below are the loaders'
full input widths, before selecting features for a game.

| Classification dataset | Features | Regression dataset | Features |
| --- | ---: | --- | ---: |
| Adult Census | 14 | Wine Quality | 12 |
| Mushroom | 22 | Communities and Crime | 101 |
| Ionosphere | 33 | NHANES I | 79 |
| Bioresponse | 1,776 | QSAR-TID11 | 1,024 |
| Breast Cancer | 30 | Miami Housing | 15 |
| Digits | 64 | Superconductivity | 81 |
| Wine (sklearn) | 13 | California Housing | 8 |
| Amazon Employee Access | 9 | Diabetes (sklearn) | 10 |
| APS Failure | 170 | Bike Sharing | 12 |
| Anneal | 38 | Airfoil Self Noise | 5 |
| Splice | 60 | Concrete Strength | 8 |
| Credit Card Default | 23 | Protein | 9 |

- **Eight comparison datasets:** Adult Census, Breast Cancer, Digits, Bioresponse,
  Wine Quality, Miami Housing, Superconductivity and QSAR-TID11.
- **Feature selection:** larger selections contain the smaller selections from
  the same seed. Include natural full widths below 16; never add artificial features.
- **Repetition:** four game-construction seeds and three estimator seeds.
- **Coverage:** exact references for **2,964 of 5,080 planned game instances**.
  The report contains **14,733 targets** after duplicate removal. The run window
  ended before all planned comparisons finished; missing results stay visible.
- **Dataset notes:** Wine classification and Wine Quality regression are distinct.
  NHANES I's supplied survival label is used as a regression surrogate.

## Comparing estimators fairly

- **22 estimators**, evaluated only on supported targets: Shapley values and
  k-SII, SII, STII, FSII and FBII interactions through order two.
- **Query budgets:** 0.5, 1, 2, 4, 8, 16, 32, 64 and 128 times the player count.
  For 16 players, 8d allows 128 coalition evaluations. Actual usage is also recorded.
- **Separate ground truth:** complete payoff tables for small games; exact tree
  or neighbor solvers for larger games. References are withheld from estimators.
- **Exactness:** relative to the specified game, subject to floating-point precision.
  Generic large games are not assigned an approximate run as “exact” truth.

## Measuring error and reading the results

**Normalized error = sum of squared estimation errors / sum of squared reference values.**

- **Lower error is better:** zero is exact; one matches predicting zero throughout.
- **Main effects and pairs:** reported separately as well as together.
- **Zero or negligible signal:** excluded from normalized comparisons. The empty
  coalition's baseline is also excluded.
- **Weights:** equal weight across applications, then subtypes and recipes;
  repeated runs share recipe weight. Datasets with more recipes can carry more weight.
- **Summaries:** weighted mean and median errors; pairwise ratings compare methods
  where both succeeded. **Higher ratings are better.**
- **Coverage matters:** failures and missing runs never count as zero error.

## Compute and reproducibility

- **Hardware:** Hopper's AMD EPYC 9754 CPUs, one thread per worker, no GPUs.
  The main run lasted about a day; concurrency was reduced from 1,024 to 1,000
  cores to leave room for other work.
- **Saved inputs:** games, model settings, selected rows and columns, seeds and
  reference values. Preparation failures and resource limits remain visible.
- **Timing:** recorded successful timings measure estimator calls on cached games.
  Timeouts also include process startup and input loading.
- **Downloads:** [saved games, results and restoration instructions](https://github.com/rtealwitter/shapiq/releases/tag/benchmark-comprehensive-2026-10-07).
  The dashboard also exports selected measurements as JSON or CSV.
