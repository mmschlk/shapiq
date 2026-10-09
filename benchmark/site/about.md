# About the benchmark

**How well can we estimate contributions with a limited number of queries?**
We compare 22 estimators across individual predictions, training data and feature selection.

[TOC]

## Games, Shapley values and estimation error

- **Game:** for players $N = \{1, \ldots, d\}$, a function $f: 2^N \to \mathbb{R}$ assigns a value
  $f(S)$ to each coalition $S \subseteq N$. The applications below define the players and $f$.
- **Shapley value:** player $i$'s average marginal contribution across all player orderings:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(d - |S| - 1)!}{d!}\left[f(S \cup \{i\}) - f(S)\right].$$

- **Estimator:** an algorithm queries $f$ at most $B$ times and returns an estimated
  vector $\hat{\boldsymbol{\phi}}$, to approximate the exact contribution vector $\boldsymbol{\phi}$.
- **Normalized mean squared error (nMSE):**

$$\mathbf{nMSE} = \frac{\lVert \hat{\boldsymbol{\phi}} - \boldsymbol{\phi} \rVert_2^2}{\lVert \boldsymbol{\phi} \rVert_2^2}.$$

**Lower is better:** zero is exact; one is the error of estimating every contribution
as zero. Zero or negligible reference vectors are excluded. For interaction targets,
the vectors contain the requested interaction coefficients; the empty-coalition
baseline is excluded. Main effects and pairs can also be viewed separately.

**Reading the tables:** an instance is one game construction with one of four
construction seeds. “Exact / planned” counts available exact references in the
published cohort, **not** successful estimator runs. The published total is
**4,568 / 5,080 instances (89.9% reference coverage)**; 512 instances have no
qualified reference. All admitted allocations have ended. Estimator coverage remains
partial: failures and unattempted cells are unscored.

## Explaining an individual prediction

**Players are input features.** Fit a predictor $h$ once and explain one held-out
point $x$. For the small games, keep $x$'s selected features and replace the others
with fixed training-background means $b$:

$$f(S) = h(z(S)) - h(b), \qquad z_i(S) = \begin{cases}x_i & i \in S,\\ b_i & i \notin S.\end{cases}$$

The output is a fixed class probability or a regression prediction.
**Exact reference:** evaluate all $2^d$ coalitions.

**Tree games** extend this to larger player sets. Interventional games average
predictions with missing features filled from fixed background rows. Path-dependent games average missing
branches using training-path weights. Matching exact tree solvers supply the
references; boosted classifiers use their recorded output margin.

**Models:** core = linear/logistic regression + random forest; extra = XGBoost,
LightGBM, RBF SVM + MLP, at **12 features only**. Tree games use random forest,
XGBoost and LightGBM. Player sets below are the planned grids, where feasible.

| Dataset / models | Features: small / tree | Exact / planned instances |
| --- | --- | ---: |
| Adult Census · core + extra | 4, 8, 12, 14 / 8 | 68 / 72 |
| Mushroom · core | 4, 8, 12, 14, 16 / 8, 16 | 85 / 88 |
| Ionosphere · core | 4, 8, 12, 14, 16 / 8, 16, 32 | 109 / 112 |
| Bioresponse · core + extra | 4, 8, 12, 14, 16 / 8, 16, 32, 64, 128, 256, 512 | 215 / 224 |
| Breast Cancer · core + extra | 4, 8, 12, 14, 16 / 8, 16 | 101 / 104 |
| Digits · core + extra | 4, 8, 12, 14, 16 / 8, 16, 32, 64 | 148 / 152 |
| Wine (classification) · core | 4, 8, 12, 13 / 8 | 53 / 56 |
| Amazon Employee Access · core | 4, 8, 9 / 8 | 44 / 48 |
| APS Failure · core | 4, 8, 12, 14, 16 / 8, 16, 32, 64, 128 | 153 / 160 |
| Anneal · core | 4, 8, 12, 14, 16 / 8, 16, 32 | 109 / 112 |
| Splice · core | 4, 8, 12, 14, 16 / 8, 16, 32 | 109 / 112 |
| Credit Card Default · core | 4, 8, 12, 14, 16 / 8, 16 | 84 / 88 |
| Wine Quality · core + extra | 4, 8, 12 / 8 | 61 / 64 |
| Communities and Crime · core | 4, 8, 12, 14, 16 / 8, 16, 32, 64 | 133 / 136 |
| NHANES I · core | 4, 8, 12, 14, 16 / 8, 16, 32, 64 | 124 / 136 |
| QSAR-TID11 · core + extra | 4, 8, 12, 14, 16 / 8, 16, 32, 64, 128, 256, 512 | 220 / 224 |
| Miami Housing · core + extra | 4, 8, 12, 14, 15 / 8 | 76 / 80 |
| Superconductivity · core + extra | 4, 8, 12, 14, 16 / 8, 16, 32, 64 | 147 / 152 |
| California Housing · core | 4, 8 / 8 | 36 / 40 |
| Diabetes · core | 4, 8, 10 / 8 | 44 / 48 |
| Bike Sharing · core | 4, 8, 12 / 8 | 45 / 48 |
| Airfoil Self Noise · core | 4, 5 / — | 16 / 16 |
| Concrete Strength · core | 4, 8 / 8 | 37 / 40 |
| Protein · core | 4, 8, 9 / 8 | 45 / 48 |

## Valuing training data

**Players are groups of training rows.** For a coalition $S$, fit $h_S$ on the union
of its groups and evaluate on a fixed held-out set $T$:

$$f(S) = \begin{cases}\operatorname{accuracy}(h_S, T) & \text{classification},\\ -\operatorname{MSE}(h_S, T) & \text{regression},\end{cases} \qquad f(\varnothing) = 0.$$

**Exact reference:** refit and score every coalition. Core models use up to
12 input features. At 12 groups, the extra-model datasets also vary the core
models' input width through 8, 16, 32, 64, 128 and full width, where available.
Input width is separate from the number of players.

**Neighbor games** instead use individual training rows as players, for a fixed
test example. KNN utility is the number of matching labels among the coalition's
nearest $\min(k, |S|)$ rows, divided by fixed $k$. TNN utility is the matching-label
fraction within a fixed radius, or $1 / \text{number of classes}$ if none are present.
Specialized formulas give **exact Shapley values** for these larger games.

**Models:** core = linear/logistic regression + random forest; extra = XGBoost,
LightGBM + MLP, at **12 groups only**. Neighbor games use KNN and TNN on the
classification datasets. “—” means no neighbor games.

| Dataset / models | Players: groups / neighbors | Exact / planned instances |
| --- | --- | ---: |
| Adult Census · core + extra | 4, 8, 12, 14, 16 / 32, 64, 128, 256, 512, 1024 | 109 / 116 |
| Mushroom · core | 4, 8, 12, 14, 16 / 32, 64, 128, 256, 512, 1024 | 81 / 88 |
| Ionosphere · core | 4, 8, 12, 14, 16 / 32, 64, 128, 256 | 66 / 72 |
| Bioresponse · core + extra | 4, 8, 12, 14, 16 / 32, 64, 128, 256, 512, 1024 | 128 / 148 |
| Breast Cancer · core + extra | 4, 8, 12, 14, 16 / 32, 64, 128, 256 | 102 / 108 |
| Digits · core + extra | 4, 8, 12, 14, 16 / 32, 64, 128, 256, 512, 1024 | 124 / 132 |
| Wine (classification) · core | 4, 8, 12, 14, 16 / 32, 64, 128 | 58 / 64 |
| Amazon Employee Access · core | 4, 8, 12, 14, 16 / 32, 64, 128, 256, 512, 1024 | 79 / 88 |
| APS Failure · core | 4, 8, 12, 14, 16 / 32, 64, 128, 256, 512, 1024 | 78 / 88 |
| Anneal · core | 4, 8, 12, 14, 16 / 32, 64, 128, 256, 512 | 75 / 80 |
| Splice · core | 4, 8, 12, 14, 16 / 32, 64, 128, 256, 512, 1024 | 81 / 88 |
| Credit Card Default · core | 4, 8, 12, 14, 16 / 32, 64, 128, 256, 512, 1024 | 78 / 88 |
| Wine Quality · core + extra | 4, 8, 12, 14, 16 / — | 45 / 60 |
| Communities and Crime · core | 4, 8, 12, 14, 16 / — | 32 / 40 |
| NHANES I · core | 4, 8, 12, 14, 16 / — | 29 / 40 |
| QSAR-TID11 · core + extra | 4, 8, 12, 14, 16 / — | 88 / 100 |
| Miami Housing · core + extra | 4, 8, 12, 14, 16 / — | 47 / 68 |
| Superconductivity · core + extra | 4, 8, 12, 14, 16 / — | 62 / 92 |
| California Housing · core | 4, 8, 12, 14, 16 / — | 25 / 40 |
| Diabetes · core | 4, 8, 12, 14, 16 / — | 35 / 40 |
| Bike Sharing · core | 4, 8, 12, 14, 16 / — | 29 / 40 |
| Airfoil Self Noise · core | 4, 8, 12, 14, 16 / — | 34 / 40 |
| Concrete Strength · core | 4, 8, 12, 14, 16 / — | 32 / 40 |
| Protein · core | 4, 8, 12, 14, 16 / — | 27 / 40 |

## Selecting useful features

**Players are input features.** For every coalition $S$, fit a fresh predictor $h_S$
using only those columns, keeping the training and held-out rows fixed:

$$f(S) = \begin{cases}\operatorname{accuracy}(h_S, T_S) & \text{classification},\\ -\operatorname{MSE}(h_S, T_S) & \text{regression},\end{cases} \qquad f(\varnothing) = 0.$$

**Exact reference:** fit and score all $2^d$ coalitions. Unlike an individual-prediction
game, this measures how well a model can learn from the chosen features.

**Models:** core = linear/logistic regression + random forest; extra = XGBoost,
LightGBM, RBF SVM + MLP, at **12 features only**.

| Dataset / models | Features | Exact / planned instances |
| --- | --- | ---: |
| Adult Census · core + extra | 4, 8, 12, 14 | 42 / 48 |
| Mushroom · core | 4, 8, 12, 14, 16 | 32 / 40 |
| Ionosphere · core | 4, 8, 12, 14, 16 | 35 / 40 |
| Bioresponse · core + extra | 4, 8, 12, 14, 16 | 42 / 56 |
| Breast Cancer · core + extra | 4, 8, 12, 14, 16 | 49 / 56 |
| Digits · core + extra | 4, 8, 12, 14, 16 | 49 / 56 |
| Wine (classification) · core | 4, 8, 12, 13 | 29 / 32 |
| Amazon Employee Access · core | 4, 8, 9 | 22 / 24 |
| APS Failure · core | 4, 8, 12, 14, 16 | 29 / 40 |
| Anneal · core | 4, 8, 12, 14, 16 | 34 / 40 |
| Splice · core | 4, 8, 12, 14, 16 | 33 / 40 |
| Credit Card Default · core | 4, 8, 12, 14, 16 | 29 / 40 |
| Wine Quality · core + extra | 4, 8, 12 | 33 / 40 |
| Communities and Crime · core | 4, 8, 12, 14, 16 | 31 / 40 |
| NHANES I · core | 4, 8, 12, 14, 16 | 27 / 40 |
| QSAR-TID11 · core + extra | 4, 8, 12, 14, 16 | 49 / 56 |
| Miami Housing · core + extra | 4, 8, 12, 14, 15 | 40 / 56 |
| Superconductivity · core + extra | 4, 8, 12, 14, 16 | 38 / 56 |
| California Housing · core | 4, 8 | 15 / 16 |
| Diabetes · core | 4, 8, 10 | 24 / 24 |
| Bike Sharing · core | 4, 8, 12 | 24 / 24 |
| Airfoil Self Noise · core | 4, 5 | 16 / 16 |
| Concrete Strength · core | 4, 8 | 16 / 16 |
| Protein · core | 4, 8, 9 | 24 / 24 |

## Comparing and reproducing results

- **Budgets:** 0.5, 1, 2, 4, 8, 16, 32, 64 and 128 × $d$ coalition queries;
  three estimator seeds per game. Actual query use can be smaller.
- **Targets:** Shapley values and order-two k-SII, SII, STII, FSII and FBII
  interactions, where supported. Neighbor games provide Shapley values only.
- **Scores:** weighted means and medians. Applications receive equal weight,
  followed by subtype and recipe; datasets with more recipes can have more weight.
- **Coverage:** failures and missing runs are unscored. Methods below **80%**
  successful/planned coverage appear at the leaderboard's bottom and leave the figures.
- **Ground truth:** exact for the saved game, subject to floating-point precision.
  Generic large games never use an approximate estimate as an exact reference.
- **Data notes:** Wine classification differs from Wine Quality regression.
  NHANES I uses its supplied survival label as a regression surrogate.
- **Timing:** measured estimator time on cached games; estimated oracle costs are
  shown separately. These are diagnostic timings, not isolated end-to-end timings.
- **Reproduction:** [download saved games, results and instructions](https://github.com/rtealwitter/shapiq/releases/tag/benchmark-comprehensive-2026-10-09).
  The dashboard also exports selected results as JSON or CSV.
