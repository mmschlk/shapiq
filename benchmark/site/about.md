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
**2,964 / 5,080 instances**; additional runs are in progress.

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
| Adult Census · core + extra | 4, 8, 12, 14 / 8 | 32 / 72 |
| Mushroom · core | 4, 8, 12, 14, 16 / 8, 16 | 47 / 88 |
| Ionosphere · core | 4, 8, 12, 14, 16 / 8, 16, 32 | 72 / 112 |
| Bioresponse · core + extra | 4, 8, 12, 14, 16 / 8, 16, 32, 64, 128, 256, 512 | 179 / 224 |
| Breast Cancer · core + extra | 4, 8, 12, 14, 16 / 8, 16 | 63 / 104 |
| Digits · core + extra | 4, 8, 12, 14, 16 / 8, 16, 32, 64 | 112 / 152 |
| Wine (classification) · core | 4, 8, 12, 13 / 8 | 24 / 56 |
| Amazon Employee Access · core | 4, 8, 9 / 8 | 24 / 48 |
| APS Failure · core | 4, 8, 12, 14, 16 / 8, 16, 32, 64, 128 | 118 / 160 |
| Anneal · core | 4, 8, 12, 14, 16 / 8, 16, 32 | 71 / 112 |
| Splice · core | 4, 8, 12, 14, 16 / 8, 16, 32 | 71 / 112 |
| Credit Card Default · core | 4, 8, 12, 14, 16 / 8, 16 | 48 / 88 |
| Wine Quality · core + extra | 4, 8, 12 / 8 | 24 / 64 |
| Communities and Crime · core | 4, 8, 12, 14, 16 / 8, 16, 32, 64 | 95 / 136 |
| NHANES I · core | 4, 8, 12, 14, 16 / 8, 16, 32, 64 | 91 / 136 |
| QSAR-TID11 · core + extra | 4, 8, 12, 14, 16 / 8, 16, 32, 64, 128, 256, 512 | 184 / 224 |
| Miami Housing · core + extra | 4, 8, 12, 14, 15 / 8 | 39 / 80 |
| Superconductivity · core + extra | 4, 8, 12, 14, 16 / 8, 16, 32, 64 | 111 / 152 |
| California Housing · core | 4, 8 / 8 | 24 / 40 |
| Diabetes · core | 4, 8, 10 / 8 | 24 / 48 |
| Bike Sharing · core | 4, 8, 12 / 8 | 23 / 48 |
| Airfoil Self Noise · core | 4, 5 / — | 16 / 16 |
| Concrete Strength · core | 4, 8 / 8 | 24 / 40 |
| Protein · core | 4, 8, 9 / 8 | 24 / 48 |

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
| Adult Census · core + extra | 4, 8, 12, 14, 16 / 32, 64, 128, 256, 512, 1024 | 74 / 116 |
| Mushroom · core | 4, 8, 12, 14, 16 / 32, 64, 128, 256, 512, 1024 | 44 / 88 |
| Ionosphere · core | 4, 8, 12, 14, 16 / 32, 64, 128, 256 | 31 / 72 |
| Bioresponse · core + extra | 4, 8, 12, 14, 16 / 32, 64, 128, 256, 512, 1024 | 95 / 148 |
| Breast Cancer · core + extra | 4, 8, 12, 14, 16 / 32, 64, 128, 256 | 67 / 108 |
| Digits · core + extra | 4, 8, 12, 14, 16 / 32, 64, 128, 256, 512, 1024 | 88 / 132 |
| Wine (classification) · core | 4, 8, 12, 14, 16 / 32, 64, 128 | 23 / 64 |
| Amazon Employee Access · core | 4, 8, 12, 14, 16 / 32, 64, 128, 256, 512, 1024 | 44 / 88 |
| APS Failure · core | 4, 8, 12, 14, 16 / 32, 64, 128, 256, 512, 1024 | 46 / 88 |
| Anneal · core | 4, 8, 12, 14, 16 / 32, 64, 128, 256, 512 | 38 / 80 |
| Splice · core | 4, 8, 12, 14, 16 / 32, 64, 128, 256, 512, 1024 | 46 / 88 |
| Credit Card Default · core | 4, 8, 12, 14, 16 / 32, 64, 128, 256, 512, 1024 | 43 / 88 |
| Wine Quality · core + extra | 4, 8, 12, 14, 16 / — | 20 / 60 |
| Communities and Crime · core | 4, 8, 12, 14, 16 / — | 21 / 40 |
| NHANES I · core | 4, 8, 12, 14, 16 / — | 19 / 40 |
| QSAR-TID11 · core + extra | 4, 8, 12, 14, 16 / — | 57 / 100 |
| Miami Housing · core + extra | 4, 8, 12, 14, 16 / — | 21 / 68 |
| Superconductivity · core + extra | 4, 8, 12, 14, 16 / — | 38 / 92 |
| California Housing · core | 4, 8, 12, 14, 16 / — | 15 / 40 |
| Diabetes · core | 4, 8, 12, 14, 16 / — | 23 / 40 |
| Bike Sharing · core | 4, 8, 12, 14, 16 / — | 18 / 40 |
| Airfoil Self Noise · core | 4, 8, 12, 14, 16 / — | 22 / 40 |
| Concrete Strength · core | 4, 8, 12, 14, 16 / — | 20 / 40 |
| Protein · core | 4, 8, 12, 14, 16 / — | 16 / 40 |

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
| Adult Census · core + extra | 4, 8, 12, 14 | 23 / 48 |
| Mushroom · core | 4, 8, 12, 14, 16 | 22 / 40 |
| Ionosphere · core | 4, 8, 12, 14, 16 | 23 / 40 |
| Bioresponse · core + extra | 4, 8, 12, 14, 16 | 19 / 56 |
| Breast Cancer · core + extra | 4, 8, 12, 14, 16 | 23 / 56 |
| Digits · core + extra | 4, 8, 12, 14, 16 | 22 / 56 |
| Wine (classification) · core | 4, 8, 12, 13 | 24 / 32 |
| Amazon Employee Access · core | 4, 8, 9 | 22 / 24 |
| APS Failure · core | 4, 8, 12, 14, 16 | 19 / 40 |
| Anneal · core | 4, 8, 12, 14, 16 | 22 / 40 |
| Splice · core | 4, 8, 12, 14, 16 | 19 / 40 |
| Credit Card Default · core | 4, 8, 12, 14, 16 | 19 / 40 |
| Wine Quality · core + extra | 4, 8, 12 | 22 / 40 |
| Communities and Crime · core | 4, 8, 12, 14, 16 | 20 / 40 |
| NHANES I · core | 4, 8, 12, 14, 16 | 19 / 40 |
| QSAR-TID11 · core + extra | 4, 8, 12, 14, 16 | 21 / 56 |
| Miami Housing · core + extra | 4, 8, 12, 14, 15 | 20 / 56 |
| Superconductivity · core + extra | 4, 8, 12, 14, 16 | 17 / 56 |
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
- **Reproduction:** [download saved games, results and instructions](https://github.com/rtealwitter/shapiq/releases/tag/benchmark-comprehensive-2026-10-07).
  The dashboard also exports selected results as JSON or CSV.
