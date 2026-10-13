# About the benchmark

**How well can we estimate contributions with a limited number of queries?**
Scope version 2 compares 22 estimator configurations across individual predictions,
training data and feature selection on all 24 datasets.

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

**The selected cohort has 512 recipes and 2,048 prepared instances**, with four
construction seeds per recipe: 477 core recipes and 35 KNN/TNN recipes.
Every selected instance has a qualified reference. All 20 required estimators
have **100% successful supported coverage**; ShaplEIG and SPEX retain partial
coverage. The tables show selected player grids; estimator coverage counts only
eligible, supported cells.

Thirty-five core recipes were excluded in full because at least one seed failed
the model-quality check against a dummy predictor. Neighbor replacements were
selected for balanced coverage without using estimator accuracy; the
[selection manifest](https://github.com/rtealwitter/shapiq/blob/benchmark/benchmark/suites/compact-selection.json)
records the exclusions and deterministic replacement rule. All 24 datasets
and all three applications remain represented. Historical broader results outside
this selection do not count toward this cohort’s coverage.

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
references. Both tree constructions use random forests only.

**Models:** linear/logistic regression and random forest for baseline games,
with nested feature sets of at most 12 players. RF tree games use 8, 16 and 32
features where available. A dataset's full width below 12 is also included in the
baseline grid. There are no extra-model contrasts. Each table lists the retained
model-specific grids; “—” means no selected recipe for that model/construction.

| Dataset | Baseline features (linear / RF) | RF tree features (path / interventional) |
| --- | --- | --- |
| Adult Census | 4, 8, 12 / 4, 8, 12 | 8 / 8 |
| Mushroom | 4, 8, 12 / 4, 8, 12 | 8, 16 / 8, 16 |
| Ionosphere | 4, 8, 12 / 4, 8, 12 | 8, 16, 32 / 8, 16, 32 |
| Bioresponse | — / 4, 8, 12 | 8, 16, 32 / 8, 16, 32 |
| Breast Cancer | 4, 8, 12 / 4, 8, 12 | 8, 16 / 8, 16 |
| Digits | 4, 8, 12 / 4, 8, 12 | 8, 16, 32 / 8, 16, 32 |
| Wine (classification) | 4, 8, 12 / 4, 8, 12 | 8 / 8 |
| Amazon Employee Access | 8, 9 / 8, 9 | 8 / 8 |
| APS Failure | 4, 8, 12 / 4, 12 | 16, 32 / 16, 32 |
| Anneal | 4, 8, 12 / 4, 8, 12 | 8, 16, 32 / 8, 16, 32 |
| Splice | 4, 8, 12 / 4, 8, 12 | 8, 16, 32 / 8, 16, 32 |
| Credit Card Default | 4, 8, 12 / 4, 8, 12 | 8, 16 / 8, 16 |
| Wine Quality | 4, 8, 12 / 4, 8, 12 | 8 / 8 |
| Communities and Crime | 4, 8, 12 / 4, 8, 12 | 8, 16, 32 / 8, 16, 32 |
| NHANES I | — / 8, 12 | 8, 32 / 8, 32 |
| QSAR-TID11 | 4, 8, 12 / 4, 8, 12 | 8, 16, 32 / 8, 16, 32 |
| Miami Housing | 4, 8, 12 / 4, 8, 12 | 8 / 8 |
| Superconductivity | 4, 8, 12 / 4, 8, 12 | 8, 16, 32 / 8, 16, 32 |
| California Housing | 4 / 4, 8 | 8 / 8 |
| Diabetes | 4, 8, 10 / 4, 8, 10 | 8 / 8 |
| Bike Sharing | 4, 8, 12 / 4, 8, 12 | 8 / 8 |
| Airfoil Self Noise | 4, 5 / 4, 5 | — / — |
| Concrete Strength | 4, 8 / 4, 8 | 8 / 8 |
| Protein | 4, 8, 9 / 4, 8, 9 | 8 / 8 |

## Valuing training data

**Players are groups of training rows.** For a coalition $S$, fit $h_S$ on the union
of its groups and evaluate on a fixed held-out set $T$:

$$f(S) = \begin{cases}\operatorname{accuracy}(h_S, T) & \text{classification},\\ -\operatorname{MSE}(h_S, T) & \text{regression},\end{cases} \qquad f(\varnothing) = 0.$$

**Exact reference:** refit and score every coalition of 4, 8 or 12 groups.
Both linear/logistic regression and random forest use exactly
$\min(12, \text{dataset width})$ input features. There is no input-width sweep;
input width is separate from the number of players.

**Neighbor games** use individual training rows as players for a fixed test
example. KNN utility counts matching labels among the coalition's nearest
$\min(k, |S|)$ rows, divided by fixed $k$. TNN utility is the matching-label fraction
within a fixed radius, or $1 / \text{number of classes}$ when no rows are present.
Selected classification games use 32 or 64 players, with exact native Shapley
references. All input selections are fixed before estimator evaluation.

| Dataset | Training-row groups (linear / RF) | Neighbor players (KNN / TNN) | Input features |
| --- | --- | --- | --- |
| Adult Census | 4, 8, 12 / 4, 8, 12 | 64 / 64 | 12 |
| Mushroom | 4, 8, 12 / 4, 8, 12 | 32 / 32, 64 | 12 |
| Ionosphere | 4, 8, 12 / 4, 8, 12 | 32, 64 / 64 | 12 |
| Bioresponse | — / 4, 8, 12 | 32, 64 / 32 | 12 |
| Breast Cancer | 4, 8, 12 / 4, 8, 12 | 64 / 32, 64 | 12 |
| Digits | 4, 8, 12 / 4, 8, 12 | 32 / 32, 64 | 12 |
| Wine (classification) | 4, 8, 12 / 4, 8, 12 | 32, 64 / 64 | 12 |
| Amazon Employee Access | 4, 8, 12 / 4, 8, 12 | 32, 64 / 32 | 9 |
| APS Failure | 4, 8, 12 / 4, 8, 12 | 64 / 32, 64 | 12 |
| Anneal | 4, 8, 12 / 4, 8, 12 | 32 / 32, 64 | 12 |
| Splice | 4, 8, 12 / 4, 8, 12 | 32, 64 / 64 | 12 |
| Credit Card Default | 4, 8, 12 / 4, 8, 12 | 32, 64 / 32 | 12 |
| Wine Quality | 4, 8, 12 / 4, 8, 12 | — / — | 12 |
| Communities and Crime | 4, 8, 12 / 4, 8, 12 | — / — | 12 |
| NHANES I | — / 4, 8, 12 | — / — | 12 |
| QSAR-TID11 | 4, 8, 12 / 4, 8, 12 | — / — | 12 |
| Miami Housing | 4, 8, 12 / 4, 8, 12 | — / — | 12 |
| Superconductivity | 4, 8, 12 / 4, 8, 12 | — / — | 12 |
| California Housing | — / 4, 8, 12 | — / — | 8 |
| Diabetes | 4, 8, 12 / 4, 8, 12 | — / — | 10 |
| Bike Sharing | 4, 8, 12 / 4, 8, 12 | — / — | 12 |
| Airfoil Self Noise | 4, 8, 12 / 4, 8, 12 | — / — | 5 |
| Concrete Strength | 4, 8, 12 / 4, 8, 12 | — / — | 8 |
| Protein | 4, 8, 12 / 4, 8, 12 | — / — | 9 |

## Selecting useful features

**Players are input features.** For every coalition $S$, fit a fresh predictor $h_S$
using only those columns, keeping the training and held-out rows fixed:

$$f(S) = \begin{cases}\operatorname{accuracy}(h_S, T_S) & \text{classification},\\ -\operatorname{MSE}(h_S, T_S) & \text{regression},\end{cases} \qquad f(\varnothing) = 0.$$

**Exact reference:** fit and score all $2^d$ coalitions. Unlike an individual-prediction
game, this measures how well a model can learn from the chosen features.

**Models:** linear/logistic regression and random forest, with the same nested
feature grids of at most 12 players used by the baseline prediction games.

| Dataset | Linear features | RF features |
| --- | --- | --- |
| Adult Census | 4, 8, 12 | 4, 8, 12 |
| Mushroom | 4, 8, 12 | 4, 8, 12 |
| Ionosphere | 4, 8, 12 | 4, 8, 12 |
| Bioresponse | — | 4, 8, 12 |
| Breast Cancer | 4, 8, 12 | 4, 8, 12 |
| Digits | 4, 8, 12 | 4, 8, 12 |
| Wine (classification) | 4, 8, 12 | 4, 8, 12 |
| Amazon Employee Access | 8, 9 | 8, 9 |
| APS Failure | 4, 8, 12 | 4, 12 |
| Anneal | 4, 8, 12 | 4, 8, 12 |
| Splice | 4, 8, 12 | 4, 8, 12 |
| Credit Card Default | 4, 8, 12 | 4, 8, 12 |
| Wine Quality | 4, 8, 12 | 4, 8, 12 |
| Communities and Crime | 4, 8, 12 | 4, 8, 12 |
| NHANES I | — | 8, 12 |
| QSAR-TID11 | 4, 8, 12 | 4, 8, 12 |
| Miami Housing | 4, 8, 12 | 4, 8, 12 |
| Superconductivity | 4, 8, 12 | 4, 8, 12 |
| California Housing | 4 | 4, 8 |
| Diabetes | 4, 8, 10 | 4, 8, 10 |
| Bike Sharing | 4, 8, 12 | 4, 8, 12 |
| Airfoil Self Noise | 4, 5 | 4, 5 |
| Concrete Strength | 4, 8 | 4, 8 |
| Protein | 4, 8, 9 | 4, 8, 9 |

## Comparing and reproducing results

- **Budgets:** 0.5, 1, 2, 4, 8, 16 and 32 × $d$ coalition queries;
  three estimator seeds per game. Actual query use can be smaller.
- **Targets:** Shapley values and order-two k-SII, SII, STII, FSII and FBII
  interactions, where supported. KNN/TNN games provide SV only.
- **Scores:** weighted means and medians. Applications receive equal weight,
  followed by subtype and recipe; datasets with more recipes can have more weight.
- **Completion:** 100% successful supported coverage for every estimator
  except ShaplEIG and SPEX. ProxySPEX is included. Failed and missing runs for the
  two exempt estimators remain visible and unscored.
- **Coverage:** exact-reference coverage and successful estimator coverage are
  separate. Unsupported targets are not failures. Zero-signal and quality
  exclusions are reported without changing frozen games to improve coverage.
- **Ground truth:** exhaustive tables for generic games of at most 12 players;
  matching exact native solvers for random-forest tree and neighbor games. References are
  subject to floating-point precision; small-game solver checks do not establish
  independent full-dimensional validation.
- **Data notes:** Wine classification differs from Wine Quality regression.
  NHANES I uses its supplied survival label as a regression surrogate.
- **Timing:** measured estimator time on cached games; estimated oracle costs are
  shown separately. These are diagnostic timings, not isolated end-to-end timings.
- **Reproduction:** the dashboard exports selected results as JSON or CSV.
  [Historical broad-cohort artifacts](https://github.com/rtealwitter/shapiq/releases/tag/benchmark-comprehensive-2026-10-11)
  remain available separately and do not contribute to this cohort. Frozen-game
  reproduction preserves authenticated references, original result identities
  and the source overlays used by each run.

## Estimator combinations

For each target and budget, we fit **signed ridge regression** with no intercept:

$$\hat{\boldsymbol{\phi}}_{\alpha} = \sum_{m=1}^{M} \alpha_m \hat{\boldsymbol{\phi}}^{(m)}.$$

- **Fit:** minimize weighted mean nMSE plus $\lambda\lVert\boldsymbol{\alpha}\rVert_2^2$.
  The penalty discourages large, cancelling coefficients; negative weights remain allowed.
- **Regularization:** choose $\lambda$ from $10^{-6}, 10^{-5}, \ldots, 1$ using
  dataset-grouped cross-validation. Each outer fold chooses its penalty using
  only its training datasets; held-out datasets enter neither fitting nor tuning.
- **Scores:** held-out normalized R² and nMSE, compared with equal weights.
  Displayed coefficients are refitted using all datasets.
- **Matching:** every panel contains the same game, budget and seed for every
  supported nonexempt estimator. All seeds and games of a dataset stay together.
- **PCA:** normalized estimator errors show shared error patterns.
  Coefficients describe combinations, not estimator importance.
