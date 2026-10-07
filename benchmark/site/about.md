# About the benchmark

Which Shapley estimator gives the most accurate answer for the computation we can
afford? We study this across three uses of machine learning: explaining a
prediction, valuing training data, and choosing features.

Each use becomes a **cooperative game**. The players might be input features or
training examples. A coalition is a subset of those players, and its payoff is a
prediction or a measure of predictive performance. A Shapley value measures a
player's contribution, averaged over the different coalitions it could join.
Interactions measure contributions that arise from players working together.

Computing these quantities exactly can require evaluating every possible
coalition. Estimators try to recover them from far fewer evaluations. Our
benchmark tests how well that works as the dataset, model and number of players
change.

**The 24-dataset benchmark described here is running.** The dashboard currently
shows the earlier published results; it will update when the new results are ready.

[TOC]

## Explaining an individual prediction

Suppose a model predicts the price of a house. We want to understand how its
inputs contributed to that particular prediction. Here, **features are players**
and the payoff is the model's prediction for one held-out example.

We first train a model and then keep it fixed. To evaluate a coalition, we retain
the example's selected features and replace its other features with their means
from the training data. For a classifier, we explain a fixed class's predicted
probability; for regression, we explain the predicted value. Adding a feature
reveals how much it changes the prediction in that context.

The core uses **all 24 datasets listed below**, with logistic regression for
classification, linear regression for regression, and random forests for both.
On Adult Census, Breast Cancer, Digits, Bioresponse, Wine Quality, Miami Housing,
Superconductivity and QSAR-TID11, we also compare XGBoost, LightGBM, support-vector
machines with an RBF kernel, and multilayer perceptrons at twelve players.

For these baseline-replacement games, we use **4, 8, 12, 14 and 16 features** where
available. We evaluate every coalition to obtain the exact reference. A
12-feature game has 4,096 coalitions; a 16-feature game has 65,536.

We also study larger tree explanations using random forests, XGBoost and
LightGBM on datasets with enough columns: **8, 16, 32, 64, 128, 256 and 512
features**. These use two tree-specific constructions. Interventional games fill
missing features from fixed background examples and average the predictions.
Path-dependent games average missing branches using the tree's training-path
weights. Each has its own matching exact tree solver. Their explanations refer
to those particular missing-feature rules; boosted classifier games use the
recorded model margin rather than assuming a probability scale.

## Valuing training data

Suppose we can collect data from several sources. Which sources help a model
predict well on new examples? Here, **groups of training rows are players**, and
the payoff is the performance of a model trained on the selected groups.

We split the training rows into **4, 8, 12, 14 or 16 groups**. For each coalition,
we combine its groups, train a fresh model, and evaluate it on the same held-out
test set. Classification games use accuracy; regression games use negative mean
squared prediction error, so larger payoffs mean better predictions. The empty
coalition is assigned a payoff of zero. Model settings are chosen before the
game and stay fixed across coalitions.

This construction uses **all 24 datasets**, with logistic/linear regression and
random forests. We additionally use XGBoost, LightGBM and multilayer perceptrons
at twelve groups on Adult Census, Breast Cancer, Digits, Bioresponse, Wine
Quality, Miami Housing, Superconductivity and QSAR-TID11. We obtain ground truth
by retraining on every coalition of groups.

The number of players and the number of input features are separate choices:
a twelve-player game can contain twelve groups of rows, each with many columns.
The core uses up to twelve input features. On the same eight comparison datasets,
we hold the group count at twelve and vary input width through **8, 16, 32, 64,
128 and the full dataset width**, where available. This tests whether estimator
behavior changes with the underlying learning problem, even when the number of
players stays fixed.

To study much larger player sets, we also use **individual training examples**
as players in nearest-neighbor games. These cover all twelve classification
datasets below, with **32, 64, 128, 256, 512 or 1,024 examples** where enough rows
are available. A coalition's utility depends on its neighbors' labels around a
held-out example, using either a fixed number of nearest neighbors (KNN) or a
distance threshold (TNN). Specialized formulas give exact Shapley values without
training and evaluating all possible subsets. These larger games test Shapley
values, not interaction indices.

## Selecting useful features

Suppose we want to decide which measurements a future model needs. Here,
**features are players**, and a coalition's payoff measures how well a model can
learn using only those features.

For every coalition, we select its columns in both the training and test data,
fit a fresh model, and score its held-out predictions. As in data valuation, the
payoff is classification accuracy or negative mean squared prediction error,
with zero assigned to the empty coalition. This asks how useful a set of features
is for learning across examples; the individual-prediction games above explain
one prediction from an already fitted model.

We use **all 24 datasets** with logistic/linear regression and random forests.
At twelve features, we also compare XGBoost, LightGBM, RBF support-vector machines
and multilayer perceptrons on Adult Census, Breast Cancer, Digits, Bioresponse,
Wine Quality, Miami Housing, Superconductivity and QSAR-TID11.

The requested feature counts are **4, 8, 12, 14 and 16**, bounded by the dataset's
actual width. Ground truth comes from fitting and scoring every coalition. This
is more expensive than querying a fixed predictor, which is why these games
remain small enough for exhaustive evaluation.

## Datasets and variation

We use twelve classification and twelve regression datasets to cover different
sample sizes, input dimensions and prediction problems. The table gives each
loader's full input width, before selecting a game's features.

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

We also include a dataset's natural full width when it has fewer than sixteen
features. Larger feature selections contain the smaller selections from the
same construction seed, so changing dimension does not select an unrelated set
of columns. We never add artificial features to reach a requested player count.
The Wine classification and Wine Quality regression datasets are distinct.
NHANES I uses its supplied survival label as a regression surrogate.

Four construction seeds vary the games' splits and selections. Each estimator
then runs with three random seeds on the same frozen game. The design contains
5,080 intended game instances; the published coverage will show which produced
usable exact references and completed estimator runs.

## Comparing estimators fairly

We compare **22 estimators**, using only the targets each method supports. The
targets include ordinary Shapley values and five interaction definitions—k-SII,
SII, STII, FSII and FBII—through order two. Each estimate is compared with the
exact reference for that same definition, game and player count.

The main computational allowance is the number of coalition evaluations.
For a game with **d players**, we give methods budgets of **0.5, 1, 2, 4, 8, 16,
32, 64 and 128 times d**. For example, a budget of 8d permits 128 evaluations in
a 16-player game. We record actual queries too: a method may use fewer, and a
small game may run out of distinct coalitions.

Ground truth is computed separately and is not provided to the estimator. Small
games use complete payoff tables; large tree and neighbor games use their
matching exact solvers. Exactness is relative to the specified game and subject
to floating-point precision. We do not use a long estimator run as supposed
exact truth for a generic high-dimensional game.

## Measuring error and reading the results

For each run, we compare the estimated contribution of every player—or every
relevant interaction—with its reference value. We sum the squared differences
and divide by the sum of squared reference contributions:

**Normalized error = sum of squared estimation errors / sum of squared reference values.**

This makes errors comparable across games with different payoff scales. **Lower
is better:** zero means an exact match, and one is the error obtained by
predicting zero for every nonzero reference contribution. The empty-coalition
baseline is excluded from this comparison. A game whose reference contributions
are all zero has no defined normalized error.

We also report first-order and pairwise errors separately, so an accurate main
effect cannot conceal poor interaction estimates. Interaction panels with
negligible reference signal are excluded from normalized comparisons.

Overall summaries give the three applications equal weight, then divide weight
among their game subtypes and recipes. Repeated runs share their recipe's weight.
This is application-and-recipe balancing; datasets with more recipes can have
more weight within an application. Weighted means and medians summarize error,
and pairwise ratings compare methods where both have valid results. Lower error
wins those comparisons; higher ratings are better. Success coverage is shown
alongside scores, because failures and missing runs are not zero-error results.

## Compute and reproducibility

The compute budget is equivalent to **1,024 CPU cores running for 24 hours** on
Hopper. The active pool uses AMD EPYC 9754 processors, with one thread per worker
and no GPUs.

We save the games, model settings, selected rows and columns, random seeds and
reference values so each comparison can be reproduced. Games that exceed the
preparation limit or fail model-quality checks are recorded as exclusions.
Cached-game timings describe estimator execution; they are not direct timings
of an application repeatedly training its models from scratch.

When the runs finish, we check the combined results and automatically publish the
new cohort, its actual coverage and reproduction artifacts. Until then, the
previous results remain available.
