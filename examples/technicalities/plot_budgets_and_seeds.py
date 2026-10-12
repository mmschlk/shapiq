"""
Budgets, Seeds, and Reproducible Results
========================================

Approximators of ``shapiq`` evaluate the game on a random sample of coalitions. Two arguments
control this: the ``budget``, the number of game evaluations, and the ``random_state``, the seed
of the sampling. This example shows what the budget counts, what happens when it covers all
coalitions, how to get the same estimate twice, and how much an estimate varies with the seed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

import shapiq

N_PLAYERS = 8
weights = np.array([0.4, 0.3, 0.2, 0.1, 0.05, -0.1, -0.2, -0.3])


def game_fun(coalitions: np.ndarray) -> np.ndarray:
    """A linear game with an interaction of two and an interaction of three players."""
    coalitions = np.atleast_2d(coalitions)
    pair = 0.5 * coalitions[:, 0] * coalitions[:, 1]
    triple = coalitions[:, 2] * coalitions[:, 3] * coalitions[:, 4]
    return coalitions @ weights + pair + triple


exact = shapiq.ExactComputer(game_fun, n_players=N_PLAYERS)(index="SV", order=1)

# %%
# The Budget Counts Game Evaluations
# ----------------------------------
# The budget is the number of coalitions an approximator may evaluate, and it is the cost that
# matters: evaluating a coalition usually means calling a model on many imputed points. We wrap
# the game to count the coalitions it is asked for.


class CountingGame:
    """Count the coalitions a game is evaluated on."""

    def __init__(self, game) -> None:
        self.game = game
        self.n_evaluations = 0

    def __call__(self, coalitions: np.ndarray) -> np.ndarray:
        self.n_evaluations += len(np.atleast_2d(coalitions))
        return self.game(coalitions)


for approximator in (
    shapiq.KernelSHAP(n=N_PLAYERS, random_state=0),
    shapiq.PermutationSamplingSV(n=N_PLAYERS, random_state=0),
    shapiq.SVARM(n=N_PLAYERS, random_state=0),
):
    counting_game = CountingGame(game_fun)
    estimate = approximator.approximate(budget=100, game=counting_game)
    print(
        f"{type(approximator).__name__:>21}: {counting_game.n_evaluations} evaluations, "
        f"estimation_budget={estimate.estimation_budget}, estimated={estimate.estimated}"
    )

# %%
# Every approximator stays within the budget. Approximators that evaluate coalitions in fixed
# groups, such as permutation sampling (whole permutations of players), may leave a few
# evaluations of the budget unused.
#
# A Budget of All Coalitions Gives Exact Values
# ---------------------------------------------
# A game with :math:`n` players has :math:`2^n` coalitions, here :math:`2^8 = 256`. With a budget
# of :math:`2^n`, the approximators that sample coalitions evaluate each coalition once and return
# the exact values, marked by ``estimated=False``. Permutation sampling samples permutations, not
# coalitions, and stays an estimate.

for approximator_class in (
    shapiq.KernelSHAP,
    shapiq.LeverageSHAP,
    shapiq.SVARM,
    shapiq.PermutationSamplingSV,
):
    estimate = approximator_class(n=N_PLAYERS, random_state=0).approximate(2**N_PLAYERS, game_fun)
    error = np.abs(estimate.get_n_order_values(1) - exact.get_n_order_values(1)).max()
    print(f"{approximator_class.__name__:>21}: estimated={estimate.estimated}, error {error:.1e}")

# %%
# (KernelSHAP gives the empty and the grand coalition a very large weight in its regression, so
# even with all coalitions, its values are exact only up to about :math:`10^{-7}`.)
#
# Same Seed, Same Estimate
# ------------------------
# The ``random_state`` seeds the random number generator of an approximator when it is created.
# Two approximators with the same seed give the same estimate. Calling the *same* approximator
# again continues its random sequence and gives a different estimate, unless the seed is set
# again with :meth:`~shapiq.approximator.Approximator.set_random_state`.

approximator = shapiq.KernelSHAP(n=N_PLAYERS, random_state=42)
first = approximator.approximate(budget=100, game=game_fun)
second = approximator.approximate(budget=100, game=game_fun)
approximator.set_random_state(42)
third = approximator.approximate(budget=100, game=game_fun)
fresh = shapiq.KernelSHAP(n=N_PLAYERS, random_state=42).approximate(budget=100, game=game_fun)
print("Same approximator, second call: ", np.array_equal(first.values, second.values))
print("After set_random_state(42):     ", np.array_equal(first.values, third.values))
print("New approximator with seed 42:  ", np.array_equal(first.values, fresh.values))

# %%
# The same holds for explainers, which have a second source of randomness: the imputer draws its
# background samples at random, too. The ``random_state`` of an explainer seeds both the imputer
# and the approximator, and ``explain(..., random_state=...)`` seeds them anew for one
# explanation.

X, y = load_diabetes(return_X_y=True)
model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
explainer = shapiq.TabularExplainer(
    model, data=X[:100], index="SV", max_order=1, sample_size=50, random_state=0
)
first = explainer.explain(X[0], budget=100)
second = explainer.explain(X[0], budget=100)
print("Explained twice:              ", np.array_equal(first.values, second.values))
first = explainer.explain(X[0], budget=100, random_state=1)
second = explainer.explain(X[0], budget=100, random_state=1)
print("Explained twice with a seed:  ", np.array_equal(first.values, second.values))

# %%
# To make a result reproducible, pass a seed to every ``explain`` call (or create the
# approximator anew with a seed), and report the seed with the result.
#
# How Much the Seed Matters
# -------------------------
# A single seed shows one estimate; the estimates of many seeds show how much the estimate
# depends on chance. We approximate the Shapley value of player ``x2``, who takes part in the
# interaction of three players, with KernelSHAP for 50 seeds and several budgets.

budgets = [16, 32, 64, 128, 256]
estimates = np.array(
    [
        [
            shapiq.KernelSHAP(n=N_PLAYERS, random_state=seed).approximate(budget, game_fun)[(2,)]
            for seed in range(50)
        ]
        for budget in budgets
    ]
)
for budget, values in zip(budgets, estimates, strict=True):
    print(f"Budget {budget:>3}: mean {values.mean():.3f}, standard deviation {values.std():.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot(
    estimates.T,
    tick_labels=[str(budget) for budget in budgets],
    widths=0.5,
    showfliers=False,  # every estimate is drawn as a point
    medianprops={"color": "#0b0b0b"},
)
jitter = np.random.default_rng(0).uniform(-0.12, 0.12, size=estimates.shape[1])
for position, values in enumerate(estimates, start=1):
    ax.scatter(position + jitter, values, s=10, color="#2a78d6", alpha=0.5, zorder=3)
ax.axhline(exact[(2,)], color="#eb6834", linestyle="--", lw=1.5, label="exact Shapley value")
ax.set_xlabel("Budget (game evaluations)")
ax.set_ylabel("Estimated Shapley value of x2")
ax.set_title("KernelSHAP estimates of one Shapley value for 50 seeds")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

# %%
# The spread over the seeds shrinks as the budget grows, and it vanishes at the budget of
# :math:`2^8 = 256`, where every seed gives the exact value. When you compare approximators or
# budgets, use several seeds: :func:`shapiq_benchmark.run` runs an approximator for a list of
# budgets and seeds and scores every estimate (see the example on benchmarking approximators).
