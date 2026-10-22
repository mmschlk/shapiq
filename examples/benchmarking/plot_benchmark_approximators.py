"""
Benchmarking Approximators
==========================

Which approximator should you use for your game, and how many evaluations does it need?
:mod:`shapiq_benchmark` answers this empirically: a :class:`~shapiq_benchmark.Benchmark` holds a
game and its exact values, and :func:`~shapiq_benchmark.run` runs approximators for several
budgets and seeds and scores every estimate against the exact values. This example compares five
approximators of the Shapley value on the explanation of a random forest's prediction and shows
how to read the results.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

import shapiq
from shapiq_benchmark import Benchmark, run, save_results
from shapiq_games import InterventionalTreeGame

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 12)

# %%
# The Game and Its Exact Values
# -----------------------------
# We explain the prediction of a random forest for one patient of the diabetes data (ten
# features) with the interventional game: a coalition is worth the average prediction when the
# features in the coalition are taken from the patient and the others from 50 reference
# patients. With ten players, the game has :math:`2^{10} = 1024` coalitions.
#
# Evaluating the model is the expensive part of an approximation. For a benchmark of a small game,
# :meth:`~shapiq.game.Game.precompute` evaluates the game once on all coalitions, and the
# approximators then look up the values instead of calling the model. (The runtimes in the results
# then measure the approximators' own work, not the cost of the game.)

X, y = load_diabetes(return_X_y=True)
model = RandomForestRegressor(n_estimators=20, max_depth=6, random_state=0).fit(X, y)
game = InterventionalTreeGame(model, reference_data=X[:50], x=X[0])
game.precompute()

benchmark = Benchmark(game)
print(benchmark)
ground_truth = benchmark.exact_values(index="SV", order=1)
print(ground_truth)

# %%
# The benchmark computes the exact values with interventional TreeSHAP-IQ from the trees of the
# model, without evaluating the game.
#
# Running the Approximators
# -------------------------
# :func:`~shapiq_benchmark.run` builds every approximator anew for every budget and seed (the
# seed is passed as ``random_state``), for the number of players of the game and the requested
# index and order, and scores its estimate. The result is a ``pandas.DataFrame`` with one
# row per approximator, budget, and seed.

APPROXIMATORS = {
    "KernelSHAP": shapiq.KernelSHAP,
    "LeverageSHAP": shapiq.LeverageSHAP,
    "OddSHAP": shapiq.OddSHAP,
    "PermutationSamplingSV": shapiq.PermutationSamplingSV,
    "SVARM": shapiq.SVARM,
}
BUDGETS = [40, 80, 160, 320, 640]
results = run(
    benchmark,
    APPROXIMATORS,
    budgets=BUDGETS,
    index="SV",
    order=1,
    seeds=range(10),
    k=3,
    with_faithfulness=True,
)
print(results.shape)
print(results.columns.tolist())
print(
    results[["approximator", "budget", "seed", "status", "runtime_s", "mse", "kendall_tau"]].head()
)

# %%
# The Metrics
# -----------
# Every estimate is compared with the ground truth on the interactions of order 1 to ``order``
# (here: the ten Shapley values):
#
# - **Errors** (lower is better): the mean squared and absolute errors ``mse`` and ``mae``, their
#   sums ``sse`` and ``sae``, and ``nmse``, the squared error relative to the squared norm of the
#   ground truth (comparable across games).
# - **Rankings** (higher is better): Kendall's tau and Spearman's rho of all values, and, for the
#   ``k`` values that are largest in absolute value (``k=3`` here), the precision
#   ``precision_at_k`` (how many of the ``k`` largest estimates are among the ``k`` largest
#   exact values) and Kendall's tau ``kendall_tau_at_k``.
# - **Faithfulness** (higher is better, with ``with_faithfulness=True``): the :math:`R^2` of the
#   game values rebuilt from the estimate on sampled coalitions. An explanation of order 1 is an
#   additive model of the game, so its faithfulness is bounded by how additive the game is.
#
# Rows whose ``status`` is not ``"ok"`` record unsupported combinations or failures (with the
# ``error``) instead of dropping them.

for metric, number_format in (("mse", "{:.2e}"), ("faithfulness", "{:.3f}")):
    table = results.pivot_table(index="approximator", columns="budget", values=metric)
    print(f"Mean {metric} over the seeds:")
    print(table.map(number_format.format))
print(results["status"].value_counts())

# %%
# Error Versus Budget
# -------------------
# We plot the mean and the spread (the 25% to 75% quantiles over the ten seeds) of the squared
# error and of Kendall's tau against the budget.

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
MARKERS = ["o", "s", "D", "^", "v"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for (name, group), color, marker in zip(
    results.groupby("approximator", sort=False), COLORS, MARKERS, strict=True
):
    by_budget = group.groupby("budget")
    for ax, metric in zip(axes, ["mse", "kendall_tau"], strict=True):
        ax.plot(BUDGETS, by_budget[metric].mean(), marker=marker, color=color, lw=2, label=name)
        ax.fill_between(
            BUDGETS,
            by_budget[metric].quantile(0.25),
            by_budget[metric].quantile(0.75),
            color=color,
            alpha=0.15,
            linewidth=0,
        )
axes[0].set_yscale("log")
axes[0].set_ylabel("Mean squared error (lower is better)")
axes[1].set_ylabel("Kendall's tau (higher is better)")
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS, [str(budget) for budget in BUDGETS])
    ax.set_xlabel("Budget (game evaluations)")
    ax.spines[["top", "right"]].set_visible(False)
axes[1].legend(frameon=False, loc="lower right")
fig.suptitle("Shapley values of a random forest's prediction (10 features, 10 seeds)")
plt.tight_layout()
plt.show()

# %%
# On this game, LeverageSHAP is the most accurate approximator for the smallest budgets and
# OddSHAP from 160 evaluations on (it needs about ten evaluations per player, see its example).
# Both recover the ranking of the features almost perfectly from 320 evaluations on. SVARM needs
# larger budgets on this game. Such results depend on the game, so benchmark the approximators on
# games like yours.
#
# Interactions and Unsupported Combinations
# -----------------------------------------
# For interactions, set ``index`` and ``order``. Approximators that cannot compute the index
# (LeverageSHAP only computes Shapley values) are recorded as ``"unsupported"``. Some Monte Carlo
# approximators estimate only the highest order of some indices (e.g. SHAP-IQ and SVARM-IQ for
# FSII); ``scored_orders`` tells which orders an estimate was scored on, so compare rows with the
# same ``scored_orders``.

interaction_results = run(
    benchmark,
    [shapiq.KernelSHAPIQ, shapiq.SVARMIQ, shapiq.PermutationSamplingSII, shapiq.LeverageSHAP],
    budgets=[200, 600],
    index="k-SII",
    order=2,
    seeds=range(3),
)
print(
    interaction_results.groupby(["approximator", "budget"]).agg(
        status=("status", "first"),
        scored_orders=("scored_orders", "first"),
        mse=("mse", "mean"),
        precision_at_k=("precision_at_k", "mean"),
    )
)

# %%
# Saving the Results
# ------------------
# :func:`~shapiq_benchmark.save_results` writes the results as CSV (for a ``.csv`` path) or as
# JSON records, e.g. to compare them with later runs.

with tempfile.TemporaryDirectory() as directory:
    path = save_results(results, Path(directory) / "results.csv")
    print(pd.read_csv(path).shape)
