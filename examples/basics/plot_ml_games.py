"""
Games from Machine Learning Models
==================================

Any machine learning model becomes a cooperative game once we decide what it means for a feature
to be absent. This example turns a scikit-learn random forest into a game by subclassing
:class:`~shapiq.game.Game`, computes the exact Shapley value (SV) and the pairwise Shapley
interaction index (SII), k-Shapley interaction index (k-SII), and Faithful Shapley interaction
index (FSII) with :class:`~shapiq.game_theory.ExactComputer`, and measures how close the
approximators get to these exact values for a few budgets.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

import shapiq

# %%
# Train a Model
# -------------
# A small random forest on the diabetes data of scikit-learn: 442 patients, 10 features, and the
# disease progression one year later as the target.

data = load_diabetes()
X, feature_names = data.data, data.feature_names
model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=0).fit(X, data.target)

# %%
# Write the Game
# --------------
# The players are the features of the point ``x`` we explain. A coalition keeps the features of
# ``x`` it contains and fills the absent ones from the rows of a small background sample. Its value
# is the mean prediction over these rows. :class:`~shapiq.game.Game` subtracts its
# ``normalization_value`` from every value, so we pass the mean prediction on the background: the
# empty coalition is then worth 0.
#
# :meth:`~shapiq.game.Game.value_function` receives a boolean matrix with one row per coalition.
# It imputes the rows of all coalitions at once and calls ``model.predict`` once per batch instead
# of once per coalition, which keeps the game fast. The basics of :class:`~shapiq.game.Game` are in
# :doc:`/auto_examples/basics/plot_custom_games`.


class ModelGame(shapiq.Game):
    """Explains the prediction of ``model`` for ``x``, with absent features from ``background``."""

    def __init__(self, model, x: np.ndarray, background: np.ndarray) -> None:
        self.model, self.x, self.background = model, x, background
        baseline = float(model.predict(background).mean())
        super().__init__(n_players=len(x), normalization_value=baseline)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        n_coalitions, n_rows = len(coalitions), len(self.background)
        # one row per coalition and background row: present features from x, absent from the row
        present = np.repeat(coalitions, n_rows, axis=0)
        imputed = np.where(present, self.x, np.tile(self.background, (n_coalitions, 1)))
        return self.model.predict(imputed).reshape(n_coalitions, n_rows).mean(axis=1)


background = X[np.random.default_rng(0).choice(len(X), size=20, replace=False)]
x = X[0]
game = ModelGame(model, x, background)
print(f"Prediction for x: {model.predict(x.reshape(1, -1))[0]:.2f}")
print(f"Mean prediction on the background: {game.normalization_value:.2f}")
print(f"Value of the grand coalition: {game.grand_coalition_value:.2f}")

# %%
# This class is the blueprint for any model: only ``value_function`` changes. A classifier would
# return the probability of a class, a neural network could zero out absent features, and a
# language model could mask absent words. Exact computation and approximators only ever see a
# function of coalitions.
#
# Exact Values
# ------------
# With 10 players, the game has :math:`2^{10} = 1024` coalitions.
# :meth:`~shapiq.game.Game.precompute` evaluates all of them once and stores their values, so
# the exact computation and every approximator below look them up instead of calling the model.

game.precompute()
exact_computer = shapiq.ExactComputer(game=game)
INDICES = {"SV": 1, "SII": 2, "k-SII": 2, "FSII": 2}  # index: maximum order
exact = {index: exact_computer(index=index, order=order) for index, order in INDICES.items()}

sv = exact["SV"].get_n_order_values(1)
print(", ".join(f"{name}: {value:.1f}" for name, value in zip(feature_names, sv, strict=True)))
print(f"Sum of the Shapley values: {sv.sum():.2f}")  # the value of the grand coalition

# %%
# Approximate Them
# ----------------
# For games with many players the exact values are out of reach, and an approximator estimates
# them from a budget of game evaluations. We use regression approximators:
# :class:`~shapiq.approximator.KernelSHAP` for SV, :class:`~shapiq.approximator.KernelSHAPIQ` for
# SII and k-SII, and :class:`~shapiq.approximator.RegressionFSII` for FSII. The error is the mean
# squared error over all interactions up to the order of the index: the single features for SV,
# and the pairs too for the others. We leave out the empty set: for SII,
# :class:`~shapiq.game_theory.ExactComputer` reports the index's value of the empty set, while
# the approximators store 0 there.

n = game.n_players
approximators = {
    "SV": shapiq.KernelSHAP(n=n),
    "SII": shapiq.KernelSHAPIQ(n=n, max_order=2, index="SII"),
    "k-SII": shapiq.KernelSHAPIQ(n=n, max_order=2, index="k-SII"),
    "FSII": shapiq.RegressionFSII(n=n, max_order=2),
}
BUDGETS, SEEDS = [100, 300, 2**n], [0, 1, 2]


def mse(estimate: shapiq.InteractionValues, exact_values: shapiq.InteractionValues) -> float:
    interactions = shapiq.powerset(range(n), min_size=1, max_size=exact_values.max_order)
    return float(np.mean([(estimate[s] - exact_values[s]) ** 2 for s in interactions]))


def mean_error(index: str, budget: int) -> float:
    errors = []
    for seed in SEEDS:
        approximators[index].set_random_state(seed)
        errors.append(mse(approximators[index].approximate(budget, game), exact[index]))
    return float(np.mean(errors))


errors = {index: [mean_error(index, budget) for budget in BUDGETS] for index in INDICES}
print("budget " + "".join(f"{budget:>10}" for budget in BUDGETS))
for index, values in errors.items():
    print(f"{index:<7}" + "".join(f"{value:>10.1e}" for value in values))

# %%
# The error falls with the budget. With the full budget of 1024 evaluations, the regression
# approximators see every coalition and recover the exact values up to floating-point round-off.
# KernelSHAP-IQ obtains k-SII from its SII estimate, so the errors of SII and k-SII nearly coincide
# (k-SII is dashed).

COLORS, MARKERS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], ["o", "s", "D", "^"]
fig, ax = plt.subplots(figsize=(6.4, 4))
for (index, values), color, marker in zip(errors.items(), COLORS, MARKERS, strict=True):
    linestyle = "--" if index == "k-SII" else "-"
    ax.plot(BUDGETS, values, marker=marker, color=color, lw=2, ls=linestyle, label=index)
ax.set(xscale="log", yscale="log", xlabel="Budget (game evaluations)", ylabel="Mean squared error")
ax.set_xticks(BUDGETS, ["100", "300", "1024\n(all)"])
ax.minorticks_off()
ax.set_title(f"Estimates against exact values (mean over {len(SEEDS)} seeds)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

# %%
# The Shortcut for Tabular Data
# -----------------------------
# For tabular data you do not need to write this game: :class:`~shapiq.imputer.MarginalImputer`
# is the same game, and :class:`~shapiq.explainer.TabularExplainer` builds one under the hood.
# With the same 20 background rows (``sample_size=None`` keeps all of them), it yields the same
# Shapley values:

imputer = shapiq.MarginalImputer(model.predict, data=background, x=x, sample_size=None)
sv_imputer = shapiq.ExactComputer(game=imputer)(index="SV", order=1)
print("Same Shapley values:", np.allclose(sv_imputer.get_n_order_values(1), sv))

# %%
# Write your own game when the model or the removal of features does not fit an imputer.
# Ready-made games of :mod:`shapiq_games` are in
# :doc:`/auto_examples/benchmarking/plot_games_tour`, and
# :doc:`/auto_examples/benchmarking/plot_benchmark_approximators` compares more approximators
# over more budgets and seeds.
