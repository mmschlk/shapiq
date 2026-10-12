"""
LeverageSHAP
============

Regression-based Shapley value approximation with leverage score sampling using
:class:`~shapiq.approximator.LeverageSHAP` :footcite:p:`Musco.2025`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import shapiq

N_PLAYERS = 8
BUDGET = 200
feature_names = [f"x{i}" for i in range(N_PLAYERS)]

weights = np.array([0.4, 0.3, 0.2, 0.1, 0.05, -0.1, -0.2, -0.3])


def game_fun(coalitions: np.ndarray) -> np.ndarray:
    coalitions = np.atleast_2d(coalitions)
    return (coalitions @ weights) + 0.5 * coalitions[:, 0] * coalitions[:, 1]


# %%
# Approximate Shapley values
# --------------------------

approximator = shapiq.LeverageSHAP(n=N_PLAYERS, random_state=42)
iv = approximator.approximate(BUDGET, game_fun)
print(iv)

# %%
# Compare with the exact values
# -----------------------------
# The game has only :math:`2^8 = 256` coalitions, so :class:`~shapiq.game_theory.ExactComputer`
# computes its exact Shapley values.

exact = shapiq.ExactComputer(game_fun, n_players=N_PLAYERS)(index="SV", order=1)
error = np.abs(iv.get_n_order_values(1) - exact.get_n_order_values(1))
print(f"Largest absolute error: {error.max():.1e}")

# %%
# Visualize the result
# --------------------

iv.plot_force(feature_names=feature_names)

# %%
# Leverage scores and paired sampling
# -----------------------------------
# Like :class:`~shapiq.approximator.KernelSHAP`, LeverageSHAP fits the Shapley values by weighted
# least squares on sampled coalitions. It differs in how it samples them:
#
# - Coalitions are sampled proportionally to their *leverage scores* in the regression problem,
#   which are :math:`1 / \binom{n}{|S|}`: every coalition size receives the same share of the
#   budget (sizes with few coalitions are evaluated completely).
# - Coalitions are sampled without replacement and in pairs: every sampled coalition comes with
#   its complement.
#
# With this sampling, the estimates are provably accurate with high probability from
# :math:`O(n \log n)` evaluations of the game :footcite:p:`Musco.2025`.
#
# The estimate above is exact up to rounding. The Shapley value only depends on the *odd* part
# of a game, and paired samples remove the even part from the regression
# :footcite:p:`Fumagalli.2026a`; a game whose interactions involve at most two players, such as
# ours, has a linear odd part, which the regression recovers exactly. To see the difference in
# the sampling, we add an interaction of three players and compare the mean squared error of
# LeverageSHAP and KernelSHAP, each with and without paired sampling, over 20 random seeds.


def game_fun_three(coalitions: np.ndarray) -> np.ndarray:
    coalitions = np.atleast_2d(coalitions)
    return game_fun(coalitions) + coalitions[:, 2] * coalitions[:, 3] * coalitions[:, 4]


exact_three = shapiq.ExactComputer(game_fun_three, n_players=N_PLAYERS)(index="SV", order=1)
budgets = [30, 60, 100, 150, 200]
methods = {
    "KernelSHAP": lambda seed: shapiq.KernelSHAP(n=N_PLAYERS, random_state=seed),
    "KernelSHAP, paired": lambda seed: shapiq.KernelSHAP(
        n=N_PLAYERS, pairing_trick=True, random_state=seed
    ),
    "LeverageSHAP, unpaired": lambda seed: shapiq.LeverageSHAP(
        n=N_PLAYERS, pairing_trick=False, random_state=seed
    ),
    "LeverageSHAP": lambda seed: shapiq.LeverageSHAP(n=N_PLAYERS, random_state=seed),
}
mse = {name: [] for name in methods}
for name, build in methods.items():
    for budget in budgets:
        errors = [
            np.mean(
                (
                    build(seed).approximate(budget, game_fun_three).get_n_order_values(1)
                    - exact_three.get_n_order_values(1)
                )
                ** 2
            )
            for seed in range(20)
        ]
        mse[name].append(np.mean(errors))
for name, values in mse.items():
    print(f"{name:>22}: " + ", ".join(f"{value:.1e}" for value in values))

fig, ax = plt.subplots(figsize=(6, 4))
for (name, values), color, marker in zip(
    mse.items(), ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"], ["o", "s", "^", "D"], strict=True
):
    ax.plot(budgets, values, marker=marker, color=color, lw=2, label=name)
ax.set_yscale("log")
ax.set_xlabel("Budget (game evaluations)")
ax.set_ylabel("Mean squared error (mean of 20 seeds)")
ax.set_title("Game with an interaction of three players")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

# %%
# On this game, the pairing reduces the error the most, for both methods. With paired sampling
# on both sides, the leverage scores still make LeverageSHAP more accurate than KernelSHAP from a
# budget of 100 on.

# %%
# References
# ----------
# .. footbibliography::
