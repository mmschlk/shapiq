"""
OddSHAP
=======

Shapley value approximation by a sparse regression on the odd part of the game using
:class:`~shapiq.approximator.OddSHAP` :footcite:p:`Fumagalli.2026a`.
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

approximator = shapiq.OddSHAP(n=N_PLAYERS, random_state=42)
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

iv.plot_force(feature_names=feature_names, abbreviate=False)

# %%
# Only the odd part matters
# -------------------------
# Every game can be written as a sum of parity functions (its Fourier representation), and it
# splits into an *odd* part, the parities of an odd number of players, and an *even* part. The
# Shapley value only depends on the odd part :footcite:p:`Fumagalli.2026a`. OddSHAP therefore
#
# 1. samples coalitions in pairs (every coalition with its complement), which removes the even
#    part from the regression,
# 2. fits a tree ensemble (LightGBM) to the sampled game values and reads the most important odd
#    interactions off its exact Fourier spectrum, and
# 3. solves a weighted least-squares problem on the selected odd terms: all players and up to
#    ``ceil(budget / interaction_factor) - n`` interactions of three, five, ... players (with the
#    default ``interaction_factor=10``, 12 interactions for our budget of 200).
#
# The game above only has an interaction of two players, so its odd part is linear and the
# estimate is exact. We add an interaction of three players, which is odd, and look at the
# interactions that OddSHAP selects.


def game_fun_three(coalitions: np.ndarray) -> np.ndarray:
    coalitions = np.atleast_2d(coalitions)
    return game_fun(coalitions) + coalitions[:, 2] * coalitions[:, 3] * coalitions[:, 4]


exact_three = shapiq.ExactComputer(game_fun_three, n_players=N_PLAYERS)(index="SV", order=1)
approximator = shapiq.OddSHAP(n=N_PLAYERS, random_state=42)
iv_three = approximator.approximate(BUDGET, game_fun_three)
selected = [
    tuple(int(player) for player in interaction)
    for interaction in approximator.odd_interaction_lookup
    if len(interaction) > 1
]
print("Selected interactions of three players:", selected)
print("The interaction (2, 3, 4) was found:", (2, 3, 4) in selected)
error = np.abs(iv_three.get_n_order_values(1) - exact_three.get_n_order_values(1))
print(f"Largest absolute error: {error.max():.1e}")

# %%
# The odd part of this game is the linear part plus the interaction of players 2, 3, and 4.
# OddSHAP selects it and the estimate is exact again. A regression on the players alone, such as
# :class:`~shapiq.approximator.LeverageSHAP` (which also samples in pairs), cannot represent it. We
# compare both over budgets and 10 random seeds.

budgets = [40, 60, 80, 100, 120, 160, 200]
methods = {"LeverageSHAP": shapiq.LeverageSHAP, "OddSHAP": shapiq.OddSHAP}
mse = {name: [] for name in methods}
for name, approximator_class in methods.items():
    for budget in budgets:
        errors = [
            np.mean(
                (
                    approximator_class(n=N_PLAYERS, random_state=seed)
                    .approximate(budget, game_fun_three)
                    .get_n_order_values(1)
                    - exact_three.get_n_order_values(1)
                )
                ** 2
            )
            for seed in range(10)
        ]
        mse[name].append(np.mean(errors))

for name, values in mse.items():
    print(f"{name:>12}: " + ", ".join(f"{value:.1e}" for value in values))

FLOOR = 1e-12  # errors below are rounding errors; drawn at the floor
fig, ax = plt.subplots(figsize=(6, 4))
for (name, values), color, marker in zip(
    mse.items(), ["#2a78d6", "#eb6834"], ["o", "D"], strict=True
):
    ax.plot(budgets, np.maximum(values, FLOOR), marker=marker, color=color, lw=2, label=name)
ax.set_yscale("log")
ax.set_ylim(FLOOR / 10, 1)
ax.axhline(FLOOR, color="#c3c2b7", linestyle=":", lw=1)
ax.text(budgets[0], FLOOR, "exact up to rounding", va="bottom", color="#52514e", fontsize=9)
ax.axvline(10 * N_PLAYERS, color="#898781", linestyle="--", lw=1)
ax.text(10 * N_PLAYERS, 1, " budget = 10 n", va="top", color="#52514e", fontsize=9)
ax.set_xlabel("Budget (game evaluations)")
ax.set_ylabel("Mean squared error (mean of 10 seeds)")
ax.set_title("Game with an interaction of three players")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.show()

# %%
# Below 80 evaluations, OddSHAP is less accurate than LeverageSHAP, and from 80 to 120 about as
# accurate: it has room for all players, but for few interactions of three players, and the
# interaction of players 2, 3, and 4 is often not among them. From a budget of 160 on, OddSHAP
# selects it for (almost) every seed and is exact up to rounding, while the error of
# LeverageSHAP only decreases slowly with the budget.
#
# Small budgets need care: OddSHAP spends one term of its regression per ``interaction_factor``
# evaluations. Below ``interaction_factor * n`` evaluations (the dashed line), it cannot include
# every player, and the players it leaves out get a Shapley value of exactly 0. With a budget of
# 50, only five of our eight players fit:

small = shapiq.OddSHAP(n=N_PLAYERS, random_state=42).approximate(50, game_fun)
print(np.round(small.get_n_order_values(1), 3) + 0.0)  # + 0.0 turns -0.0 into 0.0

# %%
# References
# ----------
# .. footbibliography::
