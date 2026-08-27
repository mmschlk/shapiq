"""
PolySHAP
========

Shapley value approximation with interaction-informed polynomial regression
using :class:`~shapiq.approximator.PolySHAP` :footcite:t:`Fumagalli.2026a`.

PolySHAP extends KernelSHAP by fitting a *k-additive* surrogate of the game -- a
polynomial of degree ``max_order`` over the players -- and reading the Shapley
values off that surrogate. ``max_order=1`` reduces to plain KernelSHAP, while
higher orders capture interactions explicitly. When the game's interactions do
not exceed ``max_order``, PolySHAP recovers the exact Shapley values.
"""

from __future__ import annotations

import numpy as np

import shapiq
from shapiq.approximator import PolySHAP

N_PLAYERS = 8
BUDGET = 200
feature_names = [f"x{i}" for i in range(N_PLAYERS)]

weights = np.array([0.4, 0.3, 0.2, 0.1, 0.05, -0.1, -0.2, -0.3])


def game_fun(coalitions: np.ndarray) -> np.ndarray:
    # A linear game plus a single pairwise interaction between x0 and x1.
    # This game is exactly 2-additive: its interactions never exceed order 2.
    coalitions = np.atleast_2d(coalitions)
    return (coalitions @ weights) + 0.5 * coalitions[:, 0] * coalitions[:, 1]


# %%
# Approximate Shapley values
# --------------------------
# The game has a pairwise interaction, so we fit a second-order (``max_order=2``)
# polynomial surrogate to recover the Shapley values.

approximator = PolySHAP(n=N_PLAYERS, max_order=2, random_state=42)
iv = approximator.approximate(BUDGET, game_fun)
print(iv)

# %%
# Force plot
# ----------

iv.plot_force(feature_names=feature_names)

# %%
# Higher order improves accuracy
# ------------------------------
# Because the game is exactly 2-additive, raising ``max_order`` to match its
# interaction order recovers the exact Shapley values, whereas ``max_order=1``
# (equivalent to KernelSHAP) leaves a visible approximation error. We compare
# both against the exact Shapley values from :class:`~shapiq.ExactComputer`.

exact = np.asarray(
    shapiq.ExactComputer(game_fun, n_players=N_PLAYERS)(index="SV", order=1).get_n_order_values(1)
)

for max_order in (1, 2):
    est = np.asarray(
        PolySHAP(n=N_PLAYERS, max_order=max_order, random_state=42)
        .approximate(BUDGET, game_fun)
        .get_n_order_values(1)
    )
    mse = float(np.mean((est - exact) ** 2))
    print(f"max_order={max_order}: MSE vs. exact Shapley = {mse:.2e}")

# %%
# References
# ----------
# .. footbibliography::
