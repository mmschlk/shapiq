"""
ExactComputer
=============

Exact computation of Shapley values and interaction indices for small games
using :class:`~shapiq.ExactComputer`.
"""

from __future__ import annotations

import numpy as np

import shapiq

N_PLAYERS = 8
feature_names = [f"x{i}" for i in range(N_PLAYERS)]

weights = np.array([0.4, 0.3, 0.2, 0.1, 0.05, -0.1, -0.2, -0.3])


def game_fun(coalitions: np.ndarray) -> np.ndarray:
    coalitions = np.atleast_2d(coalitions)
    return (coalitions @ weights) + 0.5 * coalitions[:, 0] * coalitions[:, 1]


# %%
# Compute exact Shapley values
# ----------------------------

computer = shapiq.ExactComputer(game_fun, n_players=N_PLAYERS)
sv = computer(index="SV", order=1)
print(sv)

# %%
# Force plot of Shapley values
# ----------------------------

sv.plot_force(feature_names=feature_names)

# %%
# Compute exact k-SII values
# ---------------------------

ksii = computer(index="k-SII", order=2)
print(ksii)

# %%
# Force plot of k-SII values
# --------------------------

ksii.plot_force(feature_names=feature_names)

# %%
# Network plot of k-SII values
# ----------------------------

ksii.plot_network(feature_names=feature_names)
