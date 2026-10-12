"""
RegressionMSR
=============

Proxy model-based Shapley value approximation using
:class:`~shapiq.approximator.proxy.regressionmsr.RegressionMSR`.
"""

from __future__ import annotations

import numpy as np

from shapiq.approximator import RegressionMSR

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

approximator = RegressionMSR(n=N_PLAYERS, index="SV", random_state=42)
iv = approximator.approximate(BUDGET, game_fun)
print(iv)

# %%
# Force plot
# ----------

iv.plot_force(feature_names=feature_names)

# %%
# Stacked bar plot
# ----------------

iv.plot_stacked_bar(feature_names=feature_names)
