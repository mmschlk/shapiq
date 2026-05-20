"""
SVARM
=====

Stratified SVARM for Shapley value approximation using
:class:`~shapiq.SVARM` :footcite:t:`Kolpaczki.2024a`.
"""

from __future__ import annotations

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

approximator = shapiq.SVARM(n=N_PLAYERS, random_state=42)
iv = approximator.approximate(BUDGET, game_fun)
print(iv)

# %%
# Visualize the result
# --------------------

iv.plot_force(feature_names=feature_names)

# %%
# References
# ----------
# .. footbibliography::
