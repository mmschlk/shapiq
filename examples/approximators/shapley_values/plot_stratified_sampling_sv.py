"""
StratifiedSamplingSV
====================

Estimate Shapley values with :class:`~shapiq.approximator.StratifiedSamplingSV`: it samples marginal
contributions of every player per coalition size (a stratum) and averages over the strata.
"""

# sphinx_gallery_thumbnail_path = "_static/gallery/approximators_stratified.png"  # noqa: ERA001

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

approximator = shapiq.StratifiedSamplingSV(n=N_PLAYERS, random_state=42)
iv = approximator.approximate(BUDGET, game_fun)
print(iv)

# %%
# Visualize the result
# --------------------

iv.plot_force(feature_names=feature_names)
