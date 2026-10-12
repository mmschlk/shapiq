"""
RegressionMSR
=============

Estimate Shapley values with :class:`~shapiq.approximator.RegressionMSR`: fit a surrogate model
(XGBoost by default) to the sampled coalition values, read its Shapley values off exactly, and
correct the surrogate's error with a Monte Carlo estimate on its residuals.
"""

# sphinx_gallery_thumbnail_path = "_static/gallery/approximators_surrogate.png"  # noqa: ERA001

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
