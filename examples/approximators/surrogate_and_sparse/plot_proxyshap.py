"""
ProxySHAP
=========

Estimate Shapley interactions with :class:`~shapiq.approximator.ProxySHAP`: fit a surrogate model
(XGBoost by default) to the sampled coalition values and read the interactions off it exactly; with
``adjustment=True``, a Monte Carlo estimate on the residuals corrects the surrogate's error.
"""

# sphinx_gallery_thumbnail_path = "_static/gallery/approximators_surrogate.png"  # noqa: ERA001

from __future__ import annotations

import numpy as np

from shapiq.approximator import ProxySHAP

N_PLAYERS = 8
BUDGET = 200
feature_names = [f"x{i}" for i in range(N_PLAYERS)]

weights = np.array([0.4, 0.3, 0.2, 0.1, 0.05, -0.1, -0.2, -0.3])


def game_fun(coalitions: np.ndarray) -> np.ndarray:
    coalitions = np.atleast_2d(coalitions)
    return (coalitions @ weights) + 0.5 * coalitions[:, 0] * coalitions[:, 1]


# %%
# Approximate interaction values
# ------------------------------

approximator = ProxySHAP(n=N_PLAYERS, max_order=2, random_state=42)
iv = approximator.approximate(BUDGET, game_fun)
print(iv)

# %%
# Force plot
# ----------

iv.plot_force(feature_names=feature_names)

# %%
# Network plot
# ------------

iv.plot_network(feature_names=feature_names)
