"""
ShaplEIG
========

Estimate Shapley values by Bayesian experimental design: a Gaussian process models the game,
and each next coalition is the one with the largest expected information gain about the Shapley
values.

:class:`~shapiq.approximator.ShaplEIG` :footcite:p:`Rundel.2026` also returns the posterior
variance of every estimate.
"""

# sphinx_gallery_thumbnail_path = "_static/gallery/approximators_surrogate.png"

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import shapiq

N_PLAYERS = 8
BUDGET = 24
feature_names = [f"x{i}" for i in range(N_PLAYERS)]

weights = np.array([0.4, 0.3, 0.2, 0.1, 0.05, -0.1, -0.2, -0.3])


def game_fun(coalitions: np.ndarray) -> np.ndarray:
    coalitions = np.atleast_2d(coalitions)
    return (coalitions @ weights) + 0.5 * coalitions[:, 0] * coalitions[:, 1]


# %%
# Approximate Shapley values
# --------------------------
# ShaplEIG is designed for games whose evaluations are expensive, so we give it a budget of 24
# evaluations instead of the 200 of the other examples. ShaplEIG needs the optional ``shapleig``
# extra (``pip install shapiq[shapleig]``, for PyTorch, GPyTorch, and BoTorch).

approximator = shapiq.ShaplEIG(n=N_PLAYERS, random_state=42, show_progress=False)
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
# Choosing the next coalition and quantifying the uncertainty
# -----------------------------------------------------------
# ShaplEIG does not sample coalitions at random. It fits a Gaussian process to the game values
# evaluated so far (with a kernel on coalitions that compares which players are present). The
# Gaussian process implies a posterior distribution of the Shapley values, and ShaplEIG
# evaluates next the coalition that is expected to reduce the uncertainty about the Shapley
# values the most (the *expected information gain*). It starts from ``n + 1`` coalitions, so the
# budget must exceed 9 here.
#
# Every step refits the Gaussian process, which costs far more than one evaluation of our toy
# game; the approach pays off for games whose evaluations are expensive. In return, ShaplEIG is
# accurate with few evaluations and reports its uncertainty:
# :meth:`~shapiq.approximator.ShaplEIG.approximate_with_variance` returns the posterior variance of
# every Shapley value along with the estimates. We compare the estimates and their 95% credible
# intervals (two posterior standard deviations) after 12 and after 24 evaluations with the exact
# values.

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
players = np.arange(N_PLAYERS)
for ax, budget in zip(axes, [12, BUDGET], strict=True):
    approximator = shapiq.ShaplEIG(n=N_PLAYERS, random_state=42, show_progress=False)
    estimate, variance = approximator.approximate_with_variance(budget=budget, game=game_fun)
    std = np.sqrt(variance)
    covered = np.abs(estimate.get_n_order_values(1) - exact.get_n_order_values(1)) <= 2 * std
    print(
        f"Budget {budget}: largest posterior standard deviation {std.max():.1e}, "
        f"exact values inside the intervals: {covered.sum()} of {N_PLAYERS}"
    )
    ax.errorbar(
        players,
        estimate.get_n_order_values(1),
        yerr=2 * std,
        fmt="o",
        color="#2a78d6",
        capsize=4,
        label="ShaplEIG estimate (95% interval)",
    )
    ax.scatter(
        players, exact.get_n_order_values(1), marker="_", s=300, color="#eb6834", label="Exact"
    )
    ax.axhline(0, color="#c3c2b7", linewidth=1, zorder=0)
    ax.set_xticks(players, feature_names)
    ax.set_title(f"Budget of {budget} evaluations")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Shapley value")
axes[0].legend(frameon=False, loc="lower left")
plt.tight_layout()
plt.show()

# %%
# After 12 evaluations, the intervals are wide, and they contain the exact values. After 24
# evaluations, the estimates are close to the exact values, and the largest posterior standard
# deviation has dropped from about 0.2 to below 0.01.

# %%
# References
# ----------
# .. footbibliography::
