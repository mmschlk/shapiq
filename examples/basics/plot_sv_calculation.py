"""
Computing Shapley Values
=========================

This example introduces cooperative game theory and shows how to compute
Shapley values with :mod:`shapiq` -- both exactly and via approximation --
and how to apply them for explainable AI (XAI).
"""

from __future__ import annotations

import numpy as np

import shapiq

# %%
# The Cooking Game
# ----------------
# Three cooks (Alice, Bob, Charlie) prepare a meal together. We model their
# joint productivity as a cooperative game and compute exact Shapley values.


class CookingGame(shapiq.Game):
    """Cooking game with three cooks."""

    def __init__(self) -> None:
        self.characteristic_function = {
            (): 0,
            (0,): 4,
            (1,): 3,
            (2,): 2,
            (0, 1): 9,
            (0, 2): 8,
            (1, 2): 7,
            (0, 1, 2): 15,
        }
        super().__init__(
            n_players=3,
            player_names=["Alice", "Bob", "Charlie"],
            normalization_value=self.characteristic_function[()],
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        return np.array([self.characteristic_function[tuple(np.where(c)[0])] for c in coalitions])


cooking_game = CookingGame()

# %%
# Exact Shapley Values
# ---------------------
# The :class:`~shapiq.ExactComputer` evaluates all :math:`2^n` coalitions.

exact_computer = shapiq.ExactComputer(n_players=cooking_game.n_players, game=cooking_game)
sv_exact = exact_computer(index="SV")
print(sv_exact)

sv_exact.plot_stacked_bar(
    xlabel="Cooks",
    ylabel="Shapley Values",
    feature_names=["Alice", "Bob", "Charlie"],
)

# %%
# Approximating Shapley Values
# -----------------------------
# For larger games, exact computation is infeasible. Here we define a
# 10-player restaurant game and approximate Shapley values with
# :class:`~shapiq.KernelSHAP`.

rng = np.random.default_rng(42)
quality_dict = {cooks: rng.random() * len(cooks) for cooks in shapiq.powerset(range(10))}


def restaurant_value_function(coalitions: np.ndarray) -> np.ndarray:
    return np.array([quality_dict[tuple(np.where(c)[0])] for c in coalitions])


approx = shapiq.KernelSHAP(n=10, random_state=42)
sv_approx = approx(game=restaurant_value_function, budget=100)
print(sv_approx)

sv_approx.plot_stacked_bar(
    xlabel="Cooks",
    ylabel="Shapley Values",
    feature_names=[f"Cook {i}" for i in range(10)],
)

# %%
# XAI with Shapley Values
# ------------------------
# We train a Random Forest on the California housing dataset and explain a
# single prediction using :class:`~shapiq.TabularExplainer`.

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data, targets = shapiq.datasets.load_california_housing()
feature_names = list(data.columns)
n_features = len(feature_names)

x_train, x_test, y_train, y_test = train_test_split(
    data.values,
    targets.values,
    test_size=0.2,
    random_state=42,
)
rf = RandomForestRegressor(n_estimators=30, random_state=42)
rf.fit(x_train, y_train)
print(f"Test R2: {rf.score(x_test, y_test):.4f}")

# %%
# Explain a Single Prediction
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

x_explain = x_test[2]
y_pred = rf.predict([x_explain])[0]
print(f"Predicted: {y_pred:.3f}, Average: {np.mean(rf.predict(x_test)):.3f}")

explainer = shapiq.TabularExplainer(
    model=rf,
    data=x_test,
    imputer="marginal",
    index="SV",
    max_order=1,
    sample_size=100,
    random_state=42,
)
sv = explainer.explain(x_explain, budget=2**n_features)
print(sv)

sv.plot_force(feature_names=feature_names)

# %%
# TreeExplainer for Exact Tree-based SV
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For tree models, :class:`~shapiq.TreeExplainer` computes exact Shapley values
# in linear time.

tree_explainer = shapiq.TreeExplainer(model=rf, index="SV", max_order=1)
sv_tree = tree_explainer.explain(x_explain)
print(sv_tree)

sv_tree.plot_force(feature_names=feature_names)
