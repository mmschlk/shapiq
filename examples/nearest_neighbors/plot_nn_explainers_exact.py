"""
Exact Data Shapley Values of Three Nearest-Neighbor Classifiers
===============================================================

For a nearest-neighbor classifier, the training points are the players of a cooperative game:
the value of a coalition is how well the classifier, restricted to the coalition as its training
data, predicts the class of a test point.

The Shapley values of this game value each training point for that prediction
:cite:p:`Ghorbani.2019,Jia.2019a`. For general models, they are expensive: a game with :math:`n` training points has :math:`2^n` coalitions. For nearest-neighbor
models, the structure of the prediction yields exact algorithms that are fast enough for large
datasets.

``shapiq`` has three of them, one per classifier of scikit-learn:

- :class:`~shapiq.explainer.KNNExplainer` for :class:`~sklearn.neighbors.KNeighborsClassifier`
  with ``weights="uniform"`` :cite:p:`Jia.2019a`,
- :class:`~shapiq.explainer.WeightedKNNExplainer` for ``weights="distance"``
  :cite:p:`Wang.2024a`,
- :class:`~shapiq.explainer.ThresholdNNExplainer` for
  :class:`~sklearn.neighbors.RadiusNeighborsClassifier` :cite:p:`Wang.2023`.

This example runs all three on a training set small enough to check every result against a
brute-force computation over all :math:`2^{14}` coalitions, with the games of
:mod:`shapiq_games` that define what each explainer computes.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

import shapiq
from shapiq.explainer import KNNExplainer, ThresholdNNExplainer, WeightedKNNExplainer
from shapiq_games import KNNGame, ThresholdNNGame, WeightedKNNGame

# %%
# A Small Training Set
# --------------------
# Fourteen training points of two classes and one test point ``x`` between them. All three models
# predict the class of ``x`` from training points near it: the three nearest ones (``k = 3``),
# weighted by their inverse distance or not, or all points within a radius of 1.2.

X_train, y_train = make_blobs(
    n_samples=14, centers=[[-1.0, 0.0], [1.0, 0.0]], cluster_std=0.8, random_state=1
)
x = np.array([0.1, 0.3])
explained_class = 1

models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Weighted KNN": KNeighborsClassifier(n_neighbors=3, weights="distance"),
    "Threshold NN": RadiusNeighborsClassifier(radius=1.2),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    probability = model.predict_proba(x.reshape(1, -1))[0, explained_class]
    print(f"{name}: P(class {explained_class} | x) = {probability:.3f}")

# %%
# One Explainer per Model
# -----------------------
# Each explainer needs the explained class: a training point is valuable if it supports the
# prediction of that class. :class:`~shapiq.explainer.Explainer` picks the right explainer for a
# nearest-neighbor model by itself; here we create them explicitly.
#
# The weighted explainer rounds the weights of the neighbors to multiples of
# :math:`2^{-b}` with ``n_bits`` :math:`= b` (``3`` by default), which keeps its runtime
# polynomial :cite:p:`Wang.2024a`. Its values are exact for these rounded weights, so the
# game it is checked against uses the same rounding.

explainers = {
    "KNN": KNNExplainer(models["KNN"], class_index=explained_class),
    "Weighted KNN": WeightedKNNExplainer(models["Weighted KNN"], class_index=explained_class),
    "Threshold NN": ThresholdNNExplainer(models["Threshold NN"], class_index=explained_class),
}
games = {
    "KNN": KNNGame(models["KNN"], x=x, class_index=explained_class),
    "Weighted KNN": WeightedKNNGame(
        models["Weighted KNN"],
        x=x,
        class_index=explained_class,
        n_bits=explainers["Weighted KNN"].n_bits,
    ),
    "Threshold NN": ThresholdNNGame(models["Threshold NN"], x=x, class_index=explained_class),
}

# %%
# The games define the value of a coalition :math:`S` of training points:
#
# - **KNN**: the share of the explained class among the :math:`k` nearest neighbors of ``x``
#   in :math:`S` (counted out of :math:`k`, so it is zero for the empty coalition),
# - **Weighted KNN**: one if, among the :math:`k` nearest neighbors in :math:`S`, the explained
#   class has at least the summed weight of the other class, and zero otherwise,
# - **Threshold NN**: the share of the explained class among the points of :math:`S` within the
#   radius, and one over the number of classes if there are none.
#
# Verify against Brute Force
# --------------------------
# :class:`~shapiq.game_theory.ExactComputer` evaluates each game on all :math:`2^{14} = 16{,}384`
# coalitions and computes the Shapley values from their definition. The explainers agree with
# it up to floating-point rounding, and their values sum up to :math:`v(N) - v(\emptyset)`
# (efficiency).

shapley_values = {}
for name, explainer in explainers.items():
    start = time.perf_counter()
    values = explainer.explain(x).get_n_order_values(1)
    explainer_time = time.perf_counter() - start

    game = games[name]
    start = time.perf_counter()
    brute_force = shapiq.ExactComputer(game)(index="SV", order=1).get_n_order_values(1)
    brute_force_time = time.perf_counter() - start

    shapley_values[name] = values
    empty, grand = game(game.empty_coalition)[0], game(game.grand_coalition)[0]
    print(f"{name}:")
    print(f"  max. abs. difference to brute force: {np.max(np.abs(values - brute_force)):.1e}")
    print(f"  sum of Shapley values: {values.sum():.4f}, v(N) - v(empty): {grand - empty:.4f}")
    print(f"  explainer: {explainer_time * 1000:.1f} ms, brute force: {brute_force_time:.2f} s")

# %%
# The rounding of the weights matters: against the game with the exact weights, the values of
# the weighted explainer differ slightly. More bits bring them closer, at a higher cost.

exact_weights_game = WeightedKNNGame(models["Weighted KNN"], x=x, class_index=explained_class)
exact_weights = shapiq.ExactComputer(exact_weights_game)(index="SV", order=1)
difference = np.max(np.abs(shapley_values["Weighted KNN"] - exact_weights.get_n_order_values(1)))
print(f"Max. abs. difference to the game with exact weights: {difference:.3f}")

# %%
# Where the Value Is
# ------------------
# Each panel shows the training points colored by their Shapley value, red for points that
# support the explained class and blue for points that speak against it. Their shape marks the
# class, the star is ``x``.
#
# The three models value the same data differently. For both KNN models, the nearest points of
# the explained class get the highest values. Points farther away still have some value: they
# become one of the :math:`k` nearest neighbors in coalitions without the closer points. In
# the threshold model, only the points within the radius (the dashed circle) can change a
# prediction, so all others get a Shapley value of exactly zero.

red_blue = LinearSegmentedColormap.from_list("red_blue", ["#1e88e5", "#ffffff", "#ff0d57"])
markers = {0: "s", 1: "o"}

fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharex=True, sharey=True)
for ax, (name, values) in zip(axes, shapley_values.items(), strict=True):
    limit = np.max(np.abs(values))  # each game has its own scale
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)
    for label, marker in markers.items():
        is_class = y_train == label
        points = ax.scatter(
            X_train[is_class, 0],
            X_train[is_class, 1],
            c=values[is_class],
            cmap=red_blue,
            norm=norm,
            marker=marker,
            s=140,
            edgecolors="black",
        )
    ax.scatter(*x, marker="*", s=350, color="black")
    if name == "Threshold NN":
        circle = plt.Circle(x, models[name].radius, fill=False, linestyle="--", color="gray")
        ax.add_patch(circle)
    ax.set_title(f"{name} ({type(explainers[name]).__name__})", fontsize=10)
    ax.set_xlabel("feature 1")
    ax.set_aspect("equal")
    fig.colorbar(points, ax=ax, orientation="horizontal", pad=0.18, label="Shapley value")
axes[0].set_ylabel("feature 2")
handles = [
    plt.Line2D([], [], marker=marker, color="gray", linestyle="", label=f"class {label}")
    for label, marker in markers.items()
]
handles.append(
    plt.Line2D([], [], marker="*", color="black", linestyle="", markersize=12, label="x")
)
axes[0].legend(handles=handles, loc="upper left", fontsize=8)
plt.show()

# %%
# The brute-force check is only possible for tiny training sets. The explainers run in
# :math:`O(n \log n)` (KNN, for sorting the points by distance), :math:`O(n^2)` (weighted KNN), and
# :math:`O(n)` (threshold NN) for :math:`n` training points, so they value training sets with
# thousands of points. The example
# :ref:`sphx_glr_auto_examples_nearest_neighbors_plot_nn_noisy_labels.py` uses this to find
# mislabeled training points.
