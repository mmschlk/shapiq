"""
A Tour of the Games
===================

:mod:`shapiq_games` collects cooperative games from explainable AI and data valuation, and from
synthetic settings with a known solution. Every game is a :class:`shapiq.Game <shapiq.game.Game>`:
it maps coalitions of players to values, and approximators and exact algorithms of ``shapiq`` take
it as it is. The games are built from plain objects (a model, data, a point to explain), not from
names. This example builds six games, evaluates coalitions, and computes their exact Shapley
values with :class:`~shapiq_benchmark.Benchmark`, which picks the fastest exact algorithm for
each game. Everything runs on data that ships with scikit-learn, so nothing is downloaded.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import shapiq_games as sg
from shapiq_benchmark import Benchmark

# %%
# A Game Is a Function of Coalitions
# ----------------------------------
# A coalition is a row of a boolean matrix with one column per player. The sum of unanimity
# games (:class:`~shapiq_games.SOUM`) is a synthetic game made of random interactions; since it
# is built from its Möbius transform, all of its interaction values are known exactly.

soum = sg.SOUM(
    n=10, n_basis_games=15, min_interaction_size=1, max_interaction_size=3, random_state=0
)
coalitions = np.array(
    [
        np.zeros(10, dtype=bool),  # the empty coalition
        np.arange(10) < 5,  # the first five players
        np.ones(10, dtype=bool),  # the grand coalition
    ]
)
print("Values:", soum(coalitions))
print("Players:", soum.n_players, "| empty:", soum.empty_coalition_value)
print("Möbius values that are not zero:", len(soum.moebius_coefficients.values))

# %%
# Explaining a Prediction
# -----------------------
# :class:`~shapiq_games.TabularLocalExplanation` explains the prediction of a model for one
# point ``x``: the players are the features, and the value of a coalition is the prediction when
# only the features in the coalition are known. The absent features are filled in from
# background data (``imputer="marginal"``; see the imputer example in the technicalities). The
# game is centered: the empty coalition is worth 0, and the grand coalition is worth the
# prediction minus the average prediction.

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
model = RandomForestRegressor(n_estimators=20, max_depth=6, random_state=0).fit(X, y)

local = sg.TabularLocalExplanation(model, data=X[:50], x=X[0], sample_size=50)
print(f"Average prediction {local.empty_value:.1f}, prediction {local.original_model_output:.1f}")
print(f"v(empty) = {local.empty_coalition_value:.1f}, v(N) = {local.grand_coalition_value:.1f}")

# %%
# For tree models, :class:`~shapiq_games.InterventionalTreeGame` defines the same game, but it is
# not centered (the empty coalition is worth the average prediction over the reference data).
# It exists because its exact values are known for any number of features: interventional
# TreeSHAP-IQ computes them without evaluating the game. Up to the centering, the two games agree
# on every coalition.

tree_game = sg.InterventionalTreeGame(model, reference_data=X[:50], x=X[0])
sample = np.random.default_rng(0).random((100, 10)) < 0.5
centered = tree_game(sample) - tree_game.empty_coalition_value
print("Same values up to centering:", np.allclose(centered, local(sample)))

# %%
# Valuing Features, Data, and Neighbors
# -------------------------------------
# Other games ask other questions about a model. We use the ten "mean" features of the breast
# cancer data.
#
# - :class:`~shapiq_games.FeatureSelection`: the players are the features; a coalition is worth
#   the test accuracy of a model trained on these features only (an unfitted model is passed and
#   retrained for every coalition).
# - :class:`~shapiq_games.DataValuation`: the players are training points; a coalition is worth
#   the test accuracy of a model trained on these points only (Data Shapley).
# - :class:`~shapiq_games.KNNGame`: the players are the training points of a fitted
#   k-nearest-neighbor classifier; a coalition is worth the share of its ``k`` nearest
#   neighbors of ``x`` that belong to the explained class.

cancer = load_breast_cancer()
X_c, y_c = cancer.data[:, :10], cancer.target
X_train, X_test, y_train, y_test = train_test_split(
    X_c, y_c, test_size=0.4, stratify=y_c, random_state=0
)

feature_selection = sg.FeatureSelection(
    DecisionTreeClassifier(max_depth=3, random_state=0), X_train, y_train, X_test, y_test
)
data_valuation = sg.DataValuation(
    DecisionTreeClassifier(max_depth=3, random_state=0),
    X_train[:10],
    y_train[:10],
    X_test,
    y_test,
)
knn = KNeighborsClassifier(n_neighbors=3).fit(X_train[:12], y_train[:12])
knn_game = sg.KNNGame(knn, x=X_test[0], class_index=int(y_test[0]))

# %%
# The values of every game are deterministic: they depend on the arguments of the game (seeds
# included), not on how often or in which batches the coalitions are evaluated.

games = {
    "SOUM": soum,
    "TabularLocalExplanation": local,
    "InterventionalTreeGame": tree_game,
    "FeatureSelection": feature_selection,
    "DataValuation": data_valuation,
    "KNNGame": knn_game,
}
for name, game in games.items():
    batch = np.random.default_rng(1).random((8, game.n_players)) < 0.5
    one_by_one = np.array([game(coalition[None])[0] for coalition in batch])
    same = np.array_equal(game(batch), one_by_one)
    print(f"{name:>23}: same values one by one and in a batch: {same}")

# %%
# Exact Values with a Benchmark
# -----------------------------
# :class:`~shapiq_benchmark.Benchmark` pairs a game with a computer of its exact values.
# ``Benchmark(game)`` chooses the computer: an exact algorithm for the structure of the game
# where ``shapiq`` has one (the Möbius transform of the synthetic games, interventional TreeSHAP-IQ
# for tree games, the closed form of nearest-neighbor games), and brute force over all
# :math:`2^n` coalitions otherwise (up to 25 players).

exact, computers = {}, {}
for name, game in games.items():
    benchmark = Benchmark(game)
    exact[name] = benchmark.exact_values(index="SV", order=1)
    computers[name] = benchmark.computer_for("SV", 1).name
    print(f"{name:>23}: {game.n_players} players, computer: {computers[name]}")

# %%
# The Shapley values of the six games, one value per player:

player_names = {
    "TabularLocalExplanation": diabetes.feature_names,
    "InterventionalTreeGame": diabetes.feature_names,
    "FeatureSelection": [name.removeprefix("mean ") for name in cancer.feature_names[:10]],
}
players = {"SOUM": "player", "DataValuation": "training point", "KNNGame": "training point"}
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
for ax, (name, values) in zip(axes.flat, exact.items(), strict=True):
    sv = values.get_n_order_values(1)
    labels = player_names.get(name, [str(i) for i in range(len(sv))])
    ax.bar(range(len(sv)), sv, color=np.where(sv >= 0, "#ff0d57", "#1e88e5"))
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.set_xticks(range(len(sv)), labels, rotation=90 if name in player_names else 0, fontsize=8)
    ax.set_xlabel(players.get(name, ""), fontsize=9)
    ax.set_title(f"{name}\n(computer: {computers[name]})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
axes[0, 0].set_ylabel("Shapley value")
axes[1, 0].set_ylabel("Shapley value")
plt.tight_layout()
plt.show()

# %%
# The two explanation games give the same Shapley values: they are the same game. In the feature
# selection game, the Shapley values add up to the accuracy of the model with all ten features
# minus the accuracy without features (predicting the majority class); in the data valuation
# game, they add up to the accuracy of the model trained on all ten points (a model trained on no
# data scores 0 by default, ``empty_value=0``).
#
# The setups of :mod:`shapiq_benchmark.setups` build these games from names (a dataset, a model,
# seeds) for benchmarks, e.g.
# ``TabularLocalExplanationSetup(dataset="breast_cancer", model="xgboost").build()``; they
# download the datasets on first use. The next example benchmarks approximators on a game.
