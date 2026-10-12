"""
Path-Dependent versus Interventional Tree Explanations
======================================================

:class:`~shapiq.explainer.TreeExplainer` computes Shapley values and Shapley interactions of
tree ensembles exactly, in two modes that differ in what it means for a feature to be absent.

Both modes go back to TreeSHAP :footcite:p:`Lundberg.2020`:

- ``mode="pathdependent"`` (the default): at a split on an absent feature, the explanation follows
  both children, weighted by the share of training samples that went each way. The distribution
  of the data is the one stored in the trees, so no data is needed. The values are computed with
  Quadrature-TreeSHAP :footcite:p:`Wettenstein.2026a` (:class:`~shapiq.tree.QuadratureTreeSHAP`),
  which yields any-order interactions as introduced by TreeSHAP-IQ :footcite:p:`Muschalik.2024a`.
- ``mode="interventional"``: an absent feature takes its values from the rows of a
  ``reference_dataset``, and the prediction is averaged over them
  (:class:`~shapiq.tree.InterventionalTreeSHAPIQ`). This mode also supports the Shapley-Taylor
  (STII) :footcite:p:`Sundararajan.2020` and Faithful Shapley (FSII) :footcite:p:`Tsai.2022`
  interaction indices.

This example computes second-order interactions in both modes, checks them against brute force,
and shows where the two modes differ.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import shapiq
from shapiq_games import InterventionalTreeGame, PathDependentTreeGame

# %%
# Data with a Known Interaction and a Correlated Copy
# ---------------------------------------------------
# The Friedman #1 function :math:`y = 10 \sin(\pi x_0 x_1) + 20 (x_2 - 0.5)^2 + 10 x_3 + 5 x_4 +
# \varepsilon` has uniform features, one interaction (:math:`x_0` and :math:`x_1`), and a feature
# :math:`x_5` it does not use. We add :math:`x_6`, a noisy copy of :math:`x_3` that the function
# does not use either. A random forest learns from both copies.

X, y = make_friedman1(n_samples=2_000, n_features=6, noise=0.5, random_state=0)
rng = np.random.default_rng(0)
X = np.column_stack([X, X[:, 3] + rng.normal(scale=0.05, size=len(X))])
feature_names = [f"x{i}" for i in range(X.shape[1])]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = RandomForestRegressor(
    n_estimators=50, max_depth=8, max_features=0.5, random_state=0, n_jobs=1
).fit(X_train, y_train)
print(f"Test R2: {model.score(X_test, y_test):.3f}")
print(f"Correlation of x3 and x6: {np.corrcoef(X[:, 3], X[:, 6])[0, 1]:.3f}")

x = X_test[2]
prediction = model.predict(x.reshape(1, -1))[0]
print(f"Prediction for x: {prediction:.4f}")

# %%
# Path-Dependent k-SII
# --------------------
# We explain one prediction with the pairwise k-SII :footcite:p:`Bordt.2023`, which splits the
# prediction into main effects and pairwise interactions. The explanation starts at the baseline,
# the empty interaction ``()``: the average prediction under the training data in the trees. With
# it, all values sum up to the prediction (efficiency).

path_dependent = shapiq.TreeExplainer(model, index="k-SII", max_order=2).explain(x)
print(f"Baseline: {path_dependent.baseline_value:.4f}")
print(f"Sum of all values: {path_dependent.values.sum():.4f}")

# %%
# The cooperative game behind the explanation is :class:`~shapiq_games.PathDependentTreeGame`.
# With only seven features, :class:`~shapiq.game_theory.ExactComputer` evaluates it on all
# :math:`2^7` coalitions and computes the k-SII from its definition:

game = PathDependentTreeGame(model, x=x, normalize=False)
brute_force = shapiq.ExactComputer(game)(index="k-SII", order=2)
difference = max(abs(path_dependent[s] - brute_force[s]) for s in brute_force.interaction_lookup)
print(f"Max. absolute difference to brute force: {difference:.1e}")

# %%
# Interventional k-SII, STII, and FSII
# ------------------------------------
# In interventional mode, we pass 100 training rows as the reference data. The baseline is now
# the average prediction on these rows. All three indices are efficient, and each agrees with a
# brute-force computation on :class:`~shapiq_games.InterventionalTreeGame`, whose value for a
# coalition is the average prediction when the coalition's features are taken from ``x`` and the
# others from the reference rows.

reference = X_train[:100]
game = InterventionalTreeGame(model, reference_data=reference, x=x)
computer = shapiq.ExactComputer(game)

interventional = {}
for index in ("k-SII", "STII", "FSII"):
    explainer = shapiq.TreeExplainer(
        model, mode="interventional", reference_dataset=reference, index=index, max_order=2
    )
    interventional[index] = explanation = explainer.explain(x)
    brute_force = computer(index=index, order=2)
    difference = max(abs(explanation[s] - brute_force[s]) for s in brute_force.interaction_lookup)
    print(
        f"{index:>5}: baseline {explanation.baseline_value:.4f}, "
        f"sum {explanation.values.sum():.4f}, max. difference to brute force {difference:.1e}"
    )
print(f"Average prediction on the reference rows: {model.predict(reference).mean():.4f}")

# %%
# Comparing the Explanations
# --------------------------
# All four explanations find the interaction of :math:`x_0` and :math:`x_1`. They differ in how
# they share effects between the orders: STII puts everything that a pair adds beyond its
# features' main effects into the pair, and FSII fits the best additive approximation with main
# effects and pairs. The path-dependent and the interventional k-SII are close for the
# independent features, but differ for :math:`x_3` and its copy :math:`x_6`.

terms = [(i,) for i in range(len(feature_names))] + [(0, 1), (3, 6)]
labels = [" x ".join(feature_names[i] for i in term) for term in terms]
explanations = {"path-dependent k-SII": path_dependent} | {
    f"interventional {index}": explanation for index, explanation in interventional.items()
}

fig, ax = plt.subplots(figsize=(7, 6))
height = 0.8 / len(explanations)
for k, (name, explanation) in enumerate(explanations.items()):
    positions = np.arange(len(terms)) + (k - (len(explanations) - 1) / 2) * height
    ax.barh(positions, [explanation[term] for term in terms], height=height, label=name)
ax.set_yticks(np.arange(len(terms)), labels)
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("attribution")
ax.legend(loc="lower right")
fig.tight_layout()
plt.show()

# %%
# The Modes Disagree on Correlated Features
# -----------------------------------------
# Over 200 test points, the average magnitude of the k-SII values is almost the same in both
# modes for the independent features and the interaction of :math:`x_0` and :math:`x_1`. The
# path-dependent explanations give more weight to :math:`x_3`, its copy :math:`x_6`, and their
# interaction.
#
# The path-dependent game weights the children of a split on an absent feature by the training
# samples that reached it, and these samples already passed the splits on present features above
# it. So a present :math:`x_3` also informs the splits on :math:`x_6`, as in the data. The
# interventional game takes :math:`x_6` from reference rows that are unrelated to the present
# :math:`x_3`, and evaluates the model on combinations of the two copies that do not occur in the
# data. It is "true to the model" rather than "true to the data" :footcite:p:`Chen.2020`.

batch = {
    "path-dependent": shapiq.TreeExplainer(model, index="k-SII", max_order=2),
    "interventional": shapiq.TreeExplainer(
        model, mode="interventional", reference_dataset=reference, index="k-SII", max_order=2
    ),
}
mean_abs = {}
for name, explainer in batch.items():
    explanations_200 = explainer.explain_X(X_test[:200])
    mean_abs[name] = [np.mean([abs(iv[term]) for iv in explanations_200]) for term in terms]

fig, ax = plt.subplots(figsize=(8, 3.5))
width = 0.4
for k, (name, values) in enumerate(mean_abs.items()):
    ax.bar(np.arange(len(terms)) + (k - 0.5) * width, values, width=width, label=name)
ax.set_xticks(np.arange(len(terms)), labels)
ax.set_ylabel("mean |k-SII|")
ax.legend()
fig.tight_layout()
plt.show()

# %%
# Explaining a Difference between Two Predictions
# -----------------------------------------------
# The reference data of the interventional mode sets the question an explanation answers. With a
# single reference row :math:`z`, the Shapley values explain the difference between the
# predictions for :math:`x` and :math:`z` (baseline Shapley values). Here, :math:`z` is the test
# point with the lowest prediction.

z = X_test[np.argmin(model.predict(X_test))]
contrastive = shapiq.TreeExplainer(
    model, mode="interventional", reference_dataset=z.reshape(1, -1), index="SV"
).explain(x)
print(f"f(x) - f(z) = {prediction - model.predict(z.reshape(1, -1))[0]:.4f}")
print(f"Sum of the Shapley values: {contrastive.get_n_order_values(1).sum():.4f}")
contrastive.plot_waterfall(feature_names=feature_names, abbreviate=False)

# %%
# Which Mode to Use
# -----------------
# - **Path-dependent** needs no data and its cost does not depend on a reference set. It follows
#   the training distribution stored in the trees, so correlated features share credit as they
#   occur together in the data. It supports the SV, SII, k-SII, BV, and BII indices.
# - **Interventional** explains the model's function: a feature receives credit only through the
#   model's response to it, and the reference data chooses the comparison (a population, a group,
#   or a single point). It supports all indices of the explainer, including STII, FSII, and
#   FBII. Its cost grows with the size of the reference data; for large inputs, the explainer
#   switches to the Woodelf algorithm :footcite:p:`Nadel.2026` if the optional
#   ``woodelf-explainer`` package is installed.
#
# References
# ----------
# .. footbibliography::
