"""
Kernel SVMs: Exact Shapley Values for Many Features
===================================================

The decision function of a support vector machine (SVM) with an RBF kernel is a weighted sum of
kernels over its support vectors, :math:`f(x) = \\sum_i \\alpha_i k(x^i, x) + b`, and the RBF
kernel is a product of one kernel per feature. For such product-kernel models,
:class:`~shapiq.explainer.product_kernel.ProductKernelExplainer` computes the Shapley values of
the features exactly in polynomial time :footcite:p:`Mohammadi.2025a`.

This example explains an SVM classifier with 30 features, where a brute-force computation would
evaluate :math:`2^{30} \\approx 10^9` coalitions, and compares the exact explainer with
KernelSHAP :footcite:p:`Lundberg.2017`, a model-agnostic approximator, on the same game.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import shapiq
from shapiq.explainer.product_kernel import ProductKernelExplainer
from shapiq_games import ProductKernelGame

# %%
# Train a Kernel SVM
# ------------------
# The breast cancer dataset describes 569 tumors with 30 features of their cell nuclei. Class
# ``1`` is a benign tumor, class ``0`` a malignant one. We standardize the features, since the
# RBF kernel uses one length scale for all of them.

data = load_breast_cancer()
feature_names = list(data.feature_names)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, random_state=0, stratify=data.target
)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

model = SVC(kernel="rbf").fit(X_train, y_train)
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
print(f"Support vectors: {len(model.support_vectors_)}")

# %%
# Explain the Decision Function
# -----------------------------
# The explainer supports binary :class:`~sklearn.svm.SVC` and :class:`~sklearn.svm.SVR` models
# with an RBF kernel. For a classifier, it explains the decision function: positive values speak
# for ``model.classes_[1]`` (benign), negative values for ``model.classes_[0]`` (malignant). To
# explain class ``0``, negate the values. A feature is removed from a coalition by dropping its
# factor from the kernel, so the explanation starts at :math:`v(\emptyset) = \sum_i \alpha_i + b`
# and, by efficiency, ends at the decision function.

x = X_test[3]
explainer = ProductKernelExplainer(model)
start = time.perf_counter()
shapley_values = explainer.explain(x)
exact_time = time.perf_counter() - start
exact = shapley_values.get_n_order_values(1)

print(f"Decision function: {model.decision_function(x.reshape(1, -1))[0]:.6f}")
print(f"Baseline + sum of Shapley values: {shapley_values.baseline_value + exact.sum():.6f}")
print(f"Time of the exact explanation: {exact_time * 1000:.1f} ms")

shapley_values.plot_waterfall(feature_names=feature_names, max_display=10, abbreviate=False)

# %%
# The Same Game, Approximated
# ---------------------------
# :class:`~shapiq_games.ProductKernelGame` is the cooperative game behind the explainer: its value
# for a coalition is the decision function with the kernel restricted to the coalition's features.
# A model-agnostic approximator only sees this game as a black box and estimates the Shapley values
# from a budget of game evaluations. Since the exact values are known, we can measure how far
# :class:`~shapiq.approximator.KernelSHAP` is from them for growing budgets.

game = ProductKernelGame(model, x=x)
scale = np.mean(np.abs(exact))  # the average magnitude of the exact values
budgets = [250, 500, 1_000, 2_000, 4_000, 8_000, 16_000]
relative_errors = []
for budget in budgets:
    approximator = shapiq.KernelSHAP(n=game.n_players, random_state=0)
    start = time.perf_counter()
    estimate = approximator.approximate(budget, game)
    elapsed = time.perf_counter() - start
    relative_errors.append(np.mean(np.abs(estimate.get_n_order_values(1) - exact)) / scale)
    print(f"Budget {budget:>6}: relative error {relative_errors[-1]:.1%} ({elapsed:.2f} s)")

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogx(budgets, 100 * np.array(relative_errors), marker="o", label="KernelSHAP")
ax.axhline(0, color="gray", linestyle="--", label="ProductKernelExplainer (exact)")
ax.set_xlabel("budget (game evaluations)")
ax.set_ylabel("mean abs. error / mean |Shapley value| [%]")
ax.set_title(f"30 features; the exact explainer took {exact_time * 1000:.0f} ms")
ax.legend()
fig.tight_layout()
plt.show()

# %%
# The error of KernelSHAP shrinks roughly with the square root of the budget: with 16,000 game
# evaluations, its estimates still deviate by about 7% of the average magnitude of the Shapley
# values, while the exact explainer evaluates no coalition at all.
#
# How the Cost Grows with the Number of Features
# ----------------------------------------------
# Brute force (:class:`~shapiq.game_theory.ExactComputer` on the game) evaluates all :math:`2^d`
# coalitions, so its runtime doubles with every feature. The explainer's runtime grows
# polynomially, with :math:`d^2` times the number of support vectors. We fit an SVM on synthetic
# data with :math:`d` features, explain one point both ways, and check that the results agree.

n_features_list = [4, 6, 8, 10, 12, 14, 16]
explainer_times, brute_force_times = [], []
for n_features in n_features_list:
    X, y = make_classification(
        n_samples=300, n_features=n_features, n_informative=4, n_redundant=0, random_state=0
    )
    X = StandardScaler().fit_transform(X)
    svm = SVC(kernel="rbf").fit(X, y)

    start = time.perf_counter()
    values = ProductKernelExplainer(svm).explain(X[0]).get_n_order_values(1)
    explainer_times.append(time.perf_counter() - start)

    start = time.perf_counter()
    computer = shapiq.ExactComputer(ProductKernelGame(svm, x=X[0]))
    brute_force = computer(index="SV", order=1).get_n_order_values(1)
    brute_force_times.append(time.perf_counter() - start)

    difference = np.max(np.abs(values - brute_force))
    print(f"d = {n_features:>2}: max. abs. difference to brute force {difference:.1e}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(n_features_list, brute_force_times, marker="o", label="brute force ($2^d$ coalitions)")
ax.semilogy(n_features_list, explainer_times, marker="o", label="ProductKernelExplainer")
ax.set_xlabel("number of features $d$")
ax.set_ylabel("runtime of one explanation [s]")
ax.legend()
fig.tight_layout()
plt.show()

# sphinx_gallery_thumbnail_number = 3

# %%
# For a handful of features, brute force is just as fast. From about ten features on, the
# exponential cost dominates: at 30 features, brute force would take hours for one explanation.
#
# References
# ----------
# .. footbibliography::
