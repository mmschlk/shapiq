"""
Exact Shapley Values of a Gaussian Process
==========================================

The posterior mean of a Gaussian process (GP) regression is a weighted sum of kernels,
:math:`f(x) = \\sum_i \\alpha_i k(x^i, x)`, over the training points :math:`x^i`. With an RBF
kernel, :math:`k` is a *product* of one-dimensional kernels, one per feature.
:footcite:t:`Mohammadi.2025a` showed that the Shapley values of such product-kernel models can be
computed exactly in polynomial time, without evaluating any of the :math:`2^d` coalitions of the
:math:`d` features. :class:`~shapiq.explainer.product_kernel.ProductKernelExplainer` implements
their algorithm for Gaussian processes and support vector machines.

This example fits a GP to the diabetes dataset, explains its predictions exactly, and checks the
result against a brute-force evaluation of all :math:`2^{10}` coalitions.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import shapiq
from shapiq.explainer.product_kernel import ProductKernelExplainer

# %%
# Fit a Gaussian Process
# ----------------------
# The diabetes dataset has 442 patients, ten baseline features (age, sex, body mass index, blood
# pressure, and six blood serum measurements), and a measure of disease progression one year
# later as the target.
#
# The explainer supports a GP whose kernel is a single
# :class:`~sklearn.gaussian_process.kernels.RBF` kernel with one length scale for all features. We
# therefore standardize the features (so that one length scale fits all of them) and the target
# (instead of ``normalize_y=True``, which the explainer does not take into account), and model the
# noise with the ``alpha`` parameter instead of a ``WhiteKernel``. The noise level is the one of
# three candidates that maximizes the marginal likelihood, while the length scale is fitted by
# scikit-learn.

feature_names = ["age", "sex", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"]
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

x_scaler = StandardScaler().fit(X_train)
X_train, X_test = x_scaler.transform(X_train), x_scaler.transform(X_test)
y_mean, y_std = y_train.mean(), y_train.std()
y_train, y_test = (y_train - y_mean) / y_std, (y_test - y_mean) / y_std

candidates = [
    GaussianProcessRegressor(kernel=RBF(length_scale=5.0), alpha=noise).fit(X_train, y_train)
    for noise in (0.25, 0.5, 1.0)
]
gp = max(candidates, key=lambda model: model.log_marginal_likelihood_value_)
print(f"Kernel: {gp.kernel_}, noise: {gp.alpha}")
print(f"Test R2: {gp.score(X_test, y_test):.3f}")

# %%
# The Product-Kernel Game
# -----------------------
# To explain the prediction :math:`f(x)` for a patient :math:`x`, the features are the players.
# A coalition :math:`S` of features keeps only the kernel factors of its features:
#
# .. math::
#     v(S) = \sum_i \alpha_i \prod_{j \in S} \exp\bigl(-\gamma\, (x^i_j - x_j)^2\bigr),
#     \qquad \gamma = \frac{1}{2 \ell^2},
#
# where :math:`\ell` is the length scale. With all features, :math:`v(N) = f(x)` is the GP's
# prediction. Without any feature, every kernel value is one and :math:`v(\emptyset) =
# \sum_i \alpha_i`: a feature is removed by no longer comparing the patient with the training
# points on it. The game needs no background data, unlike the imputation-based games of
# :class:`~shapiq.explainer.TabularExplainer`. It takes three lines of numpy to evaluate it for a
# batch of coalitions:

x = X_test[0]
alpha = gp.alpha_  # the weights of the training points
gamma = 1 / (2 * gp.kernel_.length_scale**2)
squared_distances = (X_train - x) ** 2  # one row per training point, one column per feature


def product_kernel_game(coalitions: np.ndarray) -> np.ndarray:
    """The GP's prediction for x with the kernel restricted to the features of each coalition."""
    distances = coalitions.astype(float) @ squared_distances.T  # (n_coalitions, n_train)
    return np.exp(-gamma * distances) @ alpha


all_features = np.ones((1, len(feature_names)), dtype=bool)
print(f"v(N) = {product_kernel_game(all_features)[0]:.6f}")
print(f"f(x) = {gp.predict(x.reshape(1, -1))[0]:.6f}")
print(f"v(empty set) = {alpha.sum():.6f}")

# %%
# Exact Shapley Values in Polynomial Time
# ---------------------------------------
# :class:`~shapiq.explainer.product_kernel.ProductKernelExplainer` converts the fitted GP and
# computes the Shapley values from the kernel values of the training points, with the elementary
# symmetric polynomials of the other features' kernel vectors. It never evaluates a coalition.
# The Shapley values are efficient: together with the baseline :math:`v(\emptyset)`, they sum up
# to the prediction.

explainer = ProductKernelExplainer(gp)
shapley_values = explainer.explain(x)
print(shapley_values)
total = shapley_values.baseline_value + shapley_values.get_n_order_values(1).sum()
print(f"Baseline + sum of Shapley values: {total:.6f}")

# %%
# Verify against Brute Force
# --------------------------
# :class:`~shapiq.game_theory.ExactComputer` evaluates our game on all :math:`2^{10} = 1024`
# coalitions and computes the Shapley values from the definition. Both agree up to floating-point
# rounding.
# The brute force doubles its work with every feature, while the explainer's cost grows only
# quadratically with the number of features and linearly with the number of training points.

exact_computer = shapiq.ExactComputer(product_kernel_game, n_players=len(feature_names))
brute_force = exact_computer(index="SV", order=1)
max_difference = np.max(
    np.abs(shapley_values.get_n_order_values(1) - brute_force.get_n_order_values(1))
)
print(f"Max. absolute difference to brute force: {max_difference:.1e}")

# %%
# The packaged version of this game is :class:`~shapiq_games.ProductKernelGame`, which also
# accepts support vector machines.
#
# Explain One Prediction
# ----------------------
# The waterfall plot starts at :math:`v(\emptyset)` and adds the Shapley value of one feature
# after the other until it reaches the prediction (in standard deviations of the target).

shapley_values.plot_waterfall(feature_names=feature_names, abbreviate=False)

# %%
# Explain the Test Set
# --------------------
# One explanation takes a few milliseconds, so we can explain all 111 test patients. The
# beeswarm plot shows each patient's Shapley value per feature, colored by the feature's value.

explanations = explainer.explain_X(X_test)
shapiq.beeswarm_plot(explanations, X_test, feature_names=feature_names, abbreviate=False)

# sphinx_gallery_thumbnail_number = 2  # noqa: ERA001

# %%
# A high body mass index (``bmi``), blood pressure (``bp``), and log triglycerides (``ltg``)
# increase the predicted disease progression. ``age`` and ``glu``, however, are negative for
# almost every patient, whatever their value. The reason is the baseline: the explanations start
# at :math:`v(\emptyset) = \sum_i \alpha_i`, not at the average prediction, and by efficiency the
# Shapley values of every patient also carry the gap between the two. The product-kernel game
# answers how the prediction changes when the kernel starts to compare the patient with the
# training points on a feature, which differs from replacing the feature by typical values.

mean_shapley_values = np.mean([iv.get_n_order_values(1) for iv in explanations], axis=0)
print(f"v(empty set): {shapley_values.baseline_value:.3f}")
print(f"Average prediction: {gp.predict(X_test).mean():.3f}")
for name, value in zip(feature_names, mean_shapley_values, strict=True):
    print(f"Average Shapley value of {name}: {value:+.3f}")

# %%
# References
# ----------
# .. footbibliography::
