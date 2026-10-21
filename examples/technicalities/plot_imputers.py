"""
Imputers on Correlated Data
===========================

To explain a prediction, ``shapiq`` evaluates the model with some features *absent*. A model
cannot leave out a feature, so an imputer fills in the absent features, and the choice of the
imputer defines the game that is explained. This example compares the
:class:`~shapiq.imputer.MarginalImputer`, the :class:`~shapiq.imputer.BaselineImputer`, the
:class:`~shapiq.imputer.GaussianImputer`, and the :class:`~shapiq.imputer.GaussianCopulaImputer` on
correlated data, where they disagree, and explains why: the first two are *interventional*, the
other two *observational*.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import shapiq

# %%
# Correlated Data and a Model That Ignores a Feature
# --------------------------------------------------
# We draw 1,000 points with four features. ``x0`` and ``x1`` are strongly correlated (with
# correlation :math:`\rho = 0.9` before ``x1`` is made skewed by an exponential), ``x2`` and
# ``x3`` are independent standard normal features. The model uses ``x0`` and ``x2`` only:
#
# .. math::
#     f(x) = 2 x_0 + x_2^2.
#
# We explain its prediction for the point :math:`x = (1, e, 2, 0)`, a plausible point: its
# ``x1 = e`` stems from the same latent value 1 as its ``x0``.

RHO = 0.9
FEATURES = ["x0", "x1", "x2", "x3"]
rng = np.random.default_rng(0)
covariance = np.eye(4)
covariance[0, 1] = covariance[1, 0] = RHO
latent = rng.multivariate_normal(np.zeros(4), covariance, size=1_000)
data = latent.copy()
data[:, 1] = np.exp(latent[:, 1])  # a skewed copy of x0
x = np.array([1.0, np.e, 2.0, 0.0])


def model(points: np.ndarray) -> np.ndarray:
    return 2 * points[:, 0] + points[:, 2] ** 2


print(f"Prediction f(x) = {model(x[None])[0]:.2f}, average prediction {model(data).mean():.2f}")

# %%
# Every Imputer Is a Game
# -----------------------
# An imputer is a :class:`~shapiq.game.Game` whose players are the features: the value of a
# coalition is the (average) prediction when the features outside the coalition are imputed.
#
# - :class:`~shapiq.imputer.MarginalImputer` replaces the absent features by rows of the background
#   data, independently of the present features (the *interventional* or marginal approach).
# - :class:`~shapiq.imputer.BaselineImputer` replaces them by a single reference point, the mean of
#   the data.
# - :class:`~shapiq.imputer.GaussianImputer` samples them from their distribution *conditional* on
#   the present features, assuming that the features are jointly Gaussian :cite:p:`Aas.2021`
#   (the *observational* approach).
# - :class:`~shapiq.imputer.GaussianCopulaImputer` also samples them conditionally, but models only
#   the dependence as Gaussian and keeps the marginal distributions of the data
#   :cite:p:`Aas.2021`.
#
# Since the game has only four players, :class:`~shapiq.game_theory.ExactComputer` computes its
# exact Shapley values from all 16 coalitions.

imputers = {
    "Marginal": shapiq.MarginalImputer(model, data, x=x, sample_size=None, random_state=0),
    "Baseline": shapiq.BaselineImputer(model, data, x=x),
    "Gaussian": shapiq.GaussianImputer(model, data, x=x, sample_size=2_000, random_state=0),
    "Gaussian copula": shapiq.GaussianCopulaImputer(
        model, data, x=x, sample_size=2_000, random_state=0
    ),
}
shapley_values = {
    name: shapiq.ExactComputer(imputer, n_players=4)(index="SV", order=1).get_n_order_values(1)
    for name, imputer in imputers.items()
}
for name, values in shapley_values.items():
    print(f"{name:>15}: {np.round(values, 2) + 0.0}")

# %%
# The Two Games Behind the Imputers
# ---------------------------------
# We know the distribution of the data, so we can write down both games exactly. The
# *interventional* game averages the model over the absent features drawn independently of the
# present ones,
#
# .. math::
#     \nu_{\text{int}}(S) = \mathbb{E}\left[f(x_S, X_{\bar S})\right],
#
# and the *observational* game conditions on the present features,
#
# .. math::
#     \nu_{\text{obs}}(S) = \mathbb{E}\left[f(X) \mid X_S = x_S\right].
#
# For our model, :math:`\mathbb{E}[x_2^2] = 1`, and given :math:`x_1`, the expected :math:`x_0` is
# :math:`\rho \log x_1`.


def interventional_game(coalitions: np.ndarray) -> np.ndarray:
    known = np.atleast_2d(coalitions).astype(bool)
    return 2 * np.where(known[:, 0], x[0], 0.0) + np.where(known[:, 2], x[2] ** 2, 1.0)


def observational_game(coalitions: np.ndarray) -> np.ndarray:
    known = np.atleast_2d(coalitions).astype(bool)
    expected_x0 = np.where(known[:, 0], x[0], np.where(known[:, 1], RHO * np.log(x[1]), 0.0))
    return 2 * expected_x0 + np.where(known[:, 2], x[2] ** 2, 1.0)


truth = {
    name: shapiq.ExactComputer(game, n_players=4)(index="SV", order=1).get_n_order_values(1)
    for name, game in (
        ("interventional", interventional_game),
        ("observational", observational_game),
    )
}
for name, values in truth.items():
    print(f"{name:>15}: {np.round(values, 2) + 0.0}")

# %%
# The Shapley values of the four imputers, with the exact Shapley values of the game that each
# imputer aims at as markers:

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
width = 0.2
positions = np.arange(len(FEATURES))
fig, ax = plt.subplots(figsize=(9, 4.2))
for i, ((name, values), color) in enumerate(zip(shapley_values.items(), COLORS, strict=True)):
    ax.bar(positions + (i - 1.5) * width, values, width=width * 0.9, color=color, label=name)
# the game each imputer aims at: interventional (first two bars), observational (last two)
for bars, name, marker in (((0, 1), "interventional", "x"), ((2, 3), "observational", "o")):
    for i in bars:
        ax.scatter(
            positions + (i - 1.5) * width,
            truth[name],
            marker=marker,
            s=45,
            zorder=3,
            label=f"exact {name} game" if i == bars[0] else None,
            **(
                {"color": "#0b0b0b"}
                if marker == "x"
                else {"facecolor": "none", "edgecolor": "#0b0b0b"}
            ),
        )
ax.axhline(0, color="#c3c2b7", linewidth=1)
ax.set_xticks(positions, FEATURES)
ax.set_ylabel("Shapley value")
ax.set_title("The same prediction, explained with four imputers")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, ncols=2)
plt.tight_layout()
plt.show()

# %%
# - The **marginal imputer** reproduces the interventional game (up to sampling error): ``x1``
#   and ``x3`` get 0, because the model does not use them. Its explanation is *true to the model*
#   :cite:p:`Chen.2020`.
# - The **baseline imputer** agrees on ``x0``, ``x1``, and ``x3`` but gives ``x2`` the value 4
#   instead of 3: it compares with the prediction at the mean point (:math:`x_2 = 0`, so
#   :math:`x_2^2 = 0`), not with the average prediction (:math:`\mathbb{E}[x_2^2] = 1`). For models
#   that are linear in a feature, both imputers agree on it.
# - The **Gaussian copula imputer** reproduces the observational game (up to sampling error):
#   ``x0`` and ``x1`` share the 2 that ``x0`` earns alone in the interventional game, and ``x1``
#   gets about :math:`\rho = 0.9`, although the model never reads it, because knowing ``x1``
#   already reveals most of ``x0``. Its explanation is *true to the data*.
# - The **Gaussian imputer** targets the same game but assumes that ``x1`` is Gaussian. It is not
#   (it is skewed), so the imputer underestimates how much ``x1`` reveals about ``x0`` and gives
#   ``x1`` too little. The copula avoids this by transforming every feature to a Gaussian first.
#
# ``x3`` is independent of everything and unused, so all imputers give it (about) 0.
#
# How the Correlation Moves the Credit
# ------------------------------------
# The difference between the two approaches grows with the correlation. For jointly Gaussian
# data (no exponential, so the assumption of the Gaussian imputer holds) and the point
# :math:`x = (1, 1, 2, 0)`, the observational Shapley values of ``x0`` and ``x1`` are
# :math:`2 - \rho` and :math:`\rho`, while the interventional ones stay at 2 and 0.

correlations = np.linspace(0, 0.95, 8)
x_gaussian = np.array([1.0, 1.0, 2.0, 0.0])
sweep = {"Marginal": [], "Gaussian": []}
for rho in correlations:
    covariance = np.eye(4)
    covariance[0, 1] = covariance[1, 0] = rho
    gaussian_data = rng.multivariate_normal(np.zeros(4), covariance, size=1_000)
    for name, imputer_class, kwargs in (
        ("Marginal", shapiq.MarginalImputer, {"sample_size": None}),
        ("Gaussian", shapiq.GaussianImputer, {"sample_size": 5_000}),
    ):
        imputer = imputer_class(model, gaussian_data, x=x_gaussian, random_state=0, **kwargs)
        sweep[name].append(shapiq.ExactComputer(imputer, n_players=4)("SV").get_n_order_values(1))

theory = {  # the Shapley values of x0 and x1 of the two games
    "Marginal": (np.full_like(correlations, 2.0), np.zeros_like(correlations)),
    "Gaussian": (2 - correlations, correlations),
}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (name, values) in zip(axes, sweep.items(), strict=True):
    estimates = np.array(values)
    for feature, color in ((0, "#2a78d6"), (1, "#eb6834")):
        ax.plot(correlations, theory[name][feature], color=color, lw=1, linestyle="--")
        ax.plot(correlations, estimates[:, feature], "o-", color=color, lw=2, label=f"x{feature}")
    ax.set_title(f"{name} imputer")
    ax.set_xlabel(r"Correlation $\rho$ of x0 and x1")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Shapley value (dashed: theory)")
axes[0].legend(frameon=False)
plt.tight_layout()
plt.show()

# %%
# Which Imputer Should You Use?
# -----------------------------
# Neither approach is wrong; they answer different questions :cite:p:`Chen.2020`.
#
# - Use an **interventional** imputer (marginal, or baseline for a single reference point) to
#   explain how the *model* computes its prediction. A feature the model does not use gets 0, and
#   the explanation follows the effect of setting a feature, as argued by
#   :cite:t:`Janzing.2020`. The price: the imputed points can be unrealistic (here, a small
#   ``x0`` next to a large ``x1``), and the model is evaluated where it has seen no data.
# - Use an **observational** imputer to explain how much each feature *tells* about the
#   prediction, given the dependencies in the data. Correlated features share the credit, and the
#   imputer must model the conditional distributions well: prefer the Gaussian copula over the
#   Gaussian imputer when the features are not Gaussian.
#
# Since every imputer is a game, it works with :class:`~shapiq.game_theory.ExactComputer` as above
# and with every approximator, e.g. ``shapiq.KernelSHAP(n=4).approximate(budget=16, game=imputer)``.
