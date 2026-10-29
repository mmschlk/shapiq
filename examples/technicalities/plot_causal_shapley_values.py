"""
Causal Shapley Values: The Same Feature, +3.00 and -2.33
=========================================================

This example contrasts the three value functions a Shapley explanation can be built on, using
the :class:`~shapiq_games.LinearGaussianGame` game: the **marginal** one, the **conditional**
one, and the **causal** one of :cite:t:`Heskes.2020`.

All three decompose the very same prediction of the very same model at the very same point.
They disagree on the sign of one feature, because they answer three different questions:

- **marginal**: what does knowing ``X3 = 1`` add, if I know nothing else?
- **conditional**: what does knowing ``X3 = 1`` add, given everything else I know?
- **causal**: what happens if I *set* ``X3 = 1``, following the causal structure?

The data-generating process puts an unobserved common cause behind the first two features and
makes the third a descendant of both::

    X1 = U + noise
    X2 = U + noise            # X1 and X2 share a hidden cause
    X3 = 2*X1 + X2 + noise    # X3 is downstream of both
    f  = X1 + 2*X2 + 3*X3

Because the model is linear and the features are Gaussian, every conditional expectation
involved is linear and all three value functions are available in closed form. Three features
means eight coalitions, enumerated exactly, so the Shapley values below carry no approximation
error.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import shapiq
from shapiq.utils import powerset
from shapiq_games import LinearGaussianGame, RandomLinearGaussianGame
from shapiq_games.synthetic import LinearGaussianSCM

MODES = ("marginal", "conditional", "causal")

# %%
# The Structural Causal Model
# ---------------------------
# :class:`~shapiq_games.synthetic.LinearGaussianSCM` holds the structural equations:
# ``weights[j, k]`` is the direct effect of feature ``k`` on feature ``j``, and the unobserved
# terms of ``X1`` and ``X2`` share the common cause ``U``, so their noise covariance is not
# diagonal. The model also carries its causal chain graph, exactly as the R package ``shapr``
# takes it: a ``causal_ordering`` of feature groups and one ``confounding`` flag per group. Here
# the first two features form one confounded group and the third follows.

weights = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
noise_cov = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
scm = LinearGaussianSCM(
    weights, noise_cov=noise_cov, causal_ordering=[[0, 1], [2]], confounding=[True, False]
)
print(f"feature covariance:\n{scm.cov}")

coefficients = np.array([1.0, 2.0, 3.0])
x_explain = np.ones(3)
feature_names = [f"X{j + 1}" for j in range(len(coefficients))]

# %%
# One Game per Value Function
# ---------------------------
# A :class:`~shapiq_games.LinearGaussianGame` explains the linear model on the features of the
# structural causal model, with the value function given by ``mode``.

games = {mode: LinearGaussianGame(coefficients, x_explain, scm=scm, mode=mode) for mode in MODES}
print(f"f(x*) = {games['causal'].original_model_output:.2f}")

# %%
# Coalition Values
# ----------------
# Two sanity checks are visible directly in the table. Intervening on ``X3``, which has no
# descendants, cannot differ from perturbing it, so the marginal and the causal column agree on
# ``{X3}``. Fixing every ancestor leaves nothing for an intervention to cut, so the conditional
# and the causal column agree on ``{X1,X2}``.

n_players = len(coefficients)
subsets = list(powerset(range(n_players)))
coalitions = np.array([[j in subset for j in range(n_players)] for subset in subsets], dtype=bool)
values = {mode: games[mode](coalitions) for mode in MODES}

print(f"{'S':<12}" + "".join(f"{mode:>13}" for mode in MODES))
for i, subset in enumerate(subsets):
    label = "{" + ",".join(feature_names[j] for j in subset) + "}"
    print(f"{label:<12}" + "".join(f"{values[mode][i]:>13.4f}" for mode in MODES))

# %%
# Because the game knows the structural equations, it can also intervene on the model itself:
# ``mode="do"`` solves the equations with the incoming effects of ``S`` cut. On the chain graph
# the model was built on, this agrees with the causal column.

do_game = LinearGaussianGame(coefficients, x_explain, scm=scm, mode="do")
print(f"do equals causal: {np.allclose(do_game(coalitions), values['causal'])}")

# %%
# Exact Shapley Values
# --------------------
# With three players all eight coalitions are enumerated, so these are exact.


def shapley_values(game: LinearGaussianGame) -> np.ndarray:
    """Return the exact Shapley values of a game as an array."""
    exact = shapiq.ExactComputer(game)("SV", 1)
    return np.array([exact[(player,)] for player in range(game.n_players)])


attributions = {mode: shapley_values(games[mode]) for mode in MODES}

print(f"{'':<14}" + "".join(f"{name:>8}" for name in feature_names) + f"{'sum':>8}")
for mode in MODES:
    phi = attributions[mode]
    print(f"{mode:<14}" + "".join(f"{value:>8.2f}" for value in phi) + f"{phi.sum():>8.2f}")

# %%
# Why the Third Feature Flips
# ---------------------------
# The marginal value function compares ``X3 = 1`` against its unconditional mean of zero, and
# the model weights it by 3, so it looks like the strongest positive contributor. The other two
# compare it against what its parents predict for it, ``2*1 + 1*1 = 3``. Observed against
# expected, this point's ``X3`` came in two units low and drags the prediction down.
#
# Conditional and causal then part ways on ``X2``. Intervening on ``X1`` says nothing about
# ``X2``, because their dependence runs through the hidden common cause and the intervention
# cuts that path. Conditioning on ``X1`` does say something. Observational attribution therefore
# lets credit leak between the two; causal attribution does not.

fig, ax = plt.subplots(figsize=(6, 4))
positions = np.arange(n_players)
width = 0.26
for offset, mode in zip((-width, 0.0, width), MODES, strict=True):
    ax.bar(positions + offset, attributions[mode], width, label=mode)
ax.axhline(0.0, color="black", linewidth=0.8)
ax.set_xticks(positions)
ax.set_xticklabels(feature_names)
ax.set_ylabel("Shapley value")
ax.set_title("One prediction, three value functions")
ax.legend()
fig.tight_layout()
plt.show()

# %%
# Random Structural Causal Models
# -------------------------------
# For benchmarks, :class:`~shapiq_games.RandomLinearGaussianGame` draws such models of any size
# with :meth:`LinearGaussianSCM.random <shapiq_games.synthetic.LinearGaussianSCM.random>`: a
# causal ordering of feature groups, unobserved common causes within some groups, and structural
# weights from earlier to later groups. The same ``random_state`` draws the same model, here once
# per value function.

random_games = {mode: RandomLinearGaussianGame(8, mode=mode, random_state=1) for mode in MODES}
print(f"causal ordering: {random_games['causal'].scm.causal_ordering}")
print(f"confounding:     {random_games['causal'].scm.confounding}")
for mode, game in random_games.items():
    print(f"{mode:<14}" + "".join(f"{value:>7.2f}" for value in shapley_values(game)))
