"""
Causal Shapley Values: The Same Feature, +3.00 and -2.33
=========================================================

This example contrasts the three value functions a Shapley explanation can be built
on, using the :class:`~shapiq_games.synthetic.ConfoundedChainSCM` game: the
**marginal** one, the **conditional** one, and the **causal** one of
Heskes et al. (2020).

All three decompose the very same prediction of the very same model at the very same
point. They disagree on the sign of one feature, because they answer three different
questions:

- **marginal**: what does knowing ``X3 = 1`` add, if I know nothing else?
- **conditional**: what does knowing ``X3 = 1`` add, given everything else I know?
- **causal**: what happens if I *set* ``X3 = 1``, following the causal structure?

The data-generating process puts an unobserved common cause behind the first two
features and makes the third a descendant of both::

    X1 = U + noise
    X2 = U + noise            # X1 and X2 share a hidden cause
    X3 = 2*X1 + X2 + noise    # X3 is downstream of both
    f  = X1 + 2*X2 + 3*X3

Because the model is linear and the features are Gaussian, every conditional
expectation involved is linear and all three value functions are available in closed
form. Three features means eight coalitions, enumerated exactly, so the Shapley values
below carry no approximation error and no random seed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import shapiq
from shapiq_games.synthetic import ConfoundedChainSCM

FEATURE_NAMES = ["X1", "X2", "X3"]
MODES = ["marginal", "conditional", "causal"]

# %%
# The Causal Chain Graph
# ----------------------
# The causal structure enters through two arguments, exactly as in the R package
# ``shapr``: a ``causal_ordering`` of feature groups and one ``confounding`` flag per
# group. Here the first two features form one confounded group and the third follows.

game = ConfoundedChainSCM(mode="causal")
print(f"causal ordering: {game.causal_ordering}")
print(f"confounding:     {game.confounding}")
print(f"f(x*) = {game.coefficients @ game.x_explain:.2f}")

# %%
# Coalition Values
# ----------------
# Two sanity checks are visible directly in the table. Intervening on ``X3``, which has
# no descendants, cannot differ from perturbing it, so the marginal and the causal
# column agree on ``{X3}``. Fixing every ancestor leaves nothing for an intervention to
# cut, so the conditional and the causal column agree on ``{X1,X2}``.

games = {mode: ConfoundedChainSCM(mode=mode) for mode in MODES}
coalitions = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
).astype(bool)

labels = [
    "{}" if not coalition.any() else "{" + ",".join(np.array(FEATURE_NAMES)[coalition]) + "}"
    for coalition in coalitions
]
values = {mode: games[mode](coalitions) for mode in MODES}

print(f"\n{'S':<12}{'marginal':>12}{'conditional':>13}{'causal':>10}")
for i, label in enumerate(labels):
    row = "".join(
        f"{values[mode][i]:>12.4f}" if mode != "conditional" else f"{values[mode][i]:>13.4f}"
        for mode in MODES
    )
    print(f"{label:<12}{row}")

# %%
# Exact Shapley Values
# --------------------
# With three players all eight coalitions are enumerated, so these are exact.

attributions = {}
for mode in MODES:
    exact_computer = shapiq.ExactComputer(n_players=3, game=games[mode])
    sv = exact_computer(index="SV", order=1)
    attributions[mode] = np.array([sv[(player,)] for player in range(3)])

print(f"\n{'':<14}{'X1':>8}{'X2':>8}{'X3':>8}{'sum':>8}")
for mode in MODES:
    phi = attributions[mode]
    print(f"{mode:<14}" + "".join(f"{value:>8.2f}" for value in phi) + f"{phi.sum():>8.2f}")

# %%
# Why the Third Feature Flips
# ---------------------------
# The marginal value function compares ``X3 = 1`` against its unconditional mean of
# zero, and the model weights it by 3, so it looks like the strongest positive
# contributor. The other two compare it against what its parents predict for it,
# ``2*1 + 1*1 = 3``. Observed against expected, this point's ``X3`` came in two units
# low and drags the prediction down.
#
# Conditional and causal then part ways on ``X2``. Intervening on ``X1`` says nothing
# about ``X2``, because their dependence runs through the hidden common cause and the
# intervention cuts that path. Conditioning on ``X1`` does say something. Observational
# attribution therefore lets credit leak between the two; causal attribution does not.

fig, ax = plt.subplots(figsize=(6, 4))
positions = np.arange(3)
width = 0.26
for offset, mode in zip((-width, 0.0, width), MODES, strict=True):
    ax.bar(positions + offset, attributions[mode], width, label=mode)
ax.axhline(0.0, color="black", linewidth=0.8)
ax.set_xticks(positions)
ax.set_xticklabels(FEATURE_NAMES)
ax.set_ylabel("Shapley value")
ax.set_title("One prediction, three value functions")
ax.legend()
fig.tight_layout()
plt.show()
