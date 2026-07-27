"""Playground for the ProxySHAP recipe: encode what you know, pay for the rest.

Run with:
    uv run python examples/proxy_shap.py

The recipe, from primitives only:

1. Spend a budget once on a plain KernelSHAP estimate.
2. Write down what you already know about the mechanism as a proxy game
   (here: a redundant sensor block you know acts through its maximum).
   The proxy's Shapley values are exact reads — no evaluations.
3. Rebase the evidence you already have onto the residual ``v - proxy``
   and re-solve the same regression for a correction. Still no new
   evaluations.
4. ``shapley_values(proxy) + correction`` beats the direct estimate,
   and ``proxy + correction`` is itself a game: measure its fidelity.

Why a *knowledge* proxy and not one fit to the same evidence? Because
the constrained Shapley-kernel regression is sample-exact on games of
order two: for any order-2 proxy the correction gives back exactly what
the proxy put in (the estimate is linear in the game values), and the
combined answer equals the direct one to float noise. The demo prints
that no-op too. The recipe pays exactly when the proxy carries
higher-order structure the order-1 kernel struggles with — which is
also why the original ProxySHAP fits its proxy to a *model*, not to the
explanation budget.
"""

from itertools import combinations

import jax.numpy as jnp
import numpy as np
from jax import Array

from shapiq import (
    SV,
    BasisGame,
    CallableGame,
    MoebiusBasis,
    Regression,
    fidelity,
    fit_game,
    shapley_values,
    to_basis,
    uniform_measure,
)

if __name__ == "__main__":
    N_PLAYERS = 10  # 1024 coalitions
    BUDGET = 160

    rng = np.random.default_rng(7)
    weights = rng.normal(size=N_PLAYERS)
    pairs = rng.normal(size=(N_PLAYERS, N_PLAYERS)) * 0.5
    weights_j = jnp.asarray(weights, dtype=jnp.float32)
    pairs_j = jnp.asarray(pairs, dtype=jnp.float32)

    calls = []

    def game_value(coalitions) -> Array:
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        calls.append(int(masks.shape[-2]))
        redundancy = 1.5 * masks[..., [2, 4, 6]].max(axis=-1)  # the sensor block
        synergy = 0.8 * masks[..., [1, 3, 5, 7]].prod(axis=-1)
        quadratic = masks @ weights_j + 0.5 * jnp.einsum(
            "...i,ij,...j->...", masks, pairs_j, masks,
        )
        return quadratic + redundancy + synergy

    game = CallableGame(fn=game_value, n_players=N_PLAYERS)
    exact_sv = shapley_values(to_basis(game, MoebiusBasis()))
    calls.clear()  # the exact sweep above is ground truth, not part of the budget

    # 1. the direct estimate
    policy = Regression(game, SV(), random_state=0, deduplicate=True)
    direct = policy.estimate(BUDGET)
    direct_sv = np.array([float(direct[(player,)]) for player in range(N_PLAYERS)])
    print(f"evaluations spent on the direct estimate: {sum(calls)}")

    # 2. the knowledge proxy: main effects, pairs, and the redundant block
    proxy_coefficients: dict[frozenset[int], float] = {}
    for i in range(N_PLAYERS):
        proxy_coefficients[frozenset([i])] = weights[i] + 0.5 * pairs[i, i]
        for j in range(i + 1, N_PLAYERS):
            proxy_coefficients[frozenset([i, j])] = 0.5 * (pairs[i, j] + pairs[j, i])
    for size in (1, 2, 3):  # moebius of 1.5 * max over {2, 4, 6}: the OR pattern
        for block in combinations((2, 4, 6), size):
            key = frozenset(block)
            proxy_coefficients[key] = proxy_coefficients.get(key, 0.0) + 1.5 * (-1) ** (size + 1)
    proxy = BasisGame(MoebiusBasis(), proxy_coefficients, N_PLAYERS)

    # 3. rebase the evidence onto the residual and re-solve — zero new calls
    evidence = direct.evidence
    correction = Regression(game - proxy, SV(), random_state=0).at_evidence(evidence.minus(proxy))

    # 4. combine and compare — basis games are closed under addition, and
    # shapley values are linear in the game, so the combined explanation is
    # one readable game and one read
    combined = shapley_values(proxy + correction)
    proxy_alone = shapley_values(proxy)
    print(f"evaluations spent in total               : {sum(calls)}  (the correction was free)")
    print(f"max |error| proxy alone (biased)         : {np.abs(proxy_alone - exact_sv).max():.4f}")
    print(f"max |error| direct KernelSHAP            : {np.abs(direct_sv - exact_sv).max():.4f}")
    print(f"max |error| proxy + correction           : {np.abs(combined - exact_sv).max():.4f}")

    surrogate = proxy + correction  # an explanation is a game: measure it
    print(f"fidelity of proxy+correction (uniform)   : "
          f"{fidelity(game, surrogate, uniform_measure(N_PLAYERS)):.3f}")

    # the no-op, for contrast: an order-2 proxy fit to the same evidence
    masks = np.asarray(evidence.coalitions.to_dense(), dtype=bool)
    self_fit = fit_game(masks, np.asarray(evidence.values), N_PLAYERS, order=2)
    self_correction = Regression(game - self_fit, SV(), random_state=0).at_evidence(
        evidence.minus(self_fit),
    )
    self_combined = shapley_values(self_fit + self_correction)
    print(f"self-fit order-2 proxy, |combined - direct|: "
          f"{np.abs(self_combined - direct_sv).max():.2e}  (a no-op, by linearity)")
