"""Playground for the coalition-functional calculus.

Run with:
    uv run python examples/functional_calculus.py

The moonshot: declare an interaction index through its weight formalism and
get the rest for free. Every cardinal interaction index and generalized value
is a linear functional of the game whose coefficient on ``v(K)`` depends only
on ``|K & S|`` and ``|K|``. shapiq derives that coefficient table from the
declared weights, contracts it densely for exact computation, and importance-
samples it for unbiased anytime estimation — the sampler's size distribution
is derived from the table's coefficient mass, so a new index defined in five
lines comes with a working approximator.
"""

from math import comb

import jax.numpy as jnp
from jax import Array

from shapiq import (
    KSII,
    SII,
    CallableGame,
    ExactExplainer,
    Moebius,
    MonteCarlo,
    define_cardinal_index,
    derive_functional,
)

if __name__ == "__main__":
    N_PLAYERS = 5
    WEIGHTS = jnp.asarray([0.7, -1.3, 0.1, 2.0, -0.4])
    PAIRS = jnp.asarray(
        [
            [0.0, 0.5, -1.0, 0.0, 0.3],
            [0.5, 0.0, 0.2, -0.7, 0.0],
            [-1.0, 0.2, 0.0, 0.4, 0.9],
            [0.0, -0.7, 0.4, 0.0, -0.2],
            [0.3, 0.0, 0.9, -0.2, 0.0],
        ],
    )

    def game_value(coalitions) -> Array:
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        return masks @ WEIGHTS + 0.5 * jnp.einsum("...i,ij,...j->...", masks, PAIRS, masks)

    game = CallableGame(fn=game_value, n_players=N_PLAYERS)

    print("=== a brand-new index in five lines ===")

    # a "latecomer semivalue": discrete derivatives weighted towards large
    # outside coalitions, normalized like a probabilistic value per size
    def latecomer_weights(n_players: int, interaction_size: int) -> Array:
        free = n_players - interaction_size
        raw = jnp.arange(1, free + 2, dtype=jnp.float32)
        total = jnp.sum(raw * jnp.asarray([comb(free, t) for t in range(free + 1)]))
        return raw / total

    latecomer = define_cardinal_index("LatecomerII", weights=latecomer_weights, order=2)

    exact = ExactExplainer(game, latecomer).explain()
    approximator = MonteCarlo(game, latecomer, random_state=0)
    print(f"index name on explanations: {exact.interaction_index!r}")
    for budget in (34, 200, 2000):
        approximator = approximator.sample(budget)
        estimate = approximator.explain()
        error = max(
            float(jnp.max(jnp.abs(estimate(key) - exact(key))))
            for size in (1, 2)
            for key in [
                tuple(range(size)),
                tuple(range(N_PLAYERS - size, N_PLAYERS)),
            ]
        )
        print(
            f"after +{budget:>5} evals | stored: {approximator.state.n_samples:>5}"
            f" | max error vs exact: {error:.4f}"
        )

    print()
    print("=== the derived functional is inspectable ===")
    functional = derive_functional(SII(order=2), N_PLAYERS, 2)
    print(f"represented interaction sizes: {functional.interaction_sizes}")
    print(f"coefficient mass per coalition size: {functional.size_mass()}")
    print("(the Monte Carlo sampler draws coalition sizes proportional to this mass)")
    truncated = derive_functional(Moebius(order=2), N_PLAYERS, 2)
    print(f"Moebius(order=2) mass: {truncated.size_mass()}  <- sizes 3+ are never sampled")

    print()
    print("=== one estimator, any linear-functional index ===")
    for index in (SII(order=2), KSII(order=2), Moebius(order=2)):
        exact = ExactExplainer(game, index).explain()
        estimate = MonteCarlo(game, index, random_state=1).sample(2000).explain()
        pair = (0, 2)
        print(
            f"{index.name:>8} {pair}: estimate {float(estimate(pair)):+.4f}"
            f" | exact {float(exact(pair)):+.4f}"
        )
