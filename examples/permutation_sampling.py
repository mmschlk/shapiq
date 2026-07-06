"""Playground for the permutation-sampling approximators.

Run with:
    uv run python examples/permutation_sampling.py

The game is a small quadratic cooperative game with known ground truth:
    v(S) = sum_{i in S} w_i + sum_{i<j in S} M_ij
so the exact Shapley values are w_i + 0.5 * sum_j M_ij and the exact
pairwise SII values are exactly M_ij. Tweak the constants, budgets, and
keys below and watch what happens.
"""

import jax.numpy as jnp
from jax import Array

from shapiq import (
    CallableGame,
    InsufficientSamplesError,
    PermutationSamplingSII,
    PermutationSamplingSTII,
    PermutationSamplingSV,
)

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
EXACT_SV = WEIGHTS + 0.5 * jnp.sum(PAIRS, axis=1)


def game_value(coalitions) -> Array:
    masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
    return masks @ WEIGHTS + 0.5 * jnp.einsum("...i,ij,...j->...", masks, PAIRS, masks)


game = CallableGame(fn=game_value, n_players=N_PLAYERS)


def order_one(explanation) -> Array:
    return jnp.stack([explanation((player,)) for player in range(N_PLAYERS)])


print("=== Shapley values via permutation walks ===")
approximator = PermutationSamplingSV.create(game, key=0)
total_payout = float(jnp.sum(EXACT_SV))  # v(N) - v(empty)
print(f"sampling quantum: {approximator.sampler.sampling_quantum} evaluations per walk")
print(f"exact SV: {EXACT_SV}")
for budget in (4, 9, 51, 400):
    approximator = approximator.sample(budget)
    estimate = order_one(approximator.explain())
    print(
        f"after +{budget:>4} evals | stored: {approximator.state.n_samples:>4}"
        f" | pending: {approximator.sampler.n_pending}"
        f" | max error: {jnp.max(jnp.abs(estimate - EXACT_SV)):.4f}"
        f" | efficiency gap: {jnp.abs(jnp.sum(estimate) - total_payout):.2e}"
    )

print()
print("=== budgets are spent exactly; splits do not matter ===")
whole = PermutationSamplingSV.create(game, key=0).sample(100)
split = PermutationSamplingSV.create(game, key=0).sample(7).sample(13).sample(80)
print(f"states equal: {split.state == whole.state}")

print()
print("=== history: watch the estimate converge ===")
approximator = PermutationSamplingSV.create(game, key=3, track_history=True)
for _ in range(5):
    approximator = approximator.sample(40)
for step, past in enumerate(approximator.history()):
    try:
        report = f"max error: {jnp.max(jnp.abs(order_one(past.explain()) - EXACT_SV)):.4f}"
    except InsufficientSamplesError:
        report = "no completed walk yet"
    print(f"state {step} | samples: {past.state.n_samples:>4} | {report}")

print()
print("=== pairwise Shapley interactions (SII) ===")
approximator = PermutationSamplingSII.create(game, key=0)
print(f"sampling quantum: {approximator.sampler.sampling_quantum} evaluations per walk")
approximator = approximator.sample(100 * approximator.sampler.sampling_quantum)
explanation = approximator.explain()
print("pair : estimate | exact")
for left in range(N_PLAYERS):
    for right in range(left + 1, N_PLAYERS):
        estimate = float(explanation((left, right)))
        print(f"({left}, {right}) : {estimate:+.4f} | {float(PAIRS[left, right]):+.4f}")

print()
print("=== Shapley-Taylor interactions (STII) ===")
# lower orders are exact discrete derivatives at the empty coalition (here: the
# raw weights), and every walk samples every top-order pair once, so a single
# walk already covers all pairs -- for this quadratic game even exactly
approximator = PermutationSamplingSTII.create(game, order=2, key=0)
print(f"sampling quantum: {approximator.sampler.sampling_quantum} evaluations per walk")
explanation = approximator.sample(approximator.sampler.sampling_quantum).explain()
print(f"order 1 (exact, = weights): {order_one(explanation)}")
print("pair : estimate | exact")
for left in range(N_PLAYERS):
    for right in range(left + 1, N_PLAYERS):
        estimate = float(explanation((left, right)))
        print(f"({left}, {right}) : {estimate:+.4f} | {float(PAIRS[left, right]):+.4f}")
