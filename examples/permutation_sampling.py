"""Playground for the permutation-sampling policies.

Run with:
    uv run python examples/permutation_sampling.py

The game is a small quadratic cooperative game with known ground truth:
    v(S) = sum_{i in S} w_i + sum_{i<j in S} M_ij
so the exact Shapley values are w_i + 0.5 * sum_j M_ij and the exact
pairwise SII values are exactly M_ij. Policies are frozen; every verb
returns an inert estimate — a game with provenance. Tweak the constants,
budgets, and keys below and watch what happens.
"""

import jax.numpy as jnp
from jax import Array

from shapiq import (
    SII,
    STII,
    SV,
    CallableGame,
    InsufficientSamplesError,
    PermutationSampling,
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
    EXACT_SV = WEIGHTS + 0.5 * jnp.sum(PAIRS, axis=1)

    def game_value(coalitions) -> Array:
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        return masks @ WEIGHTS + 0.5 * jnp.einsum("...i,ij,...j->...", masks, PAIRS, masks)

    game = CallableGame(fn=game_value, n_players=N_PLAYERS)

    def order_one(estimate) -> Array:
        return jnp.stack([estimate[(player,)] for player in range(N_PLAYERS)])

    print("=== Shapley values via permutation walks ===")
    policy = PermutationSampling(game, SV(), random_state=0)
    total_payout = float(jnp.sum(EXACT_SV))  # v(N) - v(empty)
    print(f"walk length: {policy.unit_rows} evaluations per walk")
    print(f"seed samples (paid from the first budget): {policy.n_seed_samples}")
    print(f"exact SV: {EXACT_SV}")
    estimate = policy.estimate(0)
    for budget in (6, 9, 51, 400):
        estimate = policy.refine(estimate, budget)
        values = order_one(estimate)
        print(
            f"after +{budget:>4} evals | stored: {estimate.evidence.n_samples:>4}"
            f" | banked: {estimate.bank}"
            f" | max error: {jnp.max(jnp.abs(values - EXACT_SV)):.4f}"
            f" | efficiency gap: {jnp.abs(jnp.sum(values) - total_payout):.2e}"
        )

    print()
    print("=== whole-unit spending banks the remainder; splits do not matter ===")
    whole = PermutationSampling(game, SV(), random_state=0).estimate(100)
    split_policy = PermutationSampling(game, SV(), random_state=0)
    split = split_policy.refine(split_policy.refine(split_policy.estimate(7), 13), 80)
    print(f"evidence equal: {split.evidence == whole.evidence}")

    print()
    print("=== history: watch the estimate converge ===")
    policy = PermutationSampling(game, SV(), random_state=3)
    estimate = policy.estimate(40)
    for _ in range(4):
        estimate = policy.refine(estimate, 40)
    for step, state in enumerate(estimate.evidence.history()):
        past = policy.at_evidence(state)
        try:
            report = f"max error: {jnp.max(jnp.abs(order_one(past) - EXACT_SV)):.4f}"
        except InsufficientSamplesError:
            report = "no completed walk yet"
        print(f"state {step} | samples: {state.n_samples:>4} | {report}")

    print()
    print("=== pairwise Shapley interactions (SII) ===")
    policy = PermutationSampling(game, SII(), random_state=0)
    print(f"walk length: {policy.unit_rows} evaluations per walk")
    estimate = policy.estimate(100 * policy.unit_rows)
    print("pair : estimate | exact")
    for left in range(N_PLAYERS):
        for right in range(left + 1, N_PLAYERS):
            value = float(estimate[(left, right)])
            print(f"({left}, {right}) : {value:+.4f} | {float(PAIRS[left, right]):+.4f}")

    print()
    print("=== deduplication: never pay for the same coalition twice ===")
    # budget counts only novel game evaluations; repeated coalitions reuse stored
    # values and their walks are free evidence. with only 2**5 = 32 coalitions in
    # this game, duplicates pile up fast. (a budget near 2**n stalls with a
    # SamplingStallWarning once no novel coalitions remain.)
    estimate = PermutationSampling(game, SV(), random_state=0, deduplicate=True).estimate(25)
    print("budget spent on novel evaluations: 25")
    print(f"raw samples stored (walks incl. free duplicates): {estimate.evidence.n_samples}")
    print(f"max error: {jnp.max(jnp.abs(order_one(estimate) - EXACT_SV)):.4f}")

    print()
    print("=== Shapley-Taylor interactions (STII) ===")
    # lower orders are exact discrete derivatives at the empty coalition (here: the
    # raw weights), and every walk samples every top-order pair once, so a single
    # walk already covers all pairs -- for this quadratic game even exactly
    policy = PermutationSampling(game, STII(order=2), random_state=0)
    print(f"walk length: {policy.unit_rows} evaluations per walk")
    estimate = policy.estimate(policy.n_seed_samples + policy.unit_rows)
    print(f"order 1 (exact, = weights): {order_one(estimate)}")
    print("pair : estimate | exact")
    for left in range(N_PLAYERS):
        for right in range(left + 1, N_PLAYERS):
            value = float(estimate[(left, right)])
            print(f"({left}, {right}) : {value:+.4f} | {float(PAIRS[left, right]):+.4f}")
