"""Playground for the gradient bridge: the extension is part of the method.

Run with:
    uv run python examples/gradient_bridge.py

A gradient explanation is a pair (game, extension): a differentiable
function on the cube that agrees with the game at the vertices. Owen's
theorem is the bridge — integrated gradients along the diagonal of the
*multilinear* extension are exactly the Shapley values, and the gradient
at the cube's center is exactly the Banzhaf values.

Change the extension and the attributions move, even though the game —
every value at every vertex — is unchanged. The demo adds a bump that
vanishes at all vertices and watches the attribution shift by exactly
-C/6 and +C/6 while completeness still holds. That gap is not an error
of integrated gradients; it is a property of the chosen extension, and
with the bridge it is computable.
"""

import numpy as np

from shapiq import (
    CallableGame,
    MoebiusBasis,
    banzhaf_values,
    integrated_gradients,
    multilinear_diagonal_gradient,
    shapley_values,
    to_basis,
)

if __name__ == "__main__":
    import jax.numpy as jnp
    from jax import Array

    N_PLAYERS = 8
    BUMP = 1.2
    STEPS = 512

    rng = np.random.default_rng(7)
    weights = jnp.asarray(rng.normal(size=N_PLAYERS), dtype=jnp.float32)
    pairs = jnp.asarray(rng.normal(size=(N_PLAYERS, N_PLAYERS)) * 0.5, dtype=jnp.float32)

    def game_value(coalitions) -> Array:
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        redundancy = 1.5 * masks[..., [2, 4, 6]].max(axis=-1)
        synergy = 0.8 * masks[..., [1, 3, 5, 7]].prod(axis=-1)
        quadratic = masks @ weights + 0.5 * jnp.einsum("...i,ij,...j->...", masks, pairs, masks)
        return quadratic + redundancy + synergy

    exact = to_basis(CallableGame(fn=game_value, n_players=N_PLAYERS), MoebiusBasis())
    truth = shapley_values(exact)

    # Owen's theorem, live: IG on the multilinear extension == Shapley
    owen = integrated_gradients(
        lambda t: multilinear_diagonal_gradient(exact, t),
        N_PLAYERS,
        steps=STEPS,
    )
    print(f"max |IG on multilinear extension - Shapley|: {np.abs(owen - truth).max():.2e}")

    # the center of the cube: one gradient == Banzhaf
    center = multilinear_diagonal_gradient(exact, 0.5)
    print(f"max |center gradient - Banzhaf|            : "
          f"{np.abs(center - banzhaf_values(exact)).max():.2e}")

    # same game, different extension: add C * z0 (1 - z0) z1, zero at every vertex
    def bumped(t: float) -> np.ndarray:
        gradient = multilinear_diagonal_gradient(exact, t)
        gradient[0] += BUMP * (1 - 2 * t) * t
        gradient[1] += BUMP * t * (1 - t)
        return gradient

    shifted = integrated_gradients(bumped, N_PLAYERS, steps=STEPS)
    print(f"completeness gap after the bump            : "
          f"{abs(shifted.sum() - truth.sum()):.2e}")
    print(f"attribution shift on player 0 (expect {-BUMP / 6:+.4f}): "
          f"{shifted[0] - truth[0]:+.4f}")
    print(f"attribution shift on player 1 (expect {+BUMP / 6:+.4f}): "
          f"{shifted[1] - truth[1]:+.4f}")
