from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from shapiq.sampling._base import Sampler

if TYPE_CHECKING:
    from jax import Array


class PermutationSampler(Sampler):
    """Sampler drawing random permutations, full stop.

    A draw is one permutation, given as each player's position; what
    coalitions a permutation materializes into is the consuming
    approximator's business. Draw ``unit_index`` derives from
    ``fold_in(random_state, unit_index)``, so draws are order-free, and the
    antithesis of a permutation is its reversal — wrap the sampler in
    ``PairedSampler`` to draw both.
    """

    def draws(self, unit_indices: Array) -> Array:
        """Return each player's position for one permutation per unit index."""

        def one(unit_index: Array) -> Array:
            walk_key = jax.random.fold_in(self._key, unit_index)
            players = jnp.broadcast_to(
                jnp.arange(self.n_players),
                (*self.shared_target_shape, self.n_players),
            )
            permutation = jax.random.permutation(walk_key, players, axis=-1, independent=True)
            return jnp.argsort(permutation, axis=-1)

        return jax.vmap(one)(jnp.asarray(unit_indices))

    def antithetic(self, draws: Array) -> Array:
        """Return the positions of the reversed permutations."""
        return self.n_players - 1 - draws
