from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from shapiq.sampling._schedule import UnitScheduleSampler

if TYPE_CHECKING:
    from shapiq._shape import ShapeLike
    from shapiq.sampling._base import ShareSamples


class ShapleyKernelSampler(UnitScheduleSampler):
    """Sampler drawing coalitions from the Shapley kernel size distribution.

    The seed block holds the empty and grand coalition. Each sampled unit
    draws a coalition whose size ``t`` follows the normalized Shapley kernel
    distribution ``p(t) proportional to 1 / (t * (n - t))`` over sizes ``1`` to
    ``n - 1`` and whose members are uniform given the size, so every sampled
    coalition appears with probability proportional to its kernel weight.
    Wrap the sampler in ``PairedSampler`` to add the complement coalition to
    every unit, which follows the same distribution by kernel symmetry.
    Units derive their randomness from ``fold_in(random_state, unit_index)``,
    so sampling does not depend on how a budget is split across calls.
    """

    def __init__(
        self,
        n_players: int,
        target_shape: ShapeLike = (),
        *,
        share_samples: ShareSamples = False,
        random_state: Array | int = 0,
    ) -> None:
        """Initialize a Shapley kernel sampler.

        Args:
            n_players: Number of players in the explained game. Must be at
                least two.
            target_shape: Shape of the explanation targets, matching the
                game's target shape.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes. ``False`` samples independently per
                target; ``True`` shares across all target axes; an integer or
                tuple of integers shares across the selected axes.
            random_state: Integer seed or JAX PRNG key used to derive the
                sampled coalitions.

        Raises:
            ValueError: If ``n_players`` is smaller than two.
            TypeError: If ``random_state`` is neither an integer nor a JAX
                PRNG key.
        """
        super().__init__(
            n_players,
            target_shape,
            share_samples=share_samples,
            random_state=random_state,
        )
        sizes = jnp.arange(1, self.n_players)
        weights = 1.0 / (sizes * (self.n_players - sizes))
        self._size_probabilities = weights / jnp.sum(weights)

    @property
    def sampling_quantum(self) -> int:
        """Return the unit length: one coalition."""
        return 1

    def _sampled_unit_masks(self, unit_index: int) -> Array:
        """Return one kernel-distributed coalition as a single-row unit."""
        return self._unit_from_key(jax.random.fold_in(self._key, unit_index))

    def _sampled_unit_batch(self, unit_indices: Array) -> Array:
        """Return many kernel-distributed units in a few vectorized dispatches."""
        return jax.vmap(self._unit_from_key)(self._unit_keys(unit_indices))

    def _unit_from_key(self, unit_key: Array) -> Array:
        """Render the single-row unit a unit key stands for."""
        size_key, member_key = jax.random.split(unit_key)
        sizes = jax.random.choice(
            size_key,
            jnp.arange(1, self.n_players),
            shape=self.shared_target_shape,
            p=self._size_probabilities,
        )
        players = jnp.broadcast_to(
            jnp.arange(self.n_players),
            (*self.shared_target_shape, self.n_players),
        )
        permutation = jax.random.permutation(member_key, players, axis=-1, independent=True)
        positions = jnp.argsort(permutation, axis=-1)
        mask = positions < sizes[..., None]
        return mask[..., None, :]


class BanzhafKernelSampler(UnitScheduleSampler):
    """Sampler drawing coalitions uniformly from the powerset.

    The uniform distribution is the Banzhaf kernel: every coalition appears
    with probability ``2**-n_players``, realized by independent fair
    membership coin flips per player, so sampled rows enter an unweighted
    least squares fit of a uniform-kernel index with the correct implicit
    weighting. The seed block holds the empty and grand coalition. Wrap the
    sampler in ``PairedSampler`` to add the complement coalition to every
    unit, which is uniform by symmetry. Units derive their randomness
    from ``fold_in(random_state, unit_index)``, so sampling does not depend
    on how a budget is split across calls.
    """

    def __init__(
        self,
        n_players: int,
        target_shape: ShapeLike = (),
        *,
        share_samples: ShareSamples = False,
        random_state: Array | int = 0,
    ) -> None:
        """Initialize a Banzhaf kernel sampler.

        Args:
            n_players: Number of players in the explained game.
            target_shape: Shape of the explanation targets, matching the
                game's target shape.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes. ``False`` samples independently per
                target; ``True`` shares across all target axes; an integer or
                tuple of integers shares across the selected axes.
            random_state: Integer seed or JAX PRNG key used to derive the
                sampled coalitions.

        Raises:
            TypeError: If ``random_state`` is neither an integer nor a JAX
                PRNG key.
        """
        super().__init__(
            n_players,
            target_shape,
            share_samples=share_samples,
            random_state=random_state,
        )

    @property
    def sampling_quantum(self) -> int:
        """Return the unit length: one coalition."""
        return 1

    def _sampled_unit_masks(self, unit_index: int) -> Array:
        """Return one uniform coalition as a single-row unit."""
        return self._unit_from_key(jax.random.fold_in(self._key, unit_index))

    def _sampled_unit_batch(self, unit_indices: Array) -> Array:
        """Return many uniform units in a few vectorized dispatches."""
        return jax.vmap(self._unit_from_key)(self._unit_keys(unit_indices))

    def _unit_from_key(self, unit_key: Array) -> Array:
        """Render the single-row unit a unit key stands for."""
        mask = jax.random.bernoulli(
            unit_key,
            0.5,
            (*self.shared_target_shape, self.n_players),
        )
        return mask[..., None, :]
