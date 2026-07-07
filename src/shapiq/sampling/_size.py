from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from shapiq._shape import ensure_bool
from shapiq.sampling._schedule import UnitScheduleSampler

if TYPE_CHECKING:
    from shapiq._shape import ShapeLike
    from shapiq.sampling._base import ShareSamples


class CoalitionSizeSampler(UnitScheduleSampler):
    """Sampler drawing coalitions whose size follows a given weight profile.

    The seed block holds the empty and grand coalition; the weights of those
    two sizes are therefore ignored, and each sampled unit draws a coalition
    whose size follows the normalized interior weights and whose members are
    uniform given the size. Sizes with zero weight are never sampled, so a
    weight profile derived from a coalition functional's mass concentrates
    evaluations where the index has coefficients at all. With pairing enabled
    a unit also contains the complement coalition and the interior weights
    are symmetrized, which keeps every emitted coalition's marginal
    probability equal to ``size_probabilities[size] / comb(n, size)``.
    """

    paired: bool

    def __init__(
        self,
        n_players: int,
        target_shape: ShapeLike = (),
        *,
        size_weights: Array,
        share_samples: ShareSamples = False,
        paired: bool = False,
        random_state: Array | int = 0,
    ) -> None:
        """Initialize a coalition size sampler.

        Args:
            n_players: Number of players in the explained game. Must be at
                least two.
            target_shape: Shape of the explanation targets, matching the
                game's target shape.
            size_weights: One nonnegative weight per coalition size
                ``0..n_players``, such as a coalition functional's size mass.
                The empty and grand coalition weights are ignored; the
                remaining weights are normalized into the size distribution.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes. ``False`` samples independently per
                target; ``True`` shares across all target axes; an integer or
                tuple of integers shares across the selected axes.
            paired: Whether each unit also contains the complement of the
                drawn coalition, symmetrizing the size distribution.
            random_state: Integer seed or JAX PRNG key used to derive the
                sampled coalitions.

        Raises:
            ValueError: If ``n_players`` is smaller than two, if the weights
                have the wrong shape or are negative or non-finite, or if no
                interior coalition size carries weight.
            TypeError: If ``paired`` is not a bool, or if ``random_state`` is
                neither an integer nor a JAX PRNG key.
        """
        super().__init__(
            n_players,
            target_shape,
            share_samples=share_samples,
            random_state=random_state,
        )
        self.paired = ensure_bool("paired", paired)
        weights = jnp.asarray(size_weights, dtype=jnp.float32)
        if weights.shape != (self.n_players + 1,):
            msg = (
                "size_weights must hold one weight per coalition size "
                f"0..{self.n_players}, shape ({self.n_players + 1},), got {weights.shape}"
            )
            raise ValueError(msg)
        if not bool(jnp.all(jnp.isfinite(weights)) & jnp.all(weights >= 0)):
            msg = "size_weights must be finite and nonnegative"
            raise ValueError(msg)
        interior = weights.at[0].set(0.0).at[self.n_players].set(0.0)
        if self.paired:
            interior = (interior + interior[::-1]) / 2.0
        total = jnp.sum(interior)
        if not bool(total > 0):
            msg = (
                "no coalition size between the empty and grand coalition carries "
                "weight; the index is exact from the seed block, use ExactExplainer"
            )
            raise ValueError(msg)
        self._size_probabilities = interior / total

    @property
    def size_probabilities(self) -> Array:
        """Return the sampled-size distribution over sizes ``0..n_players``.

        The empty and grand coalition carry probability zero; a sampled
        coalition of size ``k`` has marginal probability
        ``size_probabilities[k] / comb(n_players, k)``.
        """
        return self._size_probabilities

    @property
    def sampling_quantum(self) -> int:
        """Return the unit length in coalitions: two when paired, else one."""
        return 2 if self.paired else 1

    def _sampled_unit_masks(self, unit_index: int) -> Array:
        """Return one size-distributed coalition, plus its complement if paired."""
        unit_key = jax.random.fold_in(self._key, unit_index)
        size_key, member_key = jax.random.split(unit_key)
        sizes = jax.random.choice(
            size_key,
            jnp.arange(1, self.n_players),
            shape=self.shared_target_shape,
            p=self._size_probabilities[1 : self.n_players],
        )
        players = jnp.broadcast_to(
            jnp.arange(self.n_players),
            (*self.shared_target_shape, self.n_players),
        )
        permutation = jax.random.permutation(member_key, players, axis=-1, independent=True)
        positions = jnp.argsort(permutation, axis=-1)
        mask = positions < sizes[..., None]
        if self.paired:
            return jnp.stack([mask, ~mask], axis=-2)
        return mask[..., None, :]
