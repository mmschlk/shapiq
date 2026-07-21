from __future__ import annotations

from abc import abstractmethod
from math import comb
from typing import TYPE_CHECKING, Self

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln

from shapiq.sampling._schedule import UnitScheduleSampler

if TYPE_CHECKING:
    from collections.abc import Iterable

    from shapiq._shape import ShapeLike
    from shapiq.coalitions import CoalitionArray
    from shapiq.sampling._base import ShareSamples


def _log_binomial(n_players: int, sizes: Array) -> Array:
    """Return ``log(comb(n_players, sizes))`` via log-gamma, finite at any size."""
    total = gammaln(n_players + 1.0)
    return total - gammaln(sizes + 1.0) - gammaln(n_players - sizes + 1.0)


class KernelSampler(UnitScheduleSampler):
    """Base sampler for single-coalition units drawn from a coalition kernel.

    A kernel sampler draws every sampled coalition with probability
    proportional to a coalition kernel, so each sampled row enters an
    unweighted least squares fit of the kernel-matched index with the
    correct implicit weighting. The seed block holds the empty and grand
    coalition; each sampled unit is one coalition derived from
    ``fold_in(random_state, unit_index)``, so sampling does not depend on
    how a budget is split across calls. Two concrete mechanisms ship:
    ``SizeKernelSampler`` for distributions that depend only on coalition
    size and ``ProductKernelSampler`` for product measures. Wrap a kernel
    sampler in ``PairedSampler`` to add the complement coalition to every
    unit; that is variance-reducing only when the kernel is
    complement-symmetric.
    """

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

    @abstractmethod
    def _unit_from_key(self, unit_key: Array) -> Array:
        """Render the single-row unit a unit key stands for."""


class SizeKernelSampler(KernelSampler):
    """Sampler drawing coalitions from any size-based sampling distribution.

    ``size_weights`` holds one unnormalized sampling weight per coalition
    size ``0..n_players``. A sampled unit first draws a size from the
    normalized weights and then a uniform coalition of that size, so a
    coalition of size ``t`` appears with probability proportional to
    ``size_weights[t] / comb(n_players, t)``; sizes with zero weight are
    never sampled. Mind the distinction from a per-coalition kernel weight
    ``w(t)`` as declared by regression indices: sampling proportional to a
    coalition kernel ``w`` needs the size marginal
    ``comb(n_players, t) * w(t)``, which ``from_coalition_kernel`` builds.
    """

    def __init__(
        self,
        n_players: int,
        size_weights: Array | Iterable[float],
        target_shape: ShapeLike = (),
        *,
        share_samples: ShareSamples = False,
        random_state: Array | int = 0,
    ) -> None:
        """Initialize a size-kernel sampler.

        Args:
            n_players: Number of players in the explained game. Must be at
                least two.
            size_weights: One unnormalized sampling weight per coalition
                size ``0..n_players``, so ``n_players + 1`` non-negative
                finite entries with at least one positive.
            target_shape: Shape of the explanation targets, matching the
                game's target shape.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes. ``False`` samples independently per
                target; ``True`` shares across all target axes; an integer or
                tuple of integers shares across the selected axes.
            random_state: Integer seed or JAX PRNG key used to derive the
                sampled coalitions.

        Raises:
            ValueError: If ``n_players`` is smaller than two, or if the
                weights have the wrong length, contain negative or
                non-finite entries, or are all zero.
            TypeError: If ``random_state`` is neither an integer nor a JAX
                PRNG key.
        """
        super().__init__(
            n_players,
            target_shape,
            share_samples=share_samples,
            random_state=random_state,
        )
        weights = jnp.asarray(size_weights)
        if weights.ndim != 1 or weights.shape[0] != self.n_players + 1:
            msg = (
                "size_weights holds one weight per coalition size 0..n_players: "
                f"expected {self.n_players + 1} entries, got shape {tuple(weights.shape)}"
            )
            raise ValueError(msg)
        if not bool(jnp.all(jnp.isfinite(weights))) or bool(jnp.any(weights < 0)):
            msg = "size_weights must be non-negative and finite"
            raise ValueError(msg)
        support = jnp.flatnonzero(weights)
        if support.shape[0] == 0:
            msg = "size_weights must give at least one coalition size a positive weight"
            raise ValueError(msg)
        self._sizes = support
        self._size_probabilities = weights[support] / jnp.sum(weights[support])

    def log_probability(self, coalitions: CoalitionArray) -> Array:
        """Return the log-probability of drawing each given coalition.

        A coalition of size ``t`` is drawn with probability
        ``size_probability(t) / comb(n_players, t)``; sizes outside the
        sampled support answer ``-inf``. This is the sampler's
        ``LawfulSampler`` capability.
        """
        sizes = jnp.sum(jnp.asarray(coalitions.to_dense()), axis=-1).astype(jnp.int32)
        table = jnp.full(self.n_players + 1, -jnp.inf)
        table = table.at[self._sizes].set(
            jnp.log(self._size_probabilities) - _log_binomial(self.n_players, self._sizes),
        )
        return table[sizes]

    @classmethod
    def from_coalition_kernel(
        cls,
        n_players: int,
        kernel_weights: Array | Iterable[float],
        target_shape: ShapeLike = (),
        *,
        share_samples: ShareSamples = False,
        random_state: Array | int = 0,
    ) -> Self:
        """Return a sampler drawing coalitions proportional to a per-size kernel.

        Args:
            n_players: Number of players in the explained game.
            kernel_weights: One per-coalition kernel weight per size
                ``0..n_players``, as declared by ``regression_kernel``; the
                size marginal ``comb(n_players, t) * kernel_weights[t]`` is
                computed exactly in Python floats.
            target_shape: Shape of the explanation targets.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes.
            random_state: Integer seed or JAX PRNG key.

        Returns:
            A size-kernel sampler whose coalitions of size ``t`` appear with
            probability proportional to ``kernel_weights[t]``.
        """
        size_weights = [
            comb(n_players, size) * float(weight) for size, weight in enumerate(kernel_weights)
        ]
        return cls(
            n_players,
            jnp.asarray(size_weights),
            target_shape,
            share_samples=share_samples,
            random_state=random_state,
        )

    def _unit_from_key(self, unit_key: Array) -> Array:
        """Render the single-row unit a unit key stands for."""
        size_key, member_key = jax.random.split(unit_key)
        sizes = jax.random.choice(
            size_key,
            self._sizes,
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


class ShapleyKernelSampler(SizeKernelSampler):
    """Sampler drawing coalitions from the Shapley kernel size distribution.

    The seed block holds the empty and grand coalition. Each sampled unit
    draws a coalition whose size ``t`` follows the normalized Shapley kernel
    distribution ``p(t) proportional to 1 / (t * (n - t))`` over sizes ``1`` to
    ``n - 1`` and whose members are uniform given the size, so every sampled
    coalition appears with probability proportional to its kernel weight:
    the size weights are the per-coalition Shapley kernel times
    ``comb(n, t)``. Wrap the sampler in ``PairedSampler`` to add the
    complement coalition to every unit, which follows the same distribution
    by kernel symmetry. Units derive their randomness from
    ``fold_in(random_state, unit_index)``, so sampling does not depend on
    how a budget is split across calls.
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
        sizes = jnp.arange(1, n_players)
        weights = 1.0 / (sizes * (n_players - sizes))
        super().__init__(
            n_players,
            jnp.zeros(n_players + 1).at[1:n_players].set(weights),
            target_shape,
            share_samples=share_samples,
            random_state=random_state,
        )


class ProductKernelSampler(KernelSampler):
    """Sampler drawing coalitions from a product measure by membership flips.

    Every player joins a sampled coalition independently with probability
    ``p``, so a coalition of size ``t`` appears with probability
    ``p**t * (1 - p)**(n - t)`` — the product measure behind the weighted
    Banzhaf family, and the uniform (Banzhaf) kernel at ``p = 0.5``. The
    seed block holds the empty and grand coalition. The membership flips
    are exact for any number of players, and the mechanism extends to
    player-specific probabilities (a planned capability), which no
    size-based distribution can express. The product measure is
    complement-symmetric only at ``p = 0.5``, so pairing is reserved for
    the uniform case. Units derive their randomness from
    ``fold_in(random_state, unit_index)``, so sampling does not depend on
    how a budget is split across calls.
    """

    p: float

    def __init__(
        self,
        n_players: int,
        p: float = 0.5,
        target_shape: ShapeLike = (),
        *,
        share_samples: ShareSamples = False,
        random_state: Array | int = 0,
    ) -> None:
        """Initialize a product-kernel sampler.

        Args:
            n_players: Number of players in the explained game. Must be at
                least two.
            p: Probability with which each player independently joins a
                sampled coalition. Must satisfy ``0 < p < 1``.
            target_shape: Shape of the explanation targets, matching the
                game's target shape.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes. ``False`` samples independently per
                target; ``True`` shares across all target axes; an integer or
                tuple of integers shares across the selected axes.
            random_state: Integer seed or JAX PRNG key used to derive the
                sampled coalitions.

        Raises:
            ValueError: If ``n_players`` is smaller than two, or if ``p``
                lies outside the open unit interval.
            TypeError: If ``p`` is not a float, or if ``random_state`` is
                neither an integer nor a JAX PRNG key.
        """
        super().__init__(
            n_players,
            target_shape,
            share_samples=share_samples,
            random_state=random_state,
        )
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            msg = f"p must be a float, got {type(p).__name__}"
            raise TypeError(msg)
        if not 0.0 < p < 1.0:
            msg = (
                f"p must satisfy 0 < p < 1, got {p}; at the limits every sampled "
                "coalition would be the empty or the grand coalition"
            )
            raise ValueError(msg)
        self.p = float(p)

    def log_probability(self, coalitions: CoalitionArray) -> Array:
        """Return the log-probability of flipping each given coalition.

        A coalition of size ``t`` is drawn with probability
        ``p**t * (1 - p)**(n - t)``; the product measure has full support,
        including the empty and grand coalition. This is the sampler's
        ``LawfulSampler`` capability.
        """
        sizes = jnp.sum(jnp.asarray(coalitions.to_dense()), axis=-1)
        return sizes * jnp.log(self.p) + (self.n_players - sizes) * jnp.log1p(-self.p)

    def _unit_from_key(self, unit_key: Array) -> Array:
        """Render the single-row unit a unit key stands for."""
        mask = jax.random.bernoulli(
            unit_key,
            self.p,
            (*self.shared_target_shape, self.n_players),
        )
        return mask[..., None, :]


class BanzhafKernelSampler(ProductKernelSampler):
    """Sampler drawing coalitions uniformly from the powerset.

    The uniform distribution is the Banzhaf kernel — the product measure at
    ``p = 0.5``: every coalition appears with probability ``2**-n_players``,
    realized by independent fair membership coin flips per player, so
    sampled rows enter an unweighted least squares fit of a uniform-kernel
    index with the correct implicit weighting. The seed block holds the
    empty and grand coalition. Wrap the sampler in ``PairedSampler`` to add
    the complement coalition to every unit, which is uniform by symmetry.
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
            0.5,
            target_shape,
            share_samples=share_samples,
            random_state=random_state,
        )
