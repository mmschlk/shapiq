"""Interaction index objects and their capability protocols."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import jax.numpy as jnp

from shapiq._shape import validate_int

if TYPE_CHECKING:
    from jax import Array

    from shapiq.interactions._types import InteractionIndexName, OrderSemantics


@runtime_checkable
class InteractionIndex(Protocol):
    """An interaction index as an immutable value object.

    Explainers select behavior by index type and capability, never by name;
    the ``name`` is explanation metadata. The order semantics record whether
    the order is explanation coverage (attributions of shared interactions
    are unchanged across orders, as for SII and BII) or part of the index
    identity (attributions change with the order, as for STII and FSII).
    """

    @property
    def name(self) -> InteractionIndexName:
        """Return the name recorded on explanations."""
        ...

    @property
    def order(self) -> int:
        """Return the maximum interaction order of the explanation."""
        ...

    @property
    def order_semantics(self) -> OrderSemantics:
        """Return whether order is explanation coverage or index identity."""
        ...

    @property
    def includes_empty_interaction(self) -> bool:
        """Return whether explanations carry ``v(empty)`` at order zero."""
        ...


@runtime_checkable
class WeightedDerivativeIndex(InteractionIndex, Protocol):
    """Capability: attributions are weighted sums of discrete derivatives."""

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return one weight per outside-coalition size ``0..n - s``.

        The attribution of an interaction ``S`` with ``s`` players is the sum
        of its discrete derivatives over all coalitions ``T`` of players
        outside ``S``, weighted by the returned weight at ``|T|``.
        """
        ...


@runtime_checkable
class RegressionIndex(InteractionIndex, Protocol):
    """Capability: attributions solve a kernel-weighted least squares fit."""

    def regression_kernel(self, n_players: int) -> Array:
        """Return one kernel weight per coalition size ``0..n``.

        The empty and grand coalition carry weight zero; they are
        interpolated exactly as constraints rather than weighted.
        """
        ...


@dataclass(frozen=True)
class SV:
    """The Shapley value: the unique efficient attribution to single players."""

    name: ClassVar[InteractionIndexName] = "SV"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    includes_empty_interaction: ClassVar[bool] = True

    @property
    def order(self) -> int:
        """Return ``1``: the Shapley value attributes to single players only."""
        return 1

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return Shapley discrete-derivative weights per outside size."""
        return _shapley_derivative_weights(n_players, interaction_size)


@dataclass(frozen=True)
class BV:
    """The Banzhaf value: uniform-coalition attribution to single players."""

    name: ClassVar[InteractionIndexName] = "BV"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    includes_empty_interaction: ClassVar[bool] = True

    @property
    def order(self) -> int:
        """Return ``1``: the Banzhaf value attributes to single players only."""
        return 1

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return Banzhaf discrete-derivative weights per outside size."""
        return _banzhaf_derivative_weights(n_players, interaction_size)


@dataclass(frozen=True)
class SII:
    """The Shapley interaction index up to ``order``.

    Order is explanation coverage: attributions of shared interactions are
    identical across orders, so a higher order only adds interactions.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "SII"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    includes_empty_interaction: ClassVar[bool] = False

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return Shapley discrete-derivative weights per outside size."""
        return _shapley_derivative_weights(n_players, interaction_size)


@dataclass(frozen=True)
class BII:
    """The Banzhaf interaction index up to ``order``.

    Order is explanation coverage: attributions of shared interactions are
    identical across orders, so a higher order only adds interactions.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "BII"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    includes_empty_interaction: ClassVar[bool] = False

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return Banzhaf discrete-derivative weights per outside size."""
        return _banzhaf_derivative_weights(n_players, interaction_size)


@dataclass(frozen=True)
class STII:
    """The Shapley-Taylor interaction index of top order ``order``.

    Order is part of the index identity: attributions of shared interactions
    change with the order. Interactions below the top order are discrete
    derivatives at the empty coalition; the top order distributes the
    remaining game mass along permutations.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "STII"
    order_semantics: ClassVar[OrderSemantics] = "identity"
    includes_empty_interaction: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return anchored weights below the top order, Taylor weights at it."""
        if interaction_size < self.order:
            return _empty_anchor_weights(n_players, interaction_size)
        return _taylor_top_weights(n_players, interaction_size)


@dataclass(frozen=True)
class FSII:
    """The faithful Shapley interaction index of order ``order``.

    Order is part of the index identity: the index is the best
    ``order``-additive approximation of the game under the Shapley kernel,
    interpolating the empty and grand coalition exactly.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "FSII"
    order_semantics: ClassVar[OrderSemantics] = "identity"
    includes_empty_interaction: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def regression_kernel(self, n_players: int) -> Array:
        """Return Shapley kernel weights per coalition size, zero at the ends."""
        interior = [
            1.0 / ((n_players - 1) * comb(n_players - 2, size - 1)) for size in range(1, n_players)
        ]
        return jnp.asarray([0.0, *interior, 0.0])


def _shapley_derivative_weights(n_players: int, size: int) -> Array:
    """Return SII discrete-derivative weights per outside-coalition size."""
    free = n_players - size
    return jnp.asarray([1.0 / ((free + 1) * comb(free, t)) for t in range(free + 1)])


def _banzhaf_derivative_weights(n_players: int, size: int) -> Array:
    """Return BII discrete-derivative weights per outside-coalition size."""
    free = n_players - size
    return jnp.full(free + 1, 2.0**-free)


def _empty_anchor_weights(n_players: int, size: int) -> Array:
    """Return discrete-derivative-at-empty weights per outside-coalition size."""
    return jnp.zeros(n_players - size + 1).at[0].set(1.0)


def _taylor_top_weights(n_players: int, size: int) -> Array:
    """Return top-order Shapley-Taylor weights per outside-coalition size."""
    return jnp.asarray(
        [size / (n_players * comb(n_players - 1, t)) for t in range(n_players - size + 1)],
    )
