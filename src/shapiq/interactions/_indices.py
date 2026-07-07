"""Interaction index objects, their capability protocols, and generalizations."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from math import comb
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import jax.numpy as jnp

from shapiq._shape import validate_int

if TYPE_CHECKING:
    from jax import Array

    from shapiq.interactions._types import (
        InteractionIndexName,
        InteractionOrientation,
        OrderSemantics,
    )


@runtime_checkable
class InteractionIndex(Protocol):
    """An interaction index as an immutable value object.

    Explainers select behavior by index type and capability, never by name;
    the ``name`` is explanation metadata. The order semantics record whether
    the order is explanation coverage (attributions of shared interactions
    are unchanged across orders, as for SII and BII) or part of the index
    identity (attributions change with the order, as for STII and FSII).
    Indices that generalize a probabilistic value declare it: their order-1
    restriction equals that value, and the declaration is tested numerically.
    """

    @property
    def name(self) -> InteractionIndexName:
        """Return the name recorded on explanations."""
        ...

    @property
    def order(self) -> int | None:
        """Return the maximum interaction order, or ``None`` for all orders."""
        ...

    @property
    def order_semantics(self) -> OrderSemantics:
        """Return whether order is explanation coverage or index identity."""
        ...

    @property
    def orientation(self) -> InteractionOrientation:
        """Return whether the index attributes to sets or ordered tuples."""
        ...

    @property
    def includes_empty_interaction(self) -> bool:
        """Return whether explanations carry an order-0 attribution."""
        ...

    @property
    def generalizes(self) -> SV | BV | None:
        """Return the probabilistic value this index restricts to at order 1."""
        ...


@runtime_checkable
class CardinalInteractionIndex(InteractionIndex, Protocol):
    """Capability: attributions are cardinality-weighted discrete derivatives.

    Cardinal interaction indices assign to an interaction ``S`` a weighted
    sum of its discrete derivatives over outside coalitions ``T``, with
    weights depending only on the cardinalities of ``S`` and ``T``.
    """

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return one weight per outside-coalition size ``0..n - s``."""
        ...

    @property
    def min_interaction_size(self) -> int:
        """Return the smallest represented interaction size (0 for transforms)."""
        ...


@runtime_checkable
class GeneralizedValueIndex(InteractionIndex, Protocol):
    """Capability: attributions are cardinality-weighted bloc marginals.

    Generalized values assign to an interaction ``S`` a weighted sum of the
    marginal contributions ``v(T | S) - v(T)`` of the whole interaction
    joining outside coalitions ``T``, with weights depending only on
    cardinalities.
    """

    def marginal_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return one weight per outside-coalition size ``0..n - s``."""
        ...


@runtime_checkable
class AggregationIndex(InteractionIndex, Protocol):
    """Capability: attributions aggregate a base index over supersets.

    The attribution of an interaction ``S`` sums the base attributions of
    its supersets ``T`` up to the order, weighted by a coefficient that
    depends only on ``|T| - |S|``. Aggregation is linear in the base index,
    so exact and unbiased sampled estimators of the base carry over.
    """

    @property
    def base_index(self) -> InteractionIndex:
        """Return the index whose attributions are aggregated."""
        ...

    def aggregation_coefficients(self) -> tuple[float, ...]:
        """Return one aggregation coefficient per superset size difference."""
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
    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = True
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[None] = None

    @property
    def order(self) -> int:
        """Return ``1``: the Shapley value attributes to single players only."""
        return 1

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return Shapley discrete-derivative weights per outside size."""
        return _shapley_derivative_weights(n_players, interaction_size)

    def regression_kernel(self, n_players: int) -> Array:
        """Return Shapley kernel weights per coalition size, zero at the ends.

        The Shapley value is also the constrained kernel regression of order
        one (KernelSHAP), so SV satisfies both capabilities; explainers with
        a cheaper discrete-derivative path prefer it.
        """
        return _shapley_regression_kernel(n_players)


@dataclass(frozen=True)
class BV:
    """The Banzhaf value: uniform-coalition attribution to single players."""

    name: ClassVar[InteractionIndexName] = "BV"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = True
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[None] = None

    @property
    def order(self) -> int:
        """Return ``1``: the Banzhaf value attributes to single players only."""
        return 1

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return Banzhaf discrete-derivative weights per outside size."""
        return _banzhaf_derivative_weights(n_players, interaction_size)


@dataclass(frozen=True)
class SII:
    """The Shapley interaction index up to ``order``; generalizes SV.

    Order is explanation coverage: attributions of shared interactions are
    identical across orders, so a higher order only adds interactions.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "SII"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[SV] = SV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return Shapley discrete-derivative weights per outside size."""
        return _shapley_derivative_weights(n_players, interaction_size)


@dataclass(frozen=True)
class BII:
    """The Banzhaf interaction index up to ``order``; generalizes BV.

    Order is explanation coverage: attributions of shared interactions are
    identical across orders, so a higher order only adds interactions.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "BII"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[BV] = BV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return Banzhaf discrete-derivative weights per outside size."""
        return _banzhaf_derivative_weights(n_players, interaction_size)


@dataclass(frozen=True)
class CHII:
    """The chaining interaction index up to ``order``; generalizes SV.

    Order is explanation coverage: attributions of shared interactions are
    identical across orders. The chaining weights treat the interaction as
    the head of a chain of arrivals.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "CHII"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[SV] = SV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return chaining discrete-derivative weights per outside size."""
        return _chaining_derivative_weights(n_players, interaction_size)


@dataclass(frozen=True)
class STII:
    """The Shapley-Taylor interaction index of top order ``order``; generalizes SV.

    Order is part of the index identity: attributions of shared interactions
    change with the order. Interactions below the top order are discrete
    derivatives at the empty coalition; the top order distributes the
    remaining game mass along permutations.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "STII"
    order_semantics: ClassVar[OrderSemantics] = "identity"
    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = True
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[SV] = SV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return anchored weights below the top order, Taylor weights at it."""
        if interaction_size < self.order:
            return _empty_anchor_weights(n_players, interaction_size)
        return _taylor_top_weights(n_players, interaction_size)


@dataclass(frozen=True)
class KSII:
    """The efficient k-Shapley interaction index of order ``order``; generalizes SV.

    Order is part of the index identity: k-SII aggregates Shapley
    interactions of all orders up to ``order`` into an efficient explanation
    via Bernoulli numbers, so attributions of shared interactions change
    with the order.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "k-SII"
    order_semantics: ClassVar[OrderSemantics] = "identity"
    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = True
    generalizes: ClassVar[SV] = SV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    @property
    def base_index(self) -> SII:
        """Return the Shapley interaction index the aggregation starts from."""
        return SII(order=self.order)

    def aggregation_coefficients(self) -> tuple[float, ...]:
        """Return the Bernoulli numbers weighting supersets by size difference."""
        return bernoulli_numbers(self.order)


@dataclass(frozen=True)
class FSII:
    """The faithful Shapley interaction index of order ``order``; generalizes SV.

    Order is part of the index identity: the index is the best
    ``order``-additive approximation of the game under the Shapley kernel,
    interpolating the empty and grand coalition exactly.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "FSII"
    order_semantics: ClassVar[OrderSemantics] = "identity"
    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = True
    generalizes: ClassVar[SV] = SV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def regression_kernel(self, n_players: int) -> Array:
        """Return Shapley kernel weights per coalition size, zero at the ends."""
        return _shapley_regression_kernel(n_players)


@dataclass(frozen=True)
class FBII:
    """The faithful Banzhaf interaction index of order ``order``; generalizes BV.

    Order is part of the index identity: the index is the best
    ``order``-additive approximation of the game under the uniform kernel.
    The fit is unconstrained with a free intercept, so the order-0
    attribution is the fitted intercept rather than the empty-coalition
    value.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "FBII"
    order_semantics: ClassVar[OrderSemantics] = "identity"
    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = True
    generalizes: ClassVar[BV] = BV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)


@dataclass(frozen=True)
class KADDSHAP:
    """The k-additive Shapley index of order ``order``; generalizes SV.

    Order is part of the index identity: the index fits a ``order``-additive
    game in the Bernoulli-weighted interaction basis under the Shapley
    kernel, interpolating the grand coalition exactly, so its order-1
    attributions remain Shapley values at every order.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "kADD-SHAP"
    order_semantics: ClassVar[OrderSemantics] = "identity"
    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = True
    generalizes: ClassVar[SV] = SV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def regression_kernel(self, n_players: int) -> Array:
        """Return Shapley kernel weights per coalition size, zero at the ends."""
        return _shapley_regression_kernel(n_players)


@dataclass(frozen=True)
class Moebius:
    """The Moebius transform: the game's interaction masses themselves.

    A cardinal index anchored at the empty coalition; the default order
    ``None`` represents every interaction up to the grand coalition. The
    order-0 attribution is the empty-coalition value.
    """

    order: int | None = None

    name: ClassVar[InteractionIndexName] = "Moebius"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = True
    min_interaction_size: ClassVar[int] = 0
    generalizes: ClassVar[None] = None

    def __post_init__(self) -> None:
        """Validate the order when one is given."""
        if self.order is not None:
            validate_int("order", self.order, minimum=1)

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return discrete-derivative-at-empty weights per outside size."""
        return _empty_anchor_weights(n_players, interaction_size)


@dataclass(frozen=True)
class CoMoebius:
    """The Co-Moebius transform: derivatives anchored at the complement.

    A cardinal index whose derivative of an interaction is taken at the
    coalition of all remaining players; the default order ``None``
    represents every interaction. The order-0 attribution is the
    grand-coalition value.
    """

    order: int | None = None

    name: ClassVar[InteractionIndexName] = "Co-Moebius"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = True
    min_interaction_size: ClassVar[int] = 0
    generalizes: ClassVar[None] = None

    def __post_init__(self) -> None:
        """Validate the order when one is given."""
        if self.order is not None:
            validate_int("order", self.order, minimum=1)

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return discrete-derivative-at-complement weights per outside size."""
        return _grand_anchor_weights(n_players, interaction_size)


@dataclass(frozen=True)
class SGV:
    """The Shapley generalized value up to ``order``; generalizes SV.

    Attributions weight the marginal contributions of whole interactions
    joining outside coalitions with Shapley weights. Order is explanation
    coverage.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "SGV"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[SV] = SV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def marginal_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return Shapley bloc-marginal weights per outside size."""
        return _shapley_derivative_weights(n_players, interaction_size)


@dataclass(frozen=True)
class BGV:
    """The Banzhaf generalized value up to ``order``; generalizes BV.

    Attributions weight the marginal contributions of whole interactions
    joining outside coalitions uniformly. Order is explanation coverage.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "BGV"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[BV] = BV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def marginal_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return Banzhaf bloc-marginal weights per outside size."""
        return _banzhaf_derivative_weights(n_players, interaction_size)


@dataclass(frozen=True)
class CHGV:
    """The chaining generalized value up to ``order``; generalizes SV.

    Attributions weight the marginal contributions of whole interactions
    joining outside coalitions with chaining weights. Order is explanation
    coverage.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "CHGV"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[SV] = SV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def marginal_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return chaining bloc-marginal weights per outside size."""
        return _chaining_derivative_weights(n_players, interaction_size)


@dataclass(frozen=True)
class IGV:
    """The internal generalized value up to ``order``.

    The attribution of an interaction is its stand-alone worth over the
    empty coalition, ``v(S) - v(empty)``. Order is explanation coverage.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "IGV"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[None] = None

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def marginal_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return bloc-marginal weights anchored at the empty coalition."""
        return _empty_anchor_weights(n_players, interaction_size)


@dataclass(frozen=True)
class EGV:
    """The external generalized value up to ``order``.

    The attribution of an interaction is its contribution on top of all
    remaining players, ``v(N) - v(N without S)``. Order is explanation
    coverage.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "EGV"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[None] = None

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def marginal_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return bloc-marginal weights anchored at the complement."""
        return _grand_anchor_weights(n_players, interaction_size)


@dataclass(frozen=True)
class JointSV:
    """The joint Shapley value of order ``order``; generalizes SV.

    Attributions weight bloc marginal contributions with arrival-process
    weights in which coalitions of up to ``order`` players arrive together,
    so the explanation is efficient. Order is part of the index identity.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "JointSV"
    order_semantics: ClassVar[OrderSemantics] = "identity"
    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = True
    generalizes: ClassVar[SV] = SV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def marginal_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return joint-arrival bloc-marginal weights per outside size."""
        return _joint_arrival_weights(n_players, self.order)[: n_players - interaction_size + 1]


@cache
def bernoulli_numbers(order: int) -> tuple[float, ...]:
    """Return the Bernoulli numbers up to ``order`` with the B(1) = -1/2 convention."""
    numbers = [Fraction(1)]
    for m in range(1, order + 1):
        acc = sum((Fraction(comb(m + 1, j)) * numbers[j] for j in range(m)), Fraction(0))
        numbers.append(-acc / (m + 1))
    return tuple(float(number) for number in numbers)


def _shapley_regression_kernel(n_players: int) -> Array:
    """Return the Shapley kernel per coalition size with zero-weight endpoints."""
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


def _chaining_derivative_weights(n_players: int, size: int) -> Array:
    """Return CHII discrete-derivative weights per outside-coalition size."""
    return jnp.asarray(
        [size / ((size + t) * comb(n_players, size + t)) for t in range(n_players - size + 1)],
    )


def _empty_anchor_weights(n_players: int, size: int) -> Array:
    """Return discrete-derivative-at-empty weights per outside-coalition size."""
    return jnp.zeros(n_players - size + 1).at[0].set(1.0)


def _grand_anchor_weights(n_players: int, size: int) -> Array:
    """Return discrete-derivative-at-complement weights per outside-coalition size."""
    return jnp.zeros(n_players - size + 1).at[-1].set(1.0)


def _taylor_top_weights(n_players: int, size: int) -> Array:
    """Return top-order Shapley-Taylor weights per outside-coalition size."""
    return jnp.asarray(
        [size / (n_players * comb(n_players - 1, t)) for t in range(n_players - size + 1)],
    )


@cache
def _joint_arrival_weights(n_players: int, order: int) -> Array:
    """Return joint Shapley weights per coalition size from the arrival recursion."""
    weights = [0.0] * n_players
    weights[0] = 1.0 / sum(comb(n_players, size) for size in range(1, order + 1))
    for reached in range(1, n_players):
        largest_step = min(order, n_players - reached)
        smallest_origin = max(reached - order, 0)
        denominator = sum(comb(n_players - reached, size) for size in range(1, largest_step + 1))
        numerator = sum(
            comb(reached, size) * weights[size] for size in range(smallest_origin, reached)
        )
        weights[reached] = numerator / denominator
    return jnp.asarray(weights)
