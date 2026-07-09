"""Interaction index objects, their capability protocols, and generalizations."""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import cache
from math import comb
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast, runtime_checkable

import jax.numpy as jnp

from shapiq._shape import validate_int

if TYPE_CHECKING:
    from jax import Array

    from shapiq.interactions._types import (
        InteractionIndexName,
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
    def name(self) -> str:
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
    def includes_empty_interaction(self) -> bool:
        """Return whether explanations carry an order-0 attribution."""
        ...

    @property
    def generalizes(self) -> InteractionIndex | None:
        """Return the probabilistic value this index restricts to at order 1."""
        ...

    @property
    def preserves_value(self) -> bool:
        """Return whether order-1 attributions are identical at every explanation order.

        For a generalizing index this means the generalized value stays
        readable off the order-1 attributions at any order; indices with no
        generalized value preserve trivially through coverage semantics.
        """
        ...


@runtime_checkable
class CardinalInteractionIndex(InteractionIndex, Protocol):
    """Capability: attributions are cardinality-weighted discrete derivatives.

    Cardinal interaction indices assign to an interaction ``S`` a weighted
    sum of its discrete derivatives over outside coalitions ``T``, with
    weights depending only on the cardinalities of ``S`` and ``T``.
    Player-specific weightings (weighted Shapley values, per-player joining
    probabilities in the weighted Banzhaf family, Owen values under a
    coalition structure) are a separate future capability, not a
    generalization of this one.
    """

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return one weight per outside-coalition size ``0..n - s``."""
        ...

    @property
    def min_interaction_size(self) -> int:
        """Return the smallest represented interaction size.

        Zero when the rule attributes to the empty interaction on the
        centered game, as for the Co-Moebius transform.
        """
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
class RegressionIndex(InteractionIndex, Protocol):
    """Capability: attributions solve a kernel-weighted least squares fit."""

    def regression_kernel(self, n_players: int) -> Array:
        """Return one kernel weight per coalition size ``0..n``.

        A zero weight at the empty or grand coalition marks that coalition
        as an exact interpolation constraint rather than a weighted row;
        nonzero end weights (the uniform Banzhaf kernel) mean every
        coalition is fitted and the fit carries a free intercept.
        """
        ...


class ExtensionalEquality(InteractionIndex):
    """Equality of interaction indices as attribution rules on nonempty interactions.

    Instances constructed at order one collapse onto the probabilistic value
    they generalize, so ``SII(order=1) == SV() == CHII(order=1)`` and hashes
    agree; all other instances compare by type and parameters. The equality
    quantifies over nonempty interactions only — genuine order-0
    attributions (FBII's fitted intercept, the Co-Moebius grand total)
    remain per-index, and the baseline travels on explanations.
    Estimator dispatch is keyed on index types and is unaffected.
    """

    def _identity(self) -> tuple[object, ...]:
        generalized = self.generalizes
        if isinstance(generalized, ExtensionalEquality) and self.order == 1:
            return ExtensionalEquality._identity(generalized)
        params = tuple(getattr(self, field.name) for field in fields(cast("Any", self)))
        return (type(self).__name__, *params)

    def __eq__(self, other: object) -> bool:
        """Compare attribution rules, collapsing order-1 generalizations."""
        if not isinstance(other, ExtensionalEquality):
            return NotImplemented
        return self._identity() == other._identity()

    def __hash__(self) -> int:
        """Hash consistently with extensional equality."""
        return hash(self._identity())


@dataclass(frozen=True, eq=False)
class SV(ExtensionalEquality):
    """The Shapley value: the unique efficient attribution to single players."""

    name: ClassVar[InteractionIndexName] = "SV"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[None] = None

    @property
    def order(self) -> int:
        """Return ``1``: the Shapley value attributes to single players only."""
        return 1

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return Shapley discrete-derivative weights per outside size."""
        return _shapley_derivative_weights(n_players, interaction_size)

    def marginal_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return Shapley bloc-marginal weights for single players.

        The Shapley value is the singleton restriction of the Shapley
        generalized value: for an interaction of size one the discrete
        derivative and the bloc marginal coincide, so SV satisfies the
        generalized-value capability there and nowhere else.
        """
        if interaction_size != 1:
            msg = f"SV attributes to single players only, got interaction_size={interaction_size}"
            raise ValueError(msg)
        return _shapley_derivative_weights(n_players, 1)

    def regression_kernel(self, n_players: int) -> Array:
        """Return Shapley kernel weights per coalition size, zero at the ends.

        The Shapley value is also the constrained kernel regression of order
        one (KernelSHAP), so SV satisfies both capabilities; explainers with
        a cheaper discrete-derivative path prefer it.
        """
        return _shapley_regression_kernel(n_players)

ShapleyValue = SV()

@dataclass(frozen=True, eq=False)
class BV(ExtensionalEquality):
    """The Banzhaf value: uniform-coalition attribution to single players."""

    name: ClassVar[InteractionIndexName] = "BV"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[None] = None

    @property
    def order(self) -> int:
        """Return ``1``: the Banzhaf value attributes to single players only."""
        return 1

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return Banzhaf discrete-derivative weights per outside size."""
        return _banzhaf_derivative_weights(n_players, interaction_size)

    def marginal_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return Banzhaf bloc-marginal weights for single players.

        The Banzhaf value is the singleton restriction of the Banzhaf
        generalized value, mirroring SV's dual capability.
        """
        if interaction_size != 1:
            msg = f"BV attributes to single players only, got interaction_size={interaction_size}"
            raise ValueError(msg)
        return _banzhaf_derivative_weights(n_players, 1)

BanzhafValue = BV()

@dataclass(frozen=True, eq=False)
class WeightedBV(ExtensionalEquality):
    """The weighted Banzhaf value: players join independently with probability ``p``.

    Marginal contributions are weighted with binomial weights
    ``p**t * (1 - p)**(n - 1 - t)`` over outside coalitions of size ``t``.
    The uniform weighting ``p = 1/2`` is the Banzhaf value, and instances at
    ``p = 1/2`` compare equal to ``BV()``. The excluded limits ``p -> 0`` and
    ``p -> 1`` approach the Moebius and Co-Moebius anchors.
    """

    p: float = 0.5

    name: ClassVar[InteractionIndexName] = "WeightedBV"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[None] = None

    def __post_init__(self) -> None:
        """Validate the joining probability."""
        _validate_probability(self.p)

    @property
    def order(self) -> int:
        """Return ``1``: the weighted Banzhaf value attributes to single players only."""
        return 1

    def _identity(self) -> tuple[object, ...]:
        """Collapse the uniform weighting onto the Banzhaf value."""
        if self.p == 0.5:
            return ExtensionalEquality._identity(BV())  # noqa: SLF001 - sibling rule
        return super()._identity()

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return weighted Banzhaf discrete-derivative weights per outside size."""
        return _weighted_banzhaf_derivative_weights(n_players, interaction_size, self.p)


@dataclass(frozen=True, eq=False)
class SII(ExtensionalEquality):
    """The Shapley interaction index up to ``order``; generalizes SV.

    Order is explanation coverage: attributions of shared interactions are
    identical across orders, so a higher order only adds interactions.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "SII"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[SV] = SV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return Shapley discrete-derivative weights per outside size."""
        return _shapley_derivative_weights(n_players, interaction_size)


@dataclass(frozen=True, eq=False)
class BII(ExtensionalEquality):
    """The Banzhaf interaction index up to ``order``; generalizes BV.

    Order is explanation coverage: attributions of shared interactions are
    identical across orders, so a higher order only adds interactions.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "BII"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[BV] = BV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return Banzhaf discrete-derivative weights per outside size."""
        return _banzhaf_derivative_weights(n_players, interaction_size)


@dataclass(frozen=True, eq=False)
class WeightedBII(ExtensionalEquality):
    """The weighted Banzhaf interaction index up to ``order``; generalizes WeightedBV.

    Discrete derivatives are weighted with binomial weights
    ``p**t * (1 - p)**(n - s - t)``: every player outside the interaction
    joins the outside coalition independently with probability ``p``
    (Marichal and Mathonet's weighted Banzhaf index, the cardinal-probabilistic
    family). Order is explanation coverage. The uniform weighting ``p = 1/2``
    is the Banzhaf interaction index, and instances at ``p = 1/2`` compare
    equal to ``BII`` of the same order; the generalized value follows the
    weighting, so order-1 instances equal ``WeightedBV(p)``.
    """

    p: float = 0.5
    order: int = 2

    name: ClassVar[InteractionIndexName] = "WeightedBII"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1

    def __post_init__(self) -> None:
        """Validate the joining probability and the order."""
        _validate_probability(self.p)
        validate_int("order", self.order, minimum=1)

    @property
    def generalizes(self) -> WeightedBV:
        """Return the weighted Banzhaf value with the same weighting."""
        return WeightedBV(p=self.p)

    def _identity(self) -> tuple[object, ...]:
        """Collapse the uniform weighting onto the Banzhaf interaction index."""
        if self.p == 0.5:
            return ExtensionalEquality._identity(  # noqa: SLF001 - sibling rule
                BII(order=self.order),
            )
        return super()._identity()

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return weighted Banzhaf discrete-derivative weights per outside size."""
        return _weighted_banzhaf_derivative_weights(n_players, interaction_size, self.p)


@dataclass(frozen=True, eq=False)
class CHII(ExtensionalEquality):
    """The chaining interaction index up to ``order``; generalizes SV.

    Order is explanation coverage: attributions of shared interactions are
    identical across orders. The chaining weights treat the interaction as
    the head of a chain of arrivals.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "CHII"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[SV] = SV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return chaining discrete-derivative weights per outside size."""
        return _chaining_derivative_weights(n_players, interaction_size)


@dataclass(frozen=True, eq=False)
class STII(ExtensionalEquality):
    """The Shapley-Taylor interaction index of top order ``order``; generalizes SV.

    Order is part of the index identity: attributions of shared interactions
    change with the order. Interactions below the top order are discrete
    derivatives at the empty coalition; the top order distributes the
    remaining game mass along permutations.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "STII"
    order_semantics: ClassVar[OrderSemantics] = "identity"
    preserves_value: ClassVar[bool] = False
    includes_empty_interaction: ClassVar[bool] = False
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


@dataclass(frozen=True, eq=False)
class KSII(ExtensionalEquality):
    """The efficient k-Shapley interaction index of order ``order``; generalizes SV.

    Order is part of the index identity: k-SII aggregates Shapley
    interactions of all orders up to ``order`` into an efficient explanation
    via Bernoulli numbers, so attributions of shared interactions change
    with the order.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "k-SII"
    order_semantics: ClassVar[OrderSemantics] = "identity"
    preserves_value: ClassVar[bool] = False
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[SV] = SV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)


@dataclass(frozen=True, eq=False)
class FSII(ExtensionalEquality):
    """The faithful Shapley interaction index of order ``order``; generalizes SV.

    Order is part of the index identity: the index is the best
    ``order``-additive approximation of the game under the Shapley kernel,
    interpolating the empty and grand coalition exactly.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "FSII"
    order_semantics: ClassVar[OrderSemantics] = "identity"
    preserves_value: ClassVar[bool] = False
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[SV] = SV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def regression_kernel(self, n_players: int) -> Array:
        """Return Shapley kernel weights per coalition size, zero at the ends."""
        return _shapley_regression_kernel(n_players)


@dataclass(frozen=True, eq=False)
class FBII(ExtensionalEquality):
    """The faithful Banzhaf interaction index of order ``order``; generalizes BV.

    Order is part of the index identity: the index is the best
    ``order``-additive approximation of the game under the uniform kernel.
    The fit is unconstrained with a free intercept, so the order-0
    attribution is the fitted intercept of the centered game rather than
    zero; the empty-coalition value itself is the explanation's baseline.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "FBII"
    order_semantics: ClassVar[OrderSemantics] = "identity"
    preserves_value: ClassVar[bool] = False
    includes_empty_interaction: ClassVar[bool] = True
    generalizes: ClassVar[BV] = BV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def regression_kernel(self, n_players: int) -> Array:
        """Return the uniform Banzhaf kernel: unit weight for every size.

        Nonzero weights at the empty and grand coalition mean those rows are
        fitted like any other rather than interpolated as constraints, and
        the fit carries a free intercept as the order-0 attribution.
        """
        return jnp.ones(n_players + 1)


@dataclass(frozen=True, eq=False)
class KADDSHAP(ExtensionalEquality):
    """The k-additive Shapley index of order ``order``; generalizes SV.

    Order is part of the index identity: the index fits a ``order``-additive
    game in the Bernoulli-weighted interaction basis under the Shapley
    kernel, interpolating the grand coalition exactly, so its order-1
    attributions remain Shapley values at every order.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "kADD-SHAP"
    order_semantics: ClassVar[OrderSemantics] = "identity"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[SV] = SV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def regression_kernel(self, n_players: int) -> Array:
        """Return Shapley kernel weights per coalition size, zero at the ends."""
        return _shapley_regression_kernel(n_players)


@dataclass(frozen=True, eq=False)
class Moebius(ExtensionalEquality):
    """The Moebius transform: the game's interaction masses themselves.

    A cardinal index anchored at the empty coalition; the default order
    ``None`` represents every interaction up to the grand coalition. On the
    centered game the empty mass is zero, so the empty-coalition value is
    carried as the explanation's baseline rather than as an attribution.
    """

    order: int | None = None

    name: ClassVar[InteractionIndexName] = "Moebius"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    min_interaction_size: ClassVar[int] = 1
    generalizes: ClassVar[None] = None

    def __post_init__(self) -> None:
        """Validate the order when one is given."""
        if self.order is not None:
            validate_int("order", self.order, minimum=1)

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return discrete-derivative-at-empty weights per outside size."""
        return _empty_anchor_weights(n_players, interaction_size)


@dataclass(frozen=True, eq=False)
class CoMoebius(ExtensionalEquality):
    """The Co-Moebius transform: derivatives anchored at the complement.

    A cardinal index whose derivative of an interaction is taken at the
    coalition of all remaining players; the default order ``None``
    represents every interaction. Its order-0 attribution on the centered
    game is the grand total ``v(N) - v(empty)``.
    """

    order: int | None = None

    name: ClassVar[InteractionIndexName] = "Co-Moebius"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
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


@dataclass(frozen=True, eq=False)
class SGV(ExtensionalEquality):
    """The Shapley generalized value up to ``order``; generalizes SV.

    Attributions weight the marginal contributions of whole interactions
    joining outside coalitions with Shapley weights. Order is explanation
    coverage.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "SGV"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[SV] = SV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def marginal_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return Shapley bloc-marginal weights per outside size."""
        return _shapley_derivative_weights(n_players, interaction_size)


@dataclass(frozen=True, eq=False)
class BGV(ExtensionalEquality):
    """The Banzhaf generalized value up to ``order``; generalizes BV.

    Attributions weight the marginal contributions of whole interactions
    joining outside coalitions uniformly. Order is explanation coverage.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "BGV"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[BV] = BV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def marginal_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return Banzhaf bloc-marginal weights per outside size."""
        return _banzhaf_derivative_weights(n_players, interaction_size)


@dataclass(frozen=True, eq=False)
class CHGV(ExtensionalEquality):
    """The chaining generalized value up to ``order``; generalizes SV.

    Attributions weight the marginal contributions of whole interactions
    joining outside coalitions with chaining weights. Order is explanation
    coverage.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "CHGV"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[SV] = SV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def marginal_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return chaining bloc-marginal weights per outside size."""
        return _chaining_derivative_weights(n_players, interaction_size)


@dataclass(frozen=True, eq=False)
class IGV(ExtensionalEquality):
    """The internal generalized value up to ``order``.

    The attribution of an interaction is its stand-alone worth over the
    empty coalition, ``v(S) - v(empty)``. Order is explanation coverage.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "IGV"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[None] = None

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def marginal_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return bloc-marginal weights anchored at the empty coalition."""
        return _empty_anchor_weights(n_players, interaction_size)


@dataclass(frozen=True, eq=False)
class EGV(ExtensionalEquality):
    """The external generalized value up to ``order``.

    The attribution of an interaction is its contribution on top of all
    remaining players, ``v(N) - v(N without S)``. Order is explanation
    coverage.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "EGV"
    order_semantics: ClassVar[OrderSemantics] = "coverage"
    preserves_value: ClassVar[bool] = True
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[None] = None

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def marginal_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return bloc-marginal weights anchored at the complement."""
        return _grand_anchor_weights(n_players, interaction_size)


@dataclass(frozen=True, eq=False)
class JointSV(ExtensionalEquality):
    """The joint Shapley value of order ``order``; generalizes SV.

    Attributions weight bloc marginal contributions with arrival-process
    weights in which coalitions of up to ``order`` players arrive together,
    so the explanation is efficient. Order is part of the index identity.
    """

    order: int = 2

    name: ClassVar[InteractionIndexName] = "JointSV"
    order_semantics: ClassVar[OrderSemantics] = "identity"
    preserves_value: ClassVar[bool] = False
    includes_empty_interaction: ClassVar[bool] = False
    generalizes: ClassVar[SV] = SV()

    def __post_init__(self) -> None:
        """Validate the order."""
        validate_int("order", self.order, minimum=1)

    def marginal_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return joint-arrival bloc-marginal weights per outside size."""
        return _joint_arrival_weights(n_players, self.order)[: n_players - interaction_size + 1]


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


def _weighted_banzhaf_derivative_weights(n_players: int, size: int, p: float) -> Array:
    """Return weighted Banzhaf derivative weights per outside-coalition size."""
    free = n_players - size
    return jnp.asarray([p**t * (1.0 - p) ** (free - t) for t in range(free + 1)])


def _validate_probability(p: float) -> None:
    """Validate a joining probability strictly inside the unit interval."""
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        msg = f"p must be a float, got {type(p).__name__}"
        raise TypeError(msg)
    if not 0.0 < p < 1.0:
        msg = (
            f"p must satisfy 0 < p < 1, got {p}; the limits are the Moebius (p -> 0) "
            "and Co-Moebius (p -> 1) anchors, and p = 0.5 is the Banzhaf weighting"
        )
        raise ValueError(msg)


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
