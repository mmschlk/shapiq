"""Interaction Indices as classes, containing weights and the resulting game type.

``DiscreteDerivativeIndex`` and ``LeastSquaresIndex`` are protocols for the two main representations of interaction indices. Algorithms accept an index through the protocol of the
representation they use. Each Index class has a `build` method that packages the computed coefficients into an resulting game type, appropriate for the representation of the coefficients.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, override, runtime_checkable

import jax
from jaxtyping import Bool, Float, Int, Shaped
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.game import Game
from shapiq.game.discretederivative import subset_patterns
from shapiq.game.moebius import AdditiveMoebiusGame, SetMoebiusGame
from shapiq.game.weights import (
    AlphaBetaWeight,
    BanzhafWeight,
    DiscreteDerivativeWeight,
    LeastSquaresWeight,
    ShapleyWeight,
    Weight,
    WeightedBanzhafWeight,
)
from shapiq.players import FinitePlayers
from shapiq.typing import typechecked


@runtime_checkable
class DiscreteDerivativeIndex[R: Game](DiscreteDerivativeWeight, Protocol):
    """An index defined by discrete-derivative weights, e.g. for exact or Monte Carlo routes."""

    def interactions(self, n_players: int) -> Bool[jax.Array, "n_interactions {n_players}"]:
        """Return the interaction sets this index assigns coefficients to."""
        ...

    def build(
        self,
        values: Float[jax.Array, "*batch n_interactions"],
        interactions: Bool[jax.Array, "n_interactions n_players"],
        players: Shaped[FinitePlayers, "#*batch"],
    ) -> R:
        """Package coefficients on ``interactions`` into this index's result game."""
        ...


@runtime_checkable
class LeastSquaresIndex[R: Game](LeastSquaresWeight, Protocol):
    """An index defined as a weighted least-squares fit, e.g. for KernelSHAP-type routes."""

    def interactions(self, n_players: int) -> Bool[jax.Array, "n_interactions {n_players}"]:
        """Return the interaction sets this index assigns coefficients to."""
        ...

    def build(
        self,
        values: Float[jax.Array, "*batch n_interactions"],
        interactions: Bool[jax.Array, "n_interactions n_players"],
        players: Shaped[FinitePlayers, "#*batch"],
    ) -> R:
        """Package coefficients on ``interactions`` into this index's result game."""
        ...


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class Index[W: Weight, R: Game](JaxAxisProtected[jax.Array]):
    """A batch of interaction indices, defined by weights and the interaction orders covered.

    The index batch is the weights' batch. ``build`` decides which game the computed
    coefficients become, so exact and approximate computations share one result type.
    """

    weights: W
    min_order: int
    max_order: int
    # The weights carry their own batch; the index adds no axes.
    protected_axes: ClassVar[dict[str, int]] = {"weights": 0}

    def __post_init__(self) -> None:
        """Check the order range.

        Raises:
            ValueError: The orders are negative or ``min_order`` exceeds ``max_order``.
        """
        if not 0 <= self.min_order <= self.max_order:
            msg = "min_order must be nonnegative and at most max_order."
            raise ValueError(msg)

    def interactions(self, n_players: int) -> Bool[jax.Array, "n_interactions {n_players}"]:
        """Return the interaction sets this index assigns coefficients to."""
        return subset_patterns(order=n_players, min_size=self.min_order, max_size=self.max_order)

    @abstractmethod
    def build(
        self,
        values: Float[jax.Array, "*batch n_interactions"],
        interactions: Bool[jax.Array, "n_interactions n_players"],
        players: Shaped[FinitePlayers, "#*batch"],
    ) -> R:
        """Package coefficients on ``interactions`` into this index's result game."""


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class SemivalueIndex[W: DiscreteDerivativeWeight](
    Index[W, AdditiveMoebiusGame], DiscreteDerivativeIndex[AdditiveMoebiusGame]
):
    """A batch of semivalues: one coefficient per player, packaged as an additive game."""

    weights: W
    # Fixed to singletons, so the result is always additive.
    min_order: int = field(default=1, init=False)
    max_order: int = field(default=1, init=False)

    def discrete_derivative_weight(
        self, size: Int[jax.Array, "#*batch"], order: Int[jax.Array, "#*batch"], n_players: int
    ) -> Float[jax.Array, "*batch"]:
        """Return the discrete-derivative weights that define this semivalue."""
        return self.weights.discrete_derivative_weight(size, order, n_players)

    @override
    def build(
        self,
        values: Float[jax.Array, "*batch n_interactions"],
        interactions: Bool[jax.Array, "n_interactions n_players"],
        players: Shaped[FinitePlayers, "#*batch"],
    ) -> AdditiveMoebiusGame:
        """Package one coefficient per player; ``interactions`` are the singletons by construction."""
        return AdditiveMoebiusGame(values=values, players=players, interactions=interactions)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class CardinalInteractionIndex[W: DiscreteDerivativeWeight](
    Index[W, SetMoebiusGame], DiscreteDerivativeIndex[SetMoebiusGame]
):
    """A batch of cardinal interaction indices, building an k-additive game on the Moebius Basis."""

    weights: W
    min_order: int = 1

    def discrete_derivative_weight(
        self, size: Int[jax.Array, "#*batch"], order: Int[jax.Array, "#*batch"], n_players: int
    ) -> Float[jax.Array, "*batch"]:
        """Return the discrete-derivative weights that define this index."""
        return self.weights.discrete_derivative_weight(size, order, n_players)

    @override
    def build(
        self,
        values: Float[jax.Array, "*batch n_interactions"],
        interactions: Bool[jax.Array, "n_interactions n_players"],
        players: Shaped[FinitePlayers, "#*batch"],
    ) -> SetMoebiusGame:
        """Store the coefficients on their interaction sets, bounded by ``max_order``."""
        return SetMoebiusGame(
            values=values, interactions=interactions, players=players, additivity=self.max_order
        )


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class ShapleyValue(SemivalueIndex[ShapleyWeight], LeastSquaresIndex[AdditiveMoebiusGame]):
    """The Shapley value."""

    weights: ShapleyWeight = field(default_factory=ShapleyWeight)

    def least_squares_weight(
        self, size: Int[jax.Array, "#*batch"], n_players: int
    ) -> Float[jax.Array, "*batch"]:
        """Return the Shapley kernel: the Shapley value is its efficient least-squares fit."""
        return self.weights.least_squares_weight(size, n_players)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class BanzhafValue(SemivalueIndex[BanzhafWeight], LeastSquaresIndex[AdditiveMoebiusGame]):
    """The Banzhaf value."""

    weights: BanzhafWeight = field(default_factory=BanzhafWeight)

    def least_squares_weight(
        self, size: Int[jax.Array, "#*batch"], n_players: int
    ) -> Float[jax.Array, "*batch"]:
        """Return the uniform kernel: the Banzhaf value is its least-squares fit."""
        return self.weights.least_squares_weight(size, n_players)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class WeightedBanzhafValue(
    SemivalueIndex[WeightedBanzhafWeight], LeastSquaresIndex[AdditiveMoebiusGame]
):
    """The weighted Banzhaf value; a batch of ``p`` values becomes the index batch."""

    weights: WeightedBanzhafWeight

    def least_squares_weight(
        self, size: Int[jax.Array, "#*batch"], n_players: int
    ) -> Float[jax.Array, "*batch"]:
        """Return the product kernel: the weighted Banzhaf value is its least-squares fit."""
        return self.weights.least_squares_weight(size, n_players)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class BetaSemivalue(SemivalueIndex[AlphaBetaWeight]):
    """The Beta(alpha, beta) semivalue; a batch of parameters becomes the index batch."""

    weights: AlphaBetaWeight


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class ShapleyInteractionIndex(CardinalInteractionIndex[ShapleyWeight]):
    """The Shapley interaction index (SII)."""

    weights: ShapleyWeight = field(default_factory=ShapleyWeight)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class BanzhafInteractionIndex(CardinalInteractionIndex[BanzhafWeight]):
    """The Banzhaf interaction index (BII)."""

    weights: BanzhafWeight = field(default_factory=BanzhafWeight)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class WeightedBanzhafInteractionIndex(CardinalInteractionIndex[WeightedBanzhafWeight]):
    """The weighted Banzhaf interaction index; a batch of ``p`` values becomes the index batch."""

    weights: WeightedBanzhafWeight


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class BetaInteractionIndex(CardinalInteractionIndex[AlphaBetaWeight]):
    """The Beta(alpha, beta) interaction index; a batch of parameters becomes the index batch."""

    weights: AlphaBetaWeight


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class FaithfulShapleyTopOrderIndex(CardinalInteractionIndex[ShapleyWeight]):
    """Top-order Faithful Shapley interactions (FSII) of order ``k = max_order``.

    Tsai et al. (2023, Theorem 19) give the order-k FSII coefficients as discrete derivatives
    weighted by (2k-1)!/((k-1)!)**2 * (t+k-1)! (n-t-1)! / (n+k-1)! for coalitions of size t.
    Writing the Beta functions as factorials shows this equals the Beta(k, k) member of the
    alpha-beta family, B(t+k, n-t) / B(k, k): a random fill level q ~ Beta(k, k) around 1/2.
    k = 1 recovers the Shapley value; large k approaches the Banzhaf weighting.

    The index carries the Shapley weights of its family; its discrete-derivative weights are
    the Beta(k, k) weights derived from ``max_order``. Only the top-order coefficients are
    computed; as Möbius coefficients they are the top part of the FSII approximation, not the
    whole k-additive fit.
    """

    weights: ShapleyWeight = field(default_factory=ShapleyWeight)
    # Derived from max_order: only the top order is covered.
    min_order: int = field(init=False)

    @override
    def __post_init__(self) -> None:
        """Cover only ``max_order``.

        Raises:
            ValueError: ``max_order`` is negative.
        """
        object.__setattr__(self, "min_order", self.max_order)
        Index.__post_init__(self)

    @override
    def discrete_derivative_weight(
        self, size: Int[jax.Array, "#*batch"], order: Int[jax.Array, "#*batch"], n_players: int
    ) -> Float[jax.Array, "*batch"]:
        """Return the Beta(k, k) weights with ``k = max_order``."""
        k = float(self.max_order)
        return AlphaBetaWeight(alpha=k, beta=k).discrete_derivative_weight(size, order, n_players)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class FaithfulBanzhafTopOrderIndex(CardinalInteractionIndex[BanzhafWeight]):
    """Top-order Faithful Banzhaf interactions (FBII) of order ``max_order``.

    Tsai et al. (2023, Theorem 17) give the top-order FBII coefficients with the Banzhaf
    weights (1/2)**(n-k), so they equal the top-order Banzhaf interactions.

    Only the top-order coefficients are computed; as Möbius coefficients they are the top
    part of the FBII approximation, not the whole k-additive fit.
    """

    weights: BanzhafWeight = field(default_factory=BanzhafWeight)
    # Derived from max_order: only the top order is covered.
    min_order: int = field(init=False)

    @override
    def __post_init__(self) -> None:
        """Cover only ``max_order``.

        Raises:
            ValueError: ``max_order`` is negative.
        """
        object.__setattr__(self, "min_order", self.max_order)
        Index.__post_init__(self)
