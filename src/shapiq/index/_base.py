"""Interaction indices: weights, the interaction sets they cover, and the game they produce."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, override

import jax
import jax.numpy as jnp
from jax.core import Tracer
from jaxtyping import Bool, Float, Shaped
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.game import Game
from shapiq.game.discretederivative import subset_patterns
from shapiq.game.moebius import AdditiveMoebiusGame, SetMoebiusGame
from shapiq.game.weights import (
    DDAlphaBetaWeight,
    DDBanzhafWeight,
    DDShapleyWeight,
    DDWeightedBanzhafWeight,
    DiscreteDerivativeWeight,
    Weight,
)
from shapiq.players import FinitePlayers
from shapiq.typing import typechecked


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class Index[W: Weight, R: Game](JaxAxisProtected[jax.Array]):
    """A batch of interaction indices, defined by weights and the interaction orders covered.

    The index batch is the weights' batch. ``build`` decides which game the computed
    coefficients become, so exact and approximate computations share one result type.

    Subclasses must re-declare ``weights``: ndlike 0.3.0 checks ``protected_axes`` against a
    class's own annotations only.
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
        return subset_patterns(n_players, self.min_order, self.max_order)

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
class SemivalueIndex[W: DiscreteDerivativeWeight](Index[W, AdditiveMoebiusGame]):
    """A batch of semivalues: one coefficient per player, packaged as an additive game."""

    weights: W
    # Fixed to singletons, so the result is always additive.
    min_order: int = field(default=1, init=False)
    max_order: int = field(default=1, init=False)

    @override
    def build(
        self,
        values: Float[jax.Array, "*batch n_interactions"],
        interactions: Bool[jax.Array, "n_interactions n_players"],
        players: Shaped[FinitePlayers, "#*batch"],
    ) -> AdditiveMoebiusGame:
        """Package one coefficient per player; ``interactions`` are the singletons by construction."""
        del interactions
        return AdditiveMoebiusGame(values=values, players=players)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class CardinalInteractionIndex[W: DiscreteDerivativeWeight](Index[W, SetMoebiusGame]):
    """A batch of cardinal interaction indices, building an k-additive game on the Moebius Basis."""

    weights: W
    min_order: int = 1
    max_order: int

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
class ShapleyValue(SemivalueIndex[DDShapleyWeight]):
    """The Shapley value."""

    weights: DDShapleyWeight = field(default_factory=DDShapleyWeight)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class BanzhafValue(SemivalueIndex[DDBanzhafWeight]):
    """The Banzhaf value."""

    weights: DDBanzhafWeight = field(default_factory=DDBanzhafWeight)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class WeightedBanzhafValue(SemivalueIndex[DDWeightedBanzhafWeight]):
    """The weighted Banzhaf value; a batch of ``p`` values becomes the index batch."""

    weights: DDWeightedBanzhafWeight


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class ShapleyInteractionIndex(CardinalInteractionIndex[DDShapleyWeight]):
    """The Shapley interaction index (SII)."""

    weights: DDShapleyWeight = field(default_factory=DDShapleyWeight)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class BanzhafInteractionIndex(CardinalInteractionIndex[DDBanzhafWeight]):
    """The Banzhaf interaction index (BII)."""

    weights: DDBanzhafWeight = field(default_factory=DDBanzhafWeight)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class WeightedBanzhafInteractionIndex(CardinalInteractionIndex[DDWeightedBanzhafWeight]):
    """The weighted Banzhaf interaction index; a batch of ``p`` values becomes the index batch."""

    weights: DDWeightedBanzhafWeight


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True, init=False)
class FaithfulShapleyTopOrderIndex(CardinalInteractionIndex[DDAlphaBetaWeight]):
    """Top-order Faithful Shapley interactions (FSII) of order ``k = max_order``.

    Tsai et al. (2023, Theorem 19) give the order-k FSII coefficients as discrete derivatives
    weighted by (2k-1)!/((k-1)!)**2 * (t+k-1)! (n-t-1)! / (n+k-1)! for coalitions of size t.
    Writing the Beta functions as factorials shows this equals the Beta(k, k) member of the
    alpha-beta family, B(t+k, n-t) / B(k, k): a random fill level q ~ Beta(k, k) around 1/2.
    k = 1 recovers the Shapley value; large k approaches the Banzhaf weighting.

    Only the top-order coefficients are computed; as Möbius coefficients they are the top
    part of the FSII approximation, not the whole k-additive fit.
    """

    # Re-declared because ndlike 0.3.0 only sees a class's own annotations; see Index.
    weights: DDAlphaBetaWeight
    min_order: int
    max_order: int

    def __init__(
        self,
        *,
        max_order: int,
        min_order: int | None = None,
        weights: DDAlphaBetaWeight | None = None,
    ) -> None:
        """Derive the Beta(k, k) weights from ``max_order``.

        ``replace`` passes every field back in, so ``min_order`` and ``weights`` stay accepted;
        a user passes only ``max_order``.
        """
        if weights is None:
            weights = DDAlphaBetaWeight(alpha=float(max_order), beta=float(max_order))
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "min_order", max_order if min_order is None else min_order)
        object.__setattr__(self, "max_order", max_order)
        self.__post_init__()

    @override
    def __post_init__(self) -> None:
        """Check that only the top order is covered and the weights are Beta(k, k).

        Raises:
            ValueError: ``min_order`` differs from ``max_order``, or alpha/beta differ from it.
        """
        Index.__post_init__(self)
        if self.min_order != self.max_order:
            msg = "Top-order FSII covers only max_order; min_order must equal it."
            raise ValueError(msg)
        alpha, beta = self.weights.alpha, self.weights.beta
        # Inside jit the weights are traced and cannot be checked.
        concrete = not isinstance(alpha, Tracer) and not isinstance(beta, Tracer)
        if concrete and not (jnp.all(alpha == self.max_order) and jnp.all(beta == self.max_order)):
            msg = "Top-order FSII needs Beta(max_order, max_order) weights."
            raise ValueError(msg)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True, kw_only=True)
class FaithfulBanzhafTopOrderIndex(CardinalInteractionIndex[DDBanzhafWeight]):
    """Top-order Faithful Banzhaf interactions (FBII) of order ``max_order``.

    Tsai et al. (2023, Theorem 17) give the top-order FBII coefficients with the Banzhaf
    weights (1/2)**(n-k), so they equal the top-order Banzhaf interactions.

    Only the top-order coefficients are computed; as Möbius coefficients they are the top
    part of the FBII approximation, not the whole k-additive fit.
    """

    weights: DDBanzhafWeight = field(default_factory=DDBanzhafWeight)
    # Derived from max_order: only the top order is covered.
    min_order: int = field(init=False)
    max_order: int

    @override
    def __post_init__(self) -> None:
        """Cover only ``max_order``.

        Raises:
            ValueError: ``max_order`` is negative.
        """
        object.__setattr__(self, "min_order", self.max_order)
        Index.__post_init__(self)
