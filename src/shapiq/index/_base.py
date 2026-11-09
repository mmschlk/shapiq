"""Interaction Indices as classes, containing weights and the resulting game type.

``DiscreteDerivativeIndex`` and ``LeastSquaresIndex`` are protocols for the two main representations of interaction indices. Algorithms accept an index through the protocol of the
representation they use. Each Index class has a `build` method that packages the computed coefficients into an resulting game type, appropriate for the representation of the coefficients.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, override, runtime_checkable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Bool, Float, Int, Shaped, UInt8
from ndlike.jax_like import JaxLike
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.game import Game
from shapiq.game.discretederivative import subset_patterns
from shapiq.game.moebius import AdditiveMoebiusGame, SetMoebiusGame
from shapiq.players import FinitePlayers
from shapiq.typing import typechecked

_BATCH = jnp.empty((0,), dtype=jnp.uint8)


def _log_binom(n: int, k: Int[jax.Array, "*batch"]) -> Float[jax.Array, "*batch"]:
    return (
        jsp.special.gammaln(n + 1.0)
        - jsp.special.gammaln(k + 1.0)
        - jsp.special.gammaln(n - k + 1.0)
    )


@runtime_checkable
class IndexProtocol[R: Game](JaxLike[Any], Protocol):
    """A batch of interaction indices, defined by weights and the interaction orders covered."""

    @property
    def min_order(self) -> int:
        """Return the minimum interaction order this index covers."""

    @property
    def max_order(self) -> int:
        """Return the maximum interaction order this index covers."""

    def interactions(self, n_players: int) -> Bool[jax.Array, "n_interactions {n_players}"]:
        """Return the interaction sets this index assigns coefficients to."""
        return subset_patterns(size=n_players, min_size=self.min_order, max_size=self.max_order)

    @abstractmethod
    def build(
        self,
        values: Float[jax.Array, "*batch n_interactions"],
        interactions: Bool[jax.Array, "n_interactions {players.n_players}"],
        players: Shaped[FinitePlayers, "#*batch"],
    ) -> R:
        """Package coefficients on ``interactions`` into this index's result game."""
        ...


@runtime_checkable
class DiscreteDerivativeIndex[R: Game](IndexProtocol[R], Protocol):
    """An index defined by discrete-derivative weights, e.g. for exact or Monte Carlo routes."""

    @abstractmethod
    def discrete_derivative_weight(
        self, size: Int[jax.Array, "#*batch"], order: Int[jax.Array, "#*batch"], n_players: int
    ) -> Float[jax.Array, "*batch"]:
        """Return ``w(size, order, n_players)`` for coalitions of ``size`` outside the interaction."""


@runtime_checkable
class LeastSquaresIndex[R: Game](IndexProtocol[R], Protocol):
    """An index defined as a weighted least-squares fit."""

    @abstractmethod
    def least_squares_weight(
        self, size: Int[jax.Array, "#*batch"], n_players: int
    ) -> Float[jax.Array, "*batch"]:
        """Return the kernel ``mu(size, n_players)``; ``inf`` marks a hard constraint."""


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class CardinalInteractionIndex(DiscreteDerivativeIndex[SetMoebiusGame]):
    """A batch of cardinal interaction indices, building an k-additive game on the Moebius Basis."""

    min_order: int = field(default=1, init=False)
    # A required field replaces the protocol's max_order property, which would be its default.
    max_order: int = field()

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
@dataclass(frozen=True, slots=True, weakref_slot=True)
class SemivalueIndex(CardinalInteractionIndex):
    """A batch of semivalues: one coefficient per player, packaged as an additive game.

    A semivalue is a cardinal interaction index restricted to the singletons, so both orders are
    fixed to one.
    """

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
        return AdditiveMoebiusGame(values=values, players=players, interactions=interactions)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class WeightedBanzhafIndex(CardinalInteractionIndex, JaxAxisProtected[jax.Array]):
    """The weighted Banzhaf index, with a single parameter p for the Bernoulli distribution.

    The index is the discrete-derivative fit under the weights w(|S|, |T|, n) = p**|S| * (1-p)**(n-|S|-|T|).
    """

    p: float | Float[jax.Array, "#*batch"]
    protected_axes: ClassVar[dict[str, int]] = {"p": 0}
    broadcasted_batch: ClassVar[set[str]] = {"p"}

    def __post_init__(self) -> None:
        """Store ``p`` as an array, so it has a batch shape."""
        object.__setattr__(self, "p", jnp.asarray(self.p, dtype=float))

    @override
    def discrete_derivative_weight(
        self, size: Int[jax.Array, "#*batch"], order: Int[jax.Array, "#*batch"], n_players: int
    ) -> Float[jax.Array, "*batch"]:
        """Return the weighted Banzhaf weights."""
        size = jnp.broadcast_to(size, jnp.broadcast_shapes(self.shape, size.shape))
        return self.p**size * (1 - self.p) ** (n_players - size - order)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class BanzhafIndex(WeightedBanzhafIndex):
    """The Banzhaf index, with p fixed to 1/2."""

    p: float | Float[jax.Array, "#*batch"] = field(default=0.5, init=False)
    # Carry the logical batch shape without storing any elements.
    _batch: UInt8[jax.Array, "*batch 0"] = field(
        default_factory=lambda: _BATCH, kw_only=True, repr=False
    )
    protected_axes: ClassVar[dict[str, int]] = {"_batch": 1}


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class AlphaBetaIndex(CardinalInteractionIndex, JaxAxisProtected[jax.Array]):
    """The alpha-beta index, with two parameters for the Beta distribution.

    The discrete-derivative weights are ``w(|S|, |T|, n) = B(|S| + beta, n - |S| - |T| + alpha)
    / B(alpha, beta)`` (Kwon & Zou, 2022, Eq. 4).
    """

    alpha: float | Float[jax.Array, "#*batch"]
    beta: float | Float[jax.Array, "#*batch"]
    protected_axes: ClassVar[dict[str, int]] = {"alpha": 0, "beta": 0}
    broadcasted_batch: ClassVar[set[str]] = {"alpha", "beta"}

    def __post_init__(self) -> None:
        """Store alpha and beta as arrays without expanding their batches."""
        object.__setattr__(self, "alpha", jnp.asarray(self.alpha, dtype=float))
        object.__setattr__(self, "beta", jnp.asarray(self.beta, dtype=float))

    @override
    def discrete_derivative_weight(
        self, size: Int[jax.Array, "#*batch"], order: Int[jax.Array, "#*batch"], n_players: int
    ) -> Float[jax.Array, "*batch"]:
        """Return the alpha-beta weights."""
        return jnp.exp(
            jsp.special.betaln(size + self.beta, n_players - size - order + self.alpha)
            - jsp.special.betaln(self.alpha, self.beta)
        )


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class ShapleyInteractionIndex(AlphaBetaIndex):
    """The Shapley interaction index, with alpha and beta fixed to one."""

    alpha: float | Float[jax.Array, "#*batch"] = field(default=1.0, init=False)
    beta: float | Float[jax.Array, "#*batch"] = field(default=1.0, init=False)
    # Carry the logical batch shape without storing any elements.
    _batch: UInt8[jax.Array, "*batch 0"] = field(
        default_factory=lambda: _BATCH, kw_only=True, repr=False
    )
    protected_axes: ClassVar[dict[str, int]] = {"_batch": 1}

    @override
    def discrete_derivative_weight(
        self, size: Int[jax.Array, "#*batch"], order: Int[jax.Array, "#*batch"], n_players: int
    ) -> Float[jax.Array, "*batch"]:
        """Return the Shapley weights: the Shapley value is its discrete-derivative fit."""
        size = jnp.broadcast_to(size, jnp.broadcast_shapes(self.shape, size.shape))
        return 1 / ((n_players - order + 1) * jsp.special.comb(n_players - order, size))


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class ShapleyValueIndex(SemivalueIndex, ShapleyInteractionIndex, LeastSquaresIndex[SetMoebiusGame]):
    """The Shapley value: the Shapley interaction index restricted to the singletons."""

    @override
    def least_squares_weight(
        self, size: Int[jax.Array, "#*batch"], n_players: int
    ) -> Float[jax.Array, "*batch"]:
        """Return the Shapley kernel: the Shapley value is its efficient least-squares fit."""
        size = jnp.broadcast_to(size, jnp.broadcast_shapes(self.shape, size.shape))
        interior = (size > 0) & (size < n_players)
        s = jnp.clip(size, 1, max(n_players - 1, 1))
        log_kernel = (
            jnp.log(n_players - 1.0)
            - _log_binom(n_players, s)
            - jnp.log(s)
            - jnp.log(n_players - s)
        )
        return jnp.where(interior, jnp.exp(log_kernel), jnp.inf)
