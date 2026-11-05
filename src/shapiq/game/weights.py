"""Weight families and the representations they can compute.

A weight class is one family of indices whose parameters form its batch. Each method computes
one representation of the family; the capability protocols name them, so indices and
algorithms can require exactly the representation they use.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, override, runtime_checkable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Float, Int
from ndlike.jax_like import JaxLike
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.typing import typechecked


@runtime_checkable
class Weight(JaxLike[Any], Protocol):
    """A batch of members of one weight family."""


@runtime_checkable
class DiscreteDerivativeWeight(Weight, Protocol):
    """Weights of the discrete-derivative representation.

    ``phi(T) = sum over S ⊆ N - T of w(|S|, |T|, n) Δ_T v(S)``.
    """

    @abstractmethod
    def discrete_derivative_weight(
        self, size: Int[jax.Array, "#*batch"], order: Int[jax.Array, "#*batch"], n_players: int
    ) -> Float[jax.Array, "*batch"]:
        """Return ``w(size, order, n_players)`` for coalitions of ``size`` outside the interaction."""


@runtime_checkable
class LeastSquaresWeight(Weight, Protocol):
    """Kernel of a weighted least-squares representation, up to a constant factor.

    The index is the best fit of the game under the kernel ``mu(|S|, n)``. An infinite entry
    marks coalition sizes the fit must match exactly, i.e. a hard constraint.
    """

    @abstractmethod
    def least_squares_weight(
        self, size: Int[jax.Array, "#*batch"], n_players: int
    ) -> Float[jax.Array, "*batch"]:
        """Return the kernel ``mu(size, n_players)``; ``inf`` marks a hard constraint."""


def _log_binom(n: int, k: Int[jax.Array, "*batch"]) -> Float[jax.Array, "*batch"]:
    return (
        jsp.special.gammaln(n + 1.0)
        - jsp.special.gammaln(k + 1.0)
        - jsp.special.gammaln(n - k + 1.0)
    )


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class AlphaBetaWeight(DiscreteDerivativeWeight, JaxAxisProtected[jax.Array]):
    """Beta(alpha, beta) semivalue weights.

    Kwon & Zou (2022, Eq. 4) define the semivalue weights, which the interaction representation
    of Fujimoto et al. (2006, Th. 4.3) extends to interactions: ``B(s + beta, n - t - s + alpha)
    / B(alpha, beta)`` for coalitions of size s outside an interaction of size t. Their paper uses
    j = s + 1. alpha = beta = 1 gives the Shapley weights.
    """

    alpha: float | Float[jax.Array, "#*batch"]
    beta: float | Float[jax.Array, "#*batch"]
    protected_axes: ClassVar[dict[str, int]] = {"alpha": 0, "beta": 0}

    def __post_init__(self) -> None:
        """Broadcast alpha and beta to a common batch shape, as required for protected fields."""
        alpha, beta = jnp.broadcast_arrays(
            jnp.asarray(self.alpha, dtype=float), jnp.asarray(self.beta, dtype=float)
        )
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)

    @override
    def discrete_derivative_weight(
        self, size: Int[jax.Array, "#*batch"], order: Int[jax.Array, "#*batch"], n_players: int
    ) -> Float[jax.Array, "*batch"]:
        """Return the Beta weights for coalitions of ``size`` outside interactions of ``order``."""
        # Ratio of Beta functions in log space, so large alpha/beta do not overflow
        return jnp.exp(
            jsp.special.betaln(size + self.beta, n_players - size - order + self.alpha)
            - jsp.special.betaln(self.alpha, self.beta)
        )


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class ShapleyWeight(AlphaBetaWeight, LeastSquaresWeight):
    """Shapley weights: Beta(1, 1), and the Shapley kernel as least-squares representation.

    ``alpha`` and ``beta`` stay fields so ndlike can rebuild the weights; they are always one.
    """

    alpha: float | Float[jax.Array, "#*batch"] = 1.0
    beta: float | Float[jax.Array, "#*batch"] = 1.0

    @override
    def least_squares_weight(
        self, size: Int[jax.Array, "#*batch"], n_players: int
    ) -> Float[jax.Array, "*batch"]:
        """Return the Shapley kernel ``(n-1) / (C(n, s) s (n-s))``, infinite at s = 0 and s = n.

        The infinite entries are the efficiency constraint: the fit matches the empty and the
        grand coalition exactly.
        """
        interior = (size > 0) & (size < n_players)
        s = jnp.clip(size, 1, max(n_players - 1, 1))
        log_kernel = (
            jnp.log(n_players - 1.0)
            - _log_binom(n_players, s)
            - jnp.log(s)
            - jnp.log(n_players - s)
        )
        return jnp.where(interior, jnp.exp(log_kernel), jnp.inf)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class WeightedBanzhafWeight(
    DiscreteDerivativeWeight, LeastSquaresWeight, JaxAxisProtected[jax.Array]
):
    """Weighted Banzhaf weights with player inclusion probability ``p``.

    Both representations are product weights: ``p**s (1-p)**(n-t-s)`` for the discrete
    derivative, and ``p**s (1-p)**(n-s)`` as least-squares kernel (Marichal & Mathonet, 2011).
    """

    # A float, a traced scalar under jit/vmap, or a p-batch that becomes part of the index batch
    p: float | Float[jax.Array, "*batch"]
    protected_axes: ClassVar[dict[str, int]] = {"p": 0}

    def __post_init__(self) -> None:
        """Store ``p`` as an array, so it has a batch shape."""
        object.__setattr__(self, "p", jnp.asarray(self.p, dtype=float))

    @override
    def discrete_derivative_weight(
        self, size: Int[jax.Array, "#*batch"], order: Int[jax.Array, "#*batch"], n_players: int
    ) -> Float[jax.Array, "*batch"]:
        """Return ``p**size (1-p)**(n - size - order)``."""
        return self.p**size * (1 - self.p) ** (n_players - size - order)

    @override
    def least_squares_weight(
        self, size: Int[jax.Array, "#*batch"], n_players: int
    ) -> Float[jax.Array, "*batch"]:
        """Return ``p**size (1-p)**(n - size)``; no coalition is constrained."""
        return self.p**size * (1 - self.p) ** (n_players - size)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class BanzhafWeight(WeightedBanzhafWeight):
    """Banzhaf weights: weighted Banzhaf weights with ``p = 1/2``.

    ``p`` stays a field so ndlike can rebuild the weights; it is always one half.
    """

    p: float | Float[jax.Array, "*batch"] = 0.5
