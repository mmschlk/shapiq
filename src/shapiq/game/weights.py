"""Weight protocols for different types of games."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Protocol, override, runtime_checkable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Float, Int, Shaped
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.typing import typechecked


@runtime_checkable
class Weight[C, T](Protocol):
    """Weight protocol for games."""

    @abstractmethod
    def compute(
        self,
        coalition: Shaped[C, "#*batch"],
        interaction: Shaped[T, "#*batch"],
        n_players: int,
    ) -> Float[jax.Array, "*batch"]:
        """Return the weights for a batch of coalitions and interactions."""
        ...


class DiscreteDerivativeWeight(
    Weight[
        Int[jax.Array, ""],
        Int[jax.Array, ""],
    ],
    JaxAxisProtected[jax.Array],
):
    """Weights based on the discrete derivative representation of interaction indices defined in Grabisch."""


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class DDWeightedBanzhafWeight(DiscreteDerivativeWeight):
    """Weights for the discrete derivative representation of the Banzhaf interaction index."""

    # A float, a traced scalar under jit/vmap, or a p-batch that becomes part of the index batch
    weight: float | Float[jax.Array, "*batch"]
    protected_axes: ClassVar[dict[str, int]] = {"weight": 0}

    def __post_init__(self) -> None:
        """Store the parameter as an array, so it has a batch shape."""
        object.__setattr__(self, "weight", jnp.asarray(self.weight, dtype=float))

    @override
    def compute(
        self,
        coalition: Int[jax.Array, "#*batch"],
        interaction: Int[jax.Array, "#*batch"],
        n_players: int,
    ) -> Float[jax.Array, "*batch"]:
        """Return the weights for a batch of coalitions and interactions."""
        return self.weight ** (coalition) * (1 - self.weight) ** (
            n_players - coalition - interaction
        )


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class DDAlphaBetaWeight(DiscreteDerivativeWeight):
    """Beta(alpha, beta) semivalue weights for the discrete derivative representation.

    The representation is based on Kwon & Zou (2022, Eq. 4) and the interaction representation of Fujimoto et al. (2006, Th. 4.3).
    In their paper they have j = s + 1, where s = |S| is the size of the coalition without the interaction.

    We extend their definition to interactions using the Fujimoto representation, yielding the final weight formula.
    ``B(s + beta, n - k - s + alpha) / B(alpha, beta)`` with s = |S| (coalition without the interaction) and k = |T| (size of the interaction).
    The Shapley interaction index is obtained with alpha = beta = 1.
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
    def compute(
        self,
        coalition: Int[jax.Array, "#*batch"],
        interaction: Int[jax.Array, "#*batch"],
        n_players: int,
    ) -> Float[jax.Array, "*batch"]:
        """Return the weights for a batch of coalitions and interactions."""
        # Ratio of Beta functions in log space, so large alpha/beta do not overflow
        return jnp.exp(
            jsp.special.betaln(
                coalition + self.beta, n_players - coalition - interaction + self.alpha
            )
            - jsp.special.betaln(self.alpha, self.beta)
        )


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class DDShapleyWeight(DDAlphaBetaWeight):
    """Shapley interaction index weights for the discrete derivative representation.

    The Shapley interaction index is obtained with alpha = beta = 1.
    """

    alpha: float | Float[jax.Array, "#*batch"] = 1.0
    beta: float | Float[jax.Array, "#*batch"] = 1.0


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class DDBanzhafWeight(DDWeightedBanzhafWeight):
    """Banzhaf interaction index weights for the discrete derivative representation.

    The Banzhaf interaction index is obtained with weight = 0.5.
    """

    weight: float | Float[jax.Array, "*batch"] = 0.5
