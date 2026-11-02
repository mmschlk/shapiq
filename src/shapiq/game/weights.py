"""Weight protocols for different types of games."""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol, override, runtime_checkable

import jax
import jax.scipy as jsp
from jaxtyping import Float, Int


@runtime_checkable
class Weight[C,T](Protocol):
    """Weight protocol for games."""

    @abstractmethod
    def compute(
        self,
        coalition: C,
        interaction: T,
        n_players: int,
    ) -> Float[jax.Array, "*coalition_batch *interaction_batch"]:
        """Return the weights for a batch of coalitions and interactions."""
        ...

class DiscreteDerivativeWeight(Weight
                               [
                                   Int[jax.Array, "*coalition_batch"],
                                   Int[jax.Array, "*interaction_batch"],
                               ]):
    """Weights based on the discrete derivative representation of interaction indices defined in Grabisch."""



class GameBasedWeight(Weight[
    Int[jax.Array, "*coalition_batch"],
    Int[jax.Array, "*interaction_batch"],
]):
    """Weights based on the maximum-sample-reuse representation of interaction indices introduced in SHAPIQ."""

@dataclass(frozen=True, slots=True, weakref_slot=True)
class DDShapleyWeight(DiscreteDerivativeWeight):
    """Weights for the discrete derivative representation of the Shapley interaction index."""
    @override
    def compute(
        self,
        coalition: Int[jax.Array, "*coalition_batch"],
        interaction: Int[jax.Array, "*interaction_batch"],
        n_players: int,
    ) -> Float[jax.Array, "*coalition_batch *interaction_batch"]:
        """Return the weights for a batch of coalitions and interactions."""
        return 1 / ((n_players - interaction + 1) * jsp.special.comb(n_players - interaction, coalition))

@dataclass(frozen=True, slots=True, weakref_slot=True)
class DDWeightedBanzhafWeight(DiscreteDerivativeWeight):
    """Weights for the discrete derivative representation of the Banzhaf interaction index."""
    weight: Float[jax.Array, "*interaction_batch"]

    @override
    def compute(
        self,
        coalition: Int[jax.Array, "*coalition_batch"],
        interaction: Int[jax.Array, "*interaction_batch"],
        n_players: int,
    ) -> Float[jax.Array, "*coalition_batch *interaction_batch"]:
        """Return the weights for a batch of coalitions and interactions."""
        return self.weight**(coalition) * (1 - self.weight)**(n_players - coalition - interaction)

@dataclass(frozen=True, slots=True, weakref_slot=True)
class DDBanzhafWeight(DiscreteDerivativeWeight):
    """Weights for the discrete derivative representation of the Banzhaf interaction index."""
    @override
    def compute(
        self,
        coalition: Int[jax.Array, "*coalition_batch"],
        interaction: Int[jax.Array, "*interaction_batch"],
        n_players: int,
    ) -> Float[jax.Array, "*coalition_batch *interaction_batch"]:
        """Return the weights for a batch of coalitions and interactions."""
        return 1 / (2**(n_players - interaction))
