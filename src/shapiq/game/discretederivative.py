"""Discrete derivative game for explaining model predictions."""
from dataclasses import dataclass
from typing import ClassVar, override

import jax
import jax.numpy as jnp
from jax.core import Tracer
from jaxtyping import Bool, Shaped, UInt8
from ndlike.jax_functions import jax_multiply, jax_sum
from ndlike.jax_like import JaxLike
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.coalition.coalition import unpack_bool_coalition
from shapiq.game import BoolFinitePlayerGame, FinitePlayerGame
from shapiq.game.weights import DiscreteDerivativeWeight
from shapiq.players import FinitePlayers
from shapiq.typing import typechecked
from shapiq.value.value import JaxValue


@typechecked
def subset_patterns(
    order:int
) -> Bool[jax.Array, "2**{order} {order}"]:
    """Return all boolean patterns of a given order, representing the powerset of a set of that size."""
    return ((jnp.arange(1 << order, dtype=jnp.uint8)[:, None] >> jnp.arange(order)) & 1).astype(bool)

@typechecked
def powerset(coalition: Bool[jax.Array, "*batch n_players"], order: int) -> Bool[jax.Array, "2**{order} *batch n_players"]:
    """Generate the powerset of a coalition."""
    patterns = subset_patterns(order)  # (2**order, order)
    # Get the indices of the players in the coalition
    rank = jnp.cumsum(coalition, axis=-1) - 1  # (*batch n_players)
    # Compute the patterns for the players in the rank
    pattern_columns_of_rank = (rank[..., None, :] == jnp.arange(order)[:, None])  # (*batch, order, n_players)
    # Mask out those ranks, which are not in the coalition
    onehot = coalition[..., None, :] & pattern_columns_of_rank  # (*batch, order, n_players)
    # Align patterns for broadcasting
    patterns = patterns.reshape(patterns.shape[0],  * (1,) * (coalition.ndim - 1), order , 1) # (2**order, *batch, order, 1)
    # Match player positions (onehot) with subset patterns (patterns) and build concrete coalition by ORing the player positions with the patterns.
    return jnp.any(onehot & patterns, axis=-2)  # (*batch, 2**order, n_players)

def broadcast_interaction(
    game: FinitePlayerGame, weights: DiscreteDerivativeWeight, interaction: jax.Array
) -> Bool[jax.Array, "*batch n_players"]:
    """Validate (packed) interactions, unpack and broadcast them with the game and weight batches.

    The game and the weights are not broadcast; they stay compact via ``broadcasted_batch``.
    """
    if not game.is_valid(interaction):
        msg = "Interaction is inadmissible for this game."
        raise ValueError(msg)
    if interaction.dtype == jnp.uint8:
        interaction = unpack_bool_coalition(interaction, game.n_players)
    shape = jnp.broadcast_shapes(game.shape, weights.shape, interaction.shape[:-1])
    return jnp.broadcast_to(interaction, (*shape, interaction.shape[-1]))


@dataclass(frozen=True, slots=True, weakref_slot=True)
class DiscreteDerivativeInteractionIndex[P, V: JaxLike](BoolFinitePlayerGame[P, JaxValue],
                                                        JaxAxisProtected[jax.Array]):
    """Computes the discrete derivate of coalitions given an interaction."""
    game: FinitePlayerGame[P, V]
    weights: Shaped[DiscreteDerivativeWeight, "#*batch"]
    # Accepts packed interactions; __post_init__ stores them unpacked
    interaction: (
        Bool[jax.Array, "#*batch {game.n_players}"]
        | UInt8[jax.Array, "#*batch ({game.n_players}+7)//8"]
    )
    # The interactions carry the full batch shape; game and weights stay compact and broadcast into it.
    protected_axes: ClassVar[dict[str, int]] = {"interaction": 1, "game": 0, "weights": 0}
    broadcasted_batch: ClassVar[set[str]] = {"game", "weights"}
    order: int

    def __post_init__(self) -> None:
        """Validate the interaction and broadcast it to match the game and weight batches."""
        interaction = broadcast_interaction(self.game, self.weights, self.interaction)
        # Check if interaction has consistent order; inside jit the sizes are traced and cannot be checked
        sizes = jnp.sum(interaction, axis=-1)
        if not isinstance(sizes, Tracer) and sizes.size and not jnp.all(sizes == self.order):
            msg = "Interaction batch must have consistent order."
            raise ValueError(msg)
        object.__setattr__(self, "interaction", interaction)

    @property
    @typechecked
    def players(self) -> FinitePlayers:
        """Return finite player sets whose logical shape broadcasts to ``shape``."""
        return self.game.players

    @override
    def _eval_bool(
        self, coalition: Bool[jax.Array, "#*batch {self.game.n_players}"]
    ) -> Shaped[JaxValue, "*batch"]:
        # Broadcast the interaction to match the coalition batch shape
        broadcasted_shape = jnp.broadcast_shapes(coalition.shape[:-1], self.interaction.shape[:-1])
        interaction = jnp.broadcast_to(self.interaction, (*broadcasted_shape, self.interaction.shape[-1])) # (*coalition_batch, *interaction_batch, n_players)
        subsets = powerset(interaction, self.order) # (2**order, *coalition_batch, *interaction_batch, n_players)
        disjoint = ~jnp.any(coalition & interaction, axis=-1) #(*coalition_batch, *interaction_batch)
        weights = self.weights.compute(
            coalition=jnp.sum(coalition, axis=-1),
            interaction=jnp.sum(interaction, axis=-1),
            n_players=self.game.n_players,
        ) # (*coalition_batch, *interaction_batch)
        game_evals = self.game.eval(
            coalition[None,:] | subsets
        ) # (2**order, *coalition_batch, *interaction_batch)
        # Compute the signs for the discrete derivative only depending on the subset_patterns, not the specific interaction.
        differences = self.order - jnp.sum(subset_patterns(self.order), axis=-1) # (2**order,)
        signs = (-1) ** differences # (2**order)
        signs = signs.reshape(-1, * (1,) * (game_evals.ndim - 1)) # (2**order, *coalition_batch, *interaction_batch)
        signed_dd = jax_sum(
            jax_multiply(
                game_evals, signs # (2**order, *coalition_batch, *interaction_batch)
            )
            , axis=0) # (*coalition_batch, *interaction_batch)
        return jax_multiply(weights * disjoint, signed_dd) # (*coalition_batch, *interaction_batch)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class WeightedGameInteractionIndex[P, V: JaxLike](BoolFinitePlayerGame[P, JaxValue],
                                                  JaxAxisProtected[jax.Array]):
    """Computes the weighted evaluation of a game given an interaction and weights using the SHAP-IQ representation."""
    weights: Shaped[DiscreteDerivativeWeight, "#*batch"]
    game: FinitePlayerGame[P, V]
    # Accepts packed interactions; __post_init__ stores them unpacked
    interaction: (
        Bool[jax.Array, "#*batch {game.n_players}"]
        | UInt8[jax.Array, "#*batch ({game.n_players}+7)//8"]
    )
    # The interactions carry the full batch shape; game and weights stay compact and broadcast into it.
    protected_axes: ClassVar[dict[str, int]] = {"interaction": 1, "game": 0, "weights": 0}
    broadcasted_batch: ClassVar[set[str]] = {"game", "weights"}

    def __post_init__(self) -> None:
        """Validate the interactions and broadcast them to match the game and weight batches."""
        object.__setattr__(self, "interaction", broadcast_interaction(self.game, self.weights, self.interaction))

    @property
    @typechecked
    def players(self) -> FinitePlayers:
        """Return finite player sets whose logical shape broadcasts to ``shape``."""
        return self.game.players

    @override
    def _eval_bool(
        self, coalition: Bool[jax.Array, "#*batch {self.game.n_players}"]
    ) -> Shaped[JaxValue, "*batch"]:
        coalition_sizes = jnp.sum(coalition, axis=-1) # (*coalition_batch)
        interaction_sizes = jnp.sum(self.interaction, axis=-1) # (*interaction_batch)
        overlap_sizes = jnp.sum(coalition & self.interaction, axis=-1) # (*coalition_batch, *interaction_batch)
        sign = (-1) ** (interaction_sizes - overlap_sizes)
        weights = sign * self.weights.compute(
            coalition_sizes - overlap_sizes, interaction_sizes, self.game.n_players
        ) # (*coalition_batch, *interaction_batch)
        return jax_multiply(weights, self.game.eval(coalition)) # (*coalition_batch, *interaction_batch)
