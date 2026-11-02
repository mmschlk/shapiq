"""Discrete derivative game for explaining model predictions."""
from dataclasses import dataclass
from typing import ClassVar, cast, override

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Shaped
from ndlike.jax_functions import jax_multiply, jax_sum
from ndlike.jax_like import JaxLike
from ndlike.protected_axis.jax import JaxAxisProtected

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
    pattern_columns_of_rank = (rank[..., None, :] == jnp.arange(order))  # (*batch, order, n_players)
    # Mask out those ranks, which are not in the coalition
    onehot = coalition[..., None, :] & pattern_columns_of_rank  # (*batch, order, n_players)
    # Align patterns for broadcasting
    patterns = patterns.reshape(patterns.shape[0],  * (1,) * (coalition.ndim - 1), order , 1) # (2**order, *batch, order, 1)
    # Match player positions (onehot) with subset patterns (patterns) and build concrete coalition by ORing the player positions with the patterns.
    return jnp.any(onehot & patterns, axis=-2)  # (*batch, 2**order, n_players)

def broadcast_game(game: FinitePlayerGame, shape: tuple[int, ...]) -> FinitePlayerGame:
    """Broadcast a game to a new shape, preserving its players and evaluation semantics."""
    if game.shape == shape:
        return game
    game = cast("FinitePlayerGame",game.reshape((1,) * (len(shape) - game.ndim) + game.shape))
    grids = jnp.indices(game.shape, sparse=True) # (*game.shape, 1)
    # We build a fancy indexing. grids[0] is broadcasted to shape indicating the row indices
    # grids[1] is broadcasted to shape indicating the column indices
    return game[tuple(jnp.broadcast_to(grid, shape) for grid in grids)]

@dataclass(frozen=True, slots=True, weakref_slot=True)
class DiscreteDerivativeInteractionIndex[P, V: JaxLike](BoolFinitePlayerGame[P, JaxValue],
                                                        JaxAxisProtected[jax.Array]):
    """Computes the discrete derivate of coalitions given a reference coalition."""
    game: FinitePlayerGame[P, V]
    weights: DiscreteDerivativeWeight
    reference_coalition: Bool[jax.Array, "*interaction_batch {self.game.n_players}"]
    protected_axes: ClassVar[dict[str, int]] = {"game": 0, "reference_coalition": 1}
    order: int

    def __post_init__(self) -> None:
        """Validate the reference coalition and broadcast it to match the game batch shape."""
        # Check if the shapes are broadcastable
        if not self.game.is_valid(self.reference_coalition):
            msg = "Reference coalition is inadmissible for this game."
            raise ValueError(msg)
        # Check if reference coalition has consistent order
        sizes = jnp.sum(self.reference_coalition, axis=-1)
        if sizes.size and not jnp.all(sizes == self.order):
            msg = "Reference coalition batch must have consistent order."
            raise ValueError(msg)
       # Broadcast the reference coalition to match the game batch shape
        shape = jnp.broadcast_shapes(self.game.shape, self.reference_coalition.shape[:-1])
        object.__setattr__(self, "reference_coalition", jnp.broadcast_to(self.reference_coalition, (*shape, self.reference_coalition.shape[-1]))) # Here we do not need an addtional shape, as it does have a protected axis
        object.__setattr__(self, "game", broadcast_game(self.game, shape)) # Here we do not need an addtional shape, as it does not have a protected axis

    @classmethod
    def _pytree_child_names(cls) -> tuple[str, ...]:
        # We declare the game and reference_coalition as dynamic children in pytree
        return ("game","reference_coalition")

    @property
    @typechecked
    def players(self) -> FinitePlayers:
        """Return finite player sets whose logical shape broadcasts to ``shape``."""
        return self.game.players

    @override
    def _eval_bool(
        self, coalition: Bool[jax.Array, "*coalition_batch {self.game.n_players}"]
    ) -> Shaped[JaxValue, "*coalition_batch *interaction_batch"]:
        # Broadcase the reference to match the coalition batch shape
        broadcasted_shape = jnp.broadcast_shapes(coalition.shape[:-1], self.reference_coalition.shape[:-1])
        reference_coalition = jnp.broadcast_to(self.reference_coalition, (*broadcasted_shape, self.reference_coalition.shape[-1])) # (*coalition_batch, *interaction_batch, n_players)
        subsets = powerset(reference_coalition, self.order) # (2**order, *coalition_batch, *interaction_batch, n_players)
        disjoint = ~jnp.any(coalition & reference_coalition, axis=-1) #(*coalition_batch, *interaction_batch)
        weights = self.weights.compute(
            coalition=jnp.sum(coalition, axis=-1),
            interaction=jnp.sum(reference_coalition, axis=-1),
            n_players=self.game.n_players,
        ) # (*coalition_batch, *interaction_batch)
        game_evals = self.game.eval(
            coalition[None,:] | subsets
        ) # (2**order, *coalition_batch, *interaction_batch)
        # Compute the signs for the discrete derivative only depending on the subset_patterns, not the specific reference coalition.
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
class WeightedGameInteractionIndex[P, V: JaxLike](BoolFinitePlayerGame[P, JaxValue]):
    """Computes the weighted evaluation of a game given a reference coalition and weights using the SHAP-IQ representation."""
    weights: DiscreteDerivativeWeight
    game: FinitePlayerGame[P, V]
    interaction: Bool[jax.Array, "... {self.game.n_players}"]

    @override
    def _eval_bool(
        self, coalition: Bool[jax.Array, "*coalition_batch {self.game.n_players}"]
    ) -> Shaped[JaxValue, "*coalition_batch"]:
        coalition_sizes = jax.numpy.sum(coalition, axis=-1)
        interaction_sizes = jax.numpy.sum(self.interaction, axis=-1)
        n_players = self.game.n_players
        overlap_sizes = jax.numpy.sum(coalition & self.interaction, axis=-1)
        sign = (-1) ** (interaction_sizes - overlap_sizes)
        weights = sign * self.weights.compute(
            coalition_sizes - overlap_sizes, interaction_sizes, n_players
        )
        return jax_multiply(weights, self.game.eval(coalition))
