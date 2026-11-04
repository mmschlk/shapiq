"""Discrete derivative game for explaining model predictions."""

from dataclasses import dataclass
from typing import ClassVar, override

import jax
import jax.numpy as jnp
import numpy as np
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
    order: int, min_size: int = 0, max_size: int | None = None
) -> Bool[jax.Array, "n_subsets {order}"]:
    """Return the subsets of a set of size ``order`` with ``min_size <= |T| <= max_size``.

    Row ``i`` of the unfiltered powerset is the little-endian bit pattern of ``i``; filtering keeps
    that order. Built with NumPy so the size filter stays static under ``jax.jit``.
    """
    max_size = order if max_size is None else max_size
    patterns = (np.arange(1 << order)[:, None] >> np.arange(order)) & 1  # (2**order, order)
    sizes = patterns.sum(axis=-1)
    return jnp.asarray(patterns[(sizes >= min_size) & (sizes <= max_size)], dtype=bool)


@typechecked
def powerset(
    coalition: Bool[jax.Array, "*batch n_players"],
    order: int,
    min_size: int = 0,
    max_size: int | None = None,
) -> Bool[jax.Array, "n_subsets *batch n_players"]:
    """Generate the subsets of coalitions of size ``order`` with ``min_size <= |T| <= max_size``."""
    patterns = subset_patterns(order, min_size, max_size)  # (n_subsets, order)
    # Get the indices of the players in the coalition
    rank = jnp.cumsum(coalition, axis=-1) - 1  # (*batch n_players)
    # Compute the patterns for the players in the rank
    pattern_columns_of_rank = (
        rank[..., None, :] == jnp.arange(order)[:, None]
    )  # (*batch, order, n_players)
    # Mask out those ranks, which are not in the coalition
    onehot = coalition[..., None, :] & pattern_columns_of_rank  # (*batch, order, n_players)
    # Align patterns for broadcasting
    patterns = patterns.reshape(
        patterns.shape[0], *(1,) * (coalition.ndim - 1), order, 1
    )  # (n_subsets, *batch, order, 1)
    # Match player positions (onehot) with subset patterns (patterns) and build concrete coalition by ORing the player positions with the patterns.
    return jnp.any(onehot & patterns, axis=-2)  # (n_subsets, *batch, n_players)


def broadcast_interaction(
    game: FinitePlayerGame,
    interaction: Bool[jax.Array, "#*batch {game.n_players}"]
    | UInt8[jax.Array, "#*batch ({game.n_players}+7)//8"],
    *batch_shapes: tuple[int, ...],
) -> Bool[jax.Array, "*batch n_players"]:
    """Validate (packed) interactions, unpack and broadcast them with the game and other batches.

    The game and the other batched fields are not broadcast; they stay compact via
    ``broadcasted_batch``.
    """
    if not game.is_valid(interaction):
        msg = "Interaction is inadmissible for this game."
        raise ValueError(msg)
    if interaction.dtype == jnp.uint8:
        interaction = unpack_bool_coalition(interaction, game.n_players)
    shape = jnp.broadcast_shapes(game.shape, *batch_shapes, interaction.shape[:-1])
    return jnp.broadcast_to(interaction, (*shape, interaction.shape[-1]))


def _check_order(interaction: jax.Array, order: int) -> None:
    """Check that every interaction has ``order`` players; skipped inside jit.

    Raises:
        ValueError: An interaction has a different number of players.
    """
    sizes = jnp.sum(interaction, axis=-1)
    if not isinstance(sizes, Tracer) and sizes.size and not jnp.all(sizes == order):
        msg = "Interaction batch must have consistent order."
        raise ValueError(msg)


def _derivative(
    game: FinitePlayerGame,
    interaction: Bool[jax.Array, "*interaction_batch n_players"],
    order: int,
    coalition: Bool[jax.Array, "*coalition_batch n_players"],
) -> tuple[JaxValue, Bool[jax.Array, "*batch n_players"]]:
    """Return Δ_T v(S) = sum over L ⊆ T of (-1)**|T - L| v(S | L), and T broadcast against S."""
    # Broadcast the interaction to match the coalition batch shape
    broadcasted_shape = jnp.broadcast_shapes(coalition.shape[:-1], interaction.shape[:-1])
    interaction = jnp.broadcast_to(
        interaction, (*broadcasted_shape, interaction.shape[-1])
    )  # (*coalition_batch, *interaction_batch, n_players)
    subsets = powerset(interaction, order)  # (2**order, *coalition_batch, *interaction_batch, n)
    game_evals = game.eval(
        coalition[None, :] | subsets
    )  # (2**order, *coalition_batch, *interaction_batch)
    # The signs depend only on the subset patterns, not the specific interaction.
    differences = order - jnp.sum(subset_patterns(order), axis=-1)  # (2**order,)
    signs = ((-1) ** differences).reshape(
        -1, *(1,) * (game_evals.ndim - 1)
    )  # (2**order, *coalition_batch, *interaction_batch)
    derivative = jax_sum(jax_multiply(game_evals, signs), axis=0)  # (*coalition_batch, *i_batch)
    return derivative, interaction


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class DiscreteDerivative[P, V: JaxLike](
    BoolFinitePlayerGame[P, JaxValue], JaxAxisProtected[jax.Array]
):
    """The discrete derivative S ↦ Δ_T v(S) of a game with respect to interactions T.

    Δ_T v(S) = sum over L ⊆ T of (-1)**|T - L| v(S | L). It vanishes when S and T overlap.
    ``order`` is the static size of every interaction T.
    """

    game: Shaped[FinitePlayerGame[P, V], "#*batch"]
    # Accepts packed interactions; __post_init__ stores them unpacked
    interaction: (
        Bool[jax.Array, "#*batch {game.n_players}"]
        | UInt8[jax.Array, "#*batch ({game.n_players}+7)//8"]
    )
    # The interactions carry the full batch shape; the game stays compact and broadcasts into it.
    protected_axes: ClassVar[dict[str, int]] = {"interaction": 1, "game": 0}
    broadcasted_batch: ClassVar[set[str]] = {"game"}
    order: int

    def __post_init__(self) -> None:
        """Validate the interaction and broadcast it to match the game batch."""
        interaction = broadcast_interaction(self.game, self.interaction)
        _check_order(interaction, self.order)
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
        derivative, _ = _derivative(self.game, self.interaction, self.order, coalition)
        return derivative  # (*coalition_batch, *interaction_batch)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class WeightedDiscreteDerivative[P, V: JaxLike](
    BoolFinitePlayerGame[P, JaxValue], JaxAxisProtected[jax.Array]
):
    """The weighted discrete derivative S ↦ w(|S|, |T|, n) [S ∩ T = ∅] Δ_T v(S).

    Summing it over all coalitions S gives the interaction index of ``weights`` at T.
    """

    game: Shaped[FinitePlayerGame[P, V], "#*batch"]
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
        interaction = broadcast_interaction(self.game, self.interaction, self.weights.shape)
        _check_order(interaction, self.order)
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
        derivative, interaction = _derivative(self.game, self.interaction, self.order, coalition)
        disjoint = ~jnp.any(coalition & interaction, axis=-1)  # (*coalition_batch, *i_batch)
        weights = self.weights.compute(
            coalition=jnp.sum(coalition, axis=-1),
            interaction=jnp.sum(interaction, axis=-1),
            n_players=self.game.n_players,
        )  # (*coalition_batch, *interaction_batch)
        return jax_multiply(weights * disjoint, derivative)  # (*coalition_batch, *i_batch)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class WeightedGameValue[P, V: JaxLike](
    BoolFinitePlayerGame[P, JaxValue], JaxAxisProtected[jax.Array]
):
    """The weighted game value S ↦ (-1)**|T - S| w(|S - T|, |T|, n) v(S).

    Summing it over all coalitions S gives the same interaction index as summing the
    ``WeightedDiscreteDerivative``: the representation used by SHAP-IQ (Fumagalli et al., 2023).
    """

    game: Shaped[FinitePlayerGame[P, V], "#*batch"]
    weights: Shaped[DiscreteDerivativeWeight, "#*batch"]
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
        object.__setattr__(
            self,
            "interaction",
            broadcast_interaction(self.game, self.interaction, self.weights.shape),
        )

    @property
    @typechecked
    def players(self) -> FinitePlayers:
        """Return finite player sets whose logical shape broadcasts to ``shape``."""
        return self.game.players

    @override
    def _eval_bool(
        self, coalition: Bool[jax.Array, "#*batch {self.game.n_players}"]
    ) -> Shaped[JaxValue, "*batch"]:
        coalition_sizes = jnp.sum(coalition, axis=-1)  # (*coalition_batch)
        interaction_sizes = jnp.sum(self.interaction, axis=-1)  # (*interaction_batch)
        overlap_sizes = jnp.sum(
            coalition & self.interaction, axis=-1
        )  # (*coalition_batch, *interaction_batch)
        sign = (-1) ** (interaction_sizes - overlap_sizes)
        weights = sign * self.weights.compute(
            coalition_sizes - overlap_sizes, interaction_sizes, self.game.n_players
        )  # (*coalition_batch, *interaction_batch)
        return jax_multiply(
            weights, self.game.eval(coalition)
        )  # (*coalition_batch, *interaction_batch)
