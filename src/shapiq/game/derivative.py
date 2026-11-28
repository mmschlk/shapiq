"""Discrete derivative game for explaining model predictions."""

from dataclasses import dataclass
from typing import Any, ClassVar, override

import jax
import jax.numpy as jnp
from jax.core import Tracer
from jaxtyping import Bool, Shaped, UInt8
from ndlike.jax_functions import jax_multiply, jax_sum
from ndlike.jax_like import JaxLike
from ndlike.protected_axis.jax import JaxAtMethod, JaxAxisProtected

from shapiq.coalition.coalition import unpack_bool_coalition
from shapiq.players import FinitePlayers
from shapiq.typing import typechecked
from shapiq.value._dtype import op_spec
from shapiq.value.value import JaxValue

from ._transforms import powerset, subset_patterns
from .finite import CoalitionDefinedGame, FinitePlayerGame, _FiniteGame

__all__ = ["DiscreteDerivative", "powerset", "subset_patterns"]


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
) -> JaxValue:
    """Return Δ_T v(S) = sum over L ⊆ T of (-1)**|T - L| v(S | L)."""
    broadcasted_shape = jnp.broadcast_shapes(
        game.shape, coalition.shape[:-1], interaction.shape[:-1]
    )
    interaction = jnp.broadcast_to(
        interaction, (*broadcasted_shape, interaction.shape[-1])
    )  # (*coalition_batch, *interaction_batch, n_players)
    subsets = powerset(interaction, order)  # (2**order, *coalition_batch, *interaction_batch, n)
    game_evals = game.eval(
        coalition[None, :] | subsets
    )  # (2**order, *coalition_batch, *interaction_batch)
    differences = order - jnp.sum(subset_patterns(order), axis=-1)  # (2**order,)
    signs = jnp.where((differences & 1) == 0, 1, -1).reshape(
        -1, *(1,) * (game_evals.ndim - 1)
    )  # (2**order, *coalition_batch, *interaction_batch)
    return _signed_sum(game_evals, signs)


def _signed_sum(values: JaxValue, signs: jax.Array) -> JaxValue:
    """Sum the signed values over subsets while retaining narrow array dtypes."""
    signed = jax_multiply(values, signs)
    if isinstance(signed, jax.Array):
        return jnp.sum(signed, axis=0, dtype=signed.dtype)
    return jax_sum(signed, axis=0, dtype=signed.dtype)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class DiscreteDerivative[P, V: JaxLike[Any]](
    CoalitionDefinedGame[P, JaxValue],
    JaxAxisProtected["DiscreteDerivative[P, V]", Any, _FiniteGame[P]],
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
    # Both fields stay compact; their broadcast determines the logical batch shape.
    protected_axes: ClassVar[dict[str, int]] = {"interaction": 1, "game": 0}
    broadcasted_batch: ClassVar[set[str]] = {"interaction", "game"}
    order: int

    def __post_init__(self) -> None:
        """Validate the interaction and unpack it without expanding its batch."""
        interaction = self.interaction
        if not self.game.is_valid(interaction):
            msg = "Interaction is inadmissible for this game."
            raise ValueError(msg)
        if interaction.dtype == jnp.uint8:
            interaction = unpack_bool_coalition(interaction, self.game.n_players)
        _check_order(interaction, self.order)
        object.__setattr__(self, "interaction", interaction)

    @override
    def _prepare_assignment_value(self, value: object, *, method: JaxAtMethod) -> object:
        """Require matching derivative order on indexed replacement."""
        del method
        if isinstance(value, DiscreteDerivative) and value.order != self.order:
            msg = (
                f"Cannot assign DiscreteDerivative with order {value.order} "
                f"into DiscreteDerivative with order {self.order}."
            )
            raise ValueError(msg)
        return value

    @property
    @override
    @typechecked
    def players(self) -> FinitePlayers:
        """Return finite player sets whose logical shape broadcasts to ``shape``."""
        return self.game.players

    @property
    @override
    def dtype(self) -> jnp.dtype:
        """Return the dtype of the signed subset sum."""
        return self._value_spec().dtype

    @override
    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Infer the signed subset sum's spec without evaluating the child game."""
        child = self.game._value_spec()  # noqa: SLF001 - internal lazy abstract dtype hook
        return op_spec(_signed_sum, child.update(shape=(1,)), 1)

    @override
    def _coalition_bool(
        self, coalition: Bool[jax.Array, "#*batch {self.game.n_players}"]
    ) -> Shaped[JaxValue, "*batch"]:
        return _derivative(self.game, self.interaction, self.order, coalition)
