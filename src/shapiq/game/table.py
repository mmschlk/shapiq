"""Dense tables of values for every coalition of a finite game."""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Self, override

import jax
import jax.numpy as jnp
from beartype.door import is_bearable
from flextype import (
    CastUnavailable,
    cast as flex_cast,
    caster,
)
from jax.core import Tracer
from jaxtyping import Float, Shaped, TypeCheckError, UInt8
from ndlike.protected_axis.jax import JaxAtMethod, JaxAxisProtected

from shapiq.players import FinitePlayers
from shapiq.typing import typechecked
from shapiq.value._dtype import cast_value

from .discretederivative import subset_patterns
from .finite import FinitePlayerGame, PackedFinitePlayerGame, _FiniteGame

if TYPE_CHECKING:
    from jax.sharding import Sharding
    from jax.typing import DTypeLike
    from numpy.typing import DTypeLike as NpDTypeLike

_TABLE_FALLBACK_COST = 1000.0


@typechecked
@dataclass(slots=True, frozen=True, weakref_slot=True)
class TableGame[P](
    PackedFinitePlayerGame[P, Float[jax.Array, "..."]],
    JaxAxisProtected["TableGame[P]", Any, _FiniteGame[P]],
):
    """A batch of finite games stored as dense coalition-value tables.

    Args:
        values: Floating-point JAX array of shape ``(*batch, 2**n_players)``.
            Entry ``i`` holds the value of the coalition whose membership bits
            form the little-endian binary representation of ``i``. Bit ``j``
            refers to local position ``j`` in the sorted player set, not its
            latent ID. Entry zero is the empty coalition's value.
        players: Finite player sets whose batch shape broadcasts into the
            values batch without enlarging it. Table entries must already
            follow the players' canonical order.

    Indexing and reshaping act on the game batch, preserving the final table
    axis. Boolean and packed uint8 coalitions are accepted; evaluation broadcasts
    their batch with the game batch and returns scalar values. Storage grows
    exponentially with the number of players.

    Packed evaluation is JIT-compiled, fusing table broadcasting and lookup.
    Values and player IDs remain dynamic pytree inputs; shape and dtype checks
    inside the hook run during tracing. Subclasses must preserve the contract
    that ``values`` directly holds the coalition values in canonical order so
    fast table casts can operate on ``values`` without invoking ``eval``.
    """

    values: Float[jax.Array, "*batch 2**{players.n_players}"]
    # A required slot replaces the inherited abstract players property.
    players: Shaped[FinitePlayers[P], "#*batch"] = field()
    protected_axes: ClassVar[dict[str, int]] = {"values": 1, "players": 0}
    broadcasted_batch: ClassVar[set[str]] = {"players"}

    @staticmethod
    def from_game[P2](game: FinitePlayerGame[P2, Any]) -> "TableGame[P2]":
        """Convert ``game`` to a ``TableGame`` via ``flextype.cast``."""
        return flex_cast(game, TableGame)

    @override
    def _prepare_assignment_value(self, value: object, *, method: JaxAtMethod) -> object:
        """Cast compatible finite-game replacements to ``TableGame`` before scattering."""
        if (
            method == "set"
            and isinstance(value, FinitePlayerGame)
            and not isinstance(value, TableGame)
        ):
            return TableGame.from_game(value)
        return value

    @override
    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Return lazy dtype metadata preserving ``weak_type``."""
        return jax.ShapeDtypeStruct((), self.values.dtype, weak_type=self.values.weak_type)

    @property
    @override
    def dtype(self) -> jnp.dtype:
        """Return the stored coalition value dtype."""
        return self.values.dtype

    @override
    def astype(
        self,
        dtype: "DTypeLike | NpDTypeLike | None",
        copy: bool = False,
        device: "jax.Device | Sharding | None" = None,
    ) -> Self:
        """Cast the stored coalition values."""
        game = self if device is None else self.to_device(device)
        target = None if dtype is None else jax.dtypes.canonicalize_dtype(jnp.dtype(dtype))
        values = cast_value(game.values, target, copy=copy, device=None)
        return game if values is game.values else replace(game, values=values)

    @jax.jit
    @typechecked
    @override
    def _eval_packed(
        self,
        coalition: UInt8[jax.Array, "#*batch ({self.n_players}+7)//8"],
    ) -> Float[jax.Array, "*batch"]:
        """Look up packed coalitions, ignoring unused bits in the final byte.

        Raises:
            TypeCheckError: The game and coalition batch shapes are incompatible.
            ValueError: The player count exceeds the enabled index width.
        """
        if not is_bearable(self, Shaped[TableGame[P], "#*batch"]):
            msg = "Incompatible game and coalition batch shapes."
            raise TypeCheckError(msg)
        n_players = self.players.n_players
        index_dtype = jnp.uint64 if jax.config.x64_enabled else jnp.uint32
        if n_players > jnp.iinfo(index_dtype).bits:
            msg = "Table indices exceed the enabled integer width."
            raise ValueError(msg)
        shifts = 8 * jnp.arange(coalition.shape[-1], dtype=index_dtype)
        index = jnp.sum(coalition.astype(index_dtype) << shifts, axis=-1, dtype=index_dtype)
        index &= jnp.array((1 << n_players) - 1, dtype=index_dtype)
        shape = jnp.broadcast_shapes(self.shape, index.shape)
        table = jnp.broadcast_to(self.values, (*shape, self.values.shape[-1]))
        index = jnp.broadcast_to(index, shape)
        return jnp.take_along_axis(table, index[..., None], axis=-1)[..., 0]


# High fallback cost so any cheaper multi-step cast chain is preferred over full 2**n evaluation.
@caster(cost=_TABLE_FALLBACK_COST)
def _finite_game_to_table[P](game: FinitePlayerGame[P, Any]) -> TableGame[P]:
    """Evaluate ``game`` on every coalition to materialize a dense ``TableGame``."""
    spec = game._value_spec()  # noqa: SLF001 - internal lazy abstract dtype hook
    if not spec.weak_type and not jnp.issubdtype(spec.dtype, jnp.floating):
        msg = f"TableGame requires floating jax.Array values, got dtype={spec.dtype}."
        raise CastUnavailable(msg)
    n_players = game.n_players
    # subset_patterns lists coalitions in TableGame order: row i has the bits of i.
    coalitions = subset_patterns(n_players).reshape(
        2**n_players, *(1,) * game.ndim, n_players
    )  # (2**n_players, *ones, n_players)
    raw_values = game(coalitions)
    if not isinstance(raw_values, jax.Array) or not jnp.issubdtype(raw_values.dtype, jnp.floating):
        msg = (
            f"TableGame requires floating jax.Array values, got {type(raw_values).__name__} "
            f"(dtype={getattr(raw_values, 'dtype', None)})."
        )
        raise CastUnavailable(msg)
    values = jnp.moveaxis(raw_values, 0, -1)  # (*batch, 2**n_players)
    players = game.players
    if not isinstance(values, Tracer):
        devices = values.devices()
        if len(devices) == 1 and players.devices() != devices:
            players = players.to_device(next(iter(devices)))
    return TableGame(values, players)
