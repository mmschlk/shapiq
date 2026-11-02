"""Dense tables of values for every coalition of a finite game."""

from dataclasses import dataclass, field
from typing import ClassVar, override

import jax
import jax.numpy as jnp
from beartype.door import is_bearable
from jaxtyping import Float, Shaped, TypeCheckError, UInt8
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.players import FinitePlayers
from shapiq.typing import typechecked

from .finite import PackedFinitePlayerGame


@typechecked
@dataclass(slots=True, frozen=True, weakref_slot=True)
class TableGame[P](PackedFinitePlayerGame[P, Float[jax.Array, "..."]], JaxAxisProtected[jax.Array]):
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
    inside the hook run during tracing.
    """

    values: Float[jax.Array, "*batch 2**{players.n_players}"]
    # A required slot replaces the inherited abstract players property.
    players: Shaped[FinitePlayers[P], "#*batch"] = field()
    protected_axes: ClassVar[dict[str, int]] = {"values": 1, "players": 0}
    broadcasted_batch: ClassVar[set[str]] = {"players"}

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
