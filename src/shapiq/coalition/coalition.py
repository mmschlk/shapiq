"""Data structure for representing a coalition in a game."""

import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, UInt8

from shapiq.typing import typechecked

# These describe individual representations without binding dimension names.
# Operations relate batch shapes and membership/byte widths in their signatures.
BoolCoalition = Bool[Array, "... _"]
PackedBoolCoalition = UInt8[Array, "... _"]
type FiniteCoalition = BoolCoalition | PackedBoolCoalition
type Coalition = FiniteCoalition


@typechecked
def pack_bool_coalition(
    coalition: Bool[Array, "*batch n_players"],
) -> UInt8[Array, "*batch (n_players+7)//8"]:
    """Pack a boolean coalition into the smallest possible unsigned integer type."""
    return jnp.packbits(coalition, axis=-1, bitorder="little")


@typechecked
def unpack_bool_coalition(
    packed: UInt8[Array, "*batch ({n_players}+7)//8"], n_players: int
) -> Bool[Array, "*batch {n_players}"]:
    """Unpack a boolean coalition from the smallest possible unsigned integer type."""
    if not packed.size:
        # JAX unpackbits reshapes with -1, which fails for zero-sized batches.
        return jnp.zeros((*packed.shape[:-1], n_players), dtype=jnp.bool_)
    return jnp.unpackbits(packed, axis=-1, bitorder="little", count=n_players).astype(jnp.bool_)
