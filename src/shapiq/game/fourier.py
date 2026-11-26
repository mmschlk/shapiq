"""K-additive games stored as Fourier coefficients on listed interaction sets."""

from dataclasses import dataclass, field
from typing import Any, ClassVar, cast, override

import jax
import jax.numpy as jnp
from flextype import castmethod
from jaxtyping import Bool, Float, Shaped
from ndlike import jax_functions as jf
from ndlike.jax_like import JaxLike

from shapiq.game.basis import (
    BasisGame,
    _check_table_n_players,
    _require_table_values,
    _scatter_basis_coefficients,
)
from shapiq.game.table import TableGame
from shapiq.typing import typechecked
from shapiq.value.value import JaxValue


def _fast_walsh_hadamard_transform(
    table: Float[jax.Array, "*batch n_coalitions"],
    n_players: int,
) -> Float[jax.Array, "*batch n_coalitions"]:
    """Apply the unnormalized fast Walsh-Hadamard transform along the trailing ``2**n_players`` axis.

    Maps Fourier coefficients ``f(T)`` in little-endian coalition order to coalition values
    ``v(S) = sum_T f(T) * (-1)**|T & S|`` in ``O(n_players * 2**n_players)`` operations.
    """
    batch_shape = table.shape[:-1]
    out = table
    for k in range(n_players):
        out = out.reshape(*batch_shape, 1 << (n_players - 1 - k), 2, 1 << k)
        low = out[..., 0, :]
        high = out[..., 1, :]
        out = jnp.stack((low + high, low - high), axis=-2)
    return out.reshape(*batch_shape, 1 << n_players)


@typechecked
def _fourier_to_moebius_lookup(
    values: Float[JaxValue, "#*batch n_interactions"],
    stored_interactions: Bool[jax.Array, "#*batch n_interactions n_players"],
    query_interactions: Bool[jax.Array, "#*batch n_players"],
    bound: int,
) -> Float[JaxValue, "*batch"]:
    """Compute Möbius coefficients directly from stored Fourier terms via ``m(U) = (-2)**|U| sum_{T >= U} f(T)``."""
    target_batch = jnp.broadcast_shapes(
        values.shape[:-1], stored_interactions.shape[:-2], query_interactions.shape[:-1]
    )
    n_interactions = values.shape[-1]
    sizes = jnp.sum(query_interactions, axis=-1)
    # Clamp sizes above the additivity bound and use ldexp to avoid intermediate 2**|U| overflow.
    in_bound = sizes <= bound
    safe_sizes = jnp.where(in_bound, sizes, 0)
    contains = ~jnp.any(query_interactions[..., None, :] & ~stored_interactions, axis=-1)
    contains = contains & in_bound[..., None]
    contains_b = jnp.broadcast_to(contains, (*target_batch, n_interactions))
    values_b = jf.jax_broadcast_to(values, (*target_batch, n_interactions))
    zero = jf.jax_zeros_like(values, shape=())
    total = jf.jax_sum(jf.jax_where(contains_b, values_b, zero), axis=-1)
    safe_sizes_b = jnp.broadcast_to(safe_sizes, target_batch)
    if isinstance(total, jax.Array):
        scaled = jnp.ldexp(total, safe_sizes_b)
    else:  # Vector values: the exact power of two, which may overflow first.
        scaled = jf.jax_multiply(total, jnp.ldexp(jnp.ones((), values.dtype), safe_sizes_b))
    return jf.jax_where((safe_sizes_b & 1) == 0, scaled, jf.jax_negative(scaled))


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class FourierGame[P, V: JaxLike[Any]](BasisGame[P, V, "FourierGame[P, V]"]):
    """A batch of k-additive games with Fourier coefficients on explicitly stored sets.

    ``values[..., t]`` is the Fourier coefficient f(T_t) of ``interactions[..., t, :]``, and
    v(S) = sum over stored T of f(T) * (-1)**|T & S|. Unlisted sets have coefficient zero;
    the inverse transform is f(T) = 2**-n * sum over all S of v(S) * (-1)**|T & S|.

    Fourier and Möbius degrees coincide, so ``additivity`` bounds both.
    """

    interactions: Bool[jax.Array, "#*batch n_interactions {players.n_players}"] = field()
    additivity: int = field()
    protected_axes: ClassVar[dict[str, int]] = {"interactions": 2}
    broadcasted_batch: ClassVar[set[str]] = {"interactions"}

    @override
    def _basis(
        self, coalition: Bool[jax.Array, "#*batch {self.n_players}"]
    ) -> Shaped[jax.Array, "#*batch n_interactions"]:
        """Return the parity (-1)**|T & S| of every listed interaction T."""
        parity = jnp.sum(self.interactions & coalition[..., None, :], axis=-1)
        return jnp.where((parity & 1) == 0, 1, -1)

    @override
    def empty_value(self) -> Shaped[JaxValue, "..."]:
        """Sum stored Fourier coefficients directly."""
        return cast("Shaped[JaxValue, '...']", jf.jax_sum(self.values, axis=-1))

    @override
    def _coefficients_for_interactions(
        self,
        interactions: Bool[jax.Array, "#*batch {self.n_players}"],
        *,
        max_order: int | None = None,
    ) -> Shaped[JaxValue, "..."]:
        """Compute Möbius coefficients directly from stored Fourier terms."""
        bound = min(self.additivity, self.n_players)
        if max_order is not None:
            bound = min(bound, max_order)
        return cast(
            "Shaped[JaxValue, '...']",
            _fourier_to_moebius_lookup(self.values, self.interactions, interactions, bound),
        )

    @castmethod(TableGame, cost=5.0)
    def _as_table(self) -> TableGame[P]:
        """Materialize a ``TableGame`` via coefficient scatter and the fast Walsh-Hadamard transform."""
        values = _require_table_values(self.values)
        n_players = self.n_players
        _check_table_n_players(n_players)
        n_coalitions = 1 << n_players
        if values.shape[-1] == 0 or 0 in self.shape:
            return TableGame(
                jnp.zeros((*self.shape, n_coalitions), dtype=values.dtype), self.players
            )
        if self.additivity == 0:
            empty = jnp.sum(values, axis=-1, dtype=values.dtype)
            return TableGame(
                jnp.broadcast_to(empty[..., None], (*self.shape, n_coalitions)), self.players
            )
        scattered = _scatter_basis_coefficients(values, self.interactions, n_players)
        table = _fast_walsh_hadamard_transform(scattered, n_players)
        return TableGame(table, self.players)
