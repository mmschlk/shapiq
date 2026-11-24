"""K-additive games stored as Fourier coefficients on listed interaction sets."""

from dataclasses import dataclass, field
from typing import Any, ClassVar, cast, override

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Float, Shaped
from ndlike.jax_like import JaxLike

from shapiq.game.basis import BasisGame
from shapiq.typing import typechecked
from shapiq.value.value import JaxValue


@typechecked
def _fourier_to_moebius_lookup(
    values: Float[jax.Array, "#*batch n_interactions"],
    stored_interactions: Bool[jax.Array, "#*batch n_interactions n_players"],
    query_interactions: Bool[jax.Array, "#*batch n_players"],
    bound: int,
) -> Float[jax.Array, "*batch"]:
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
    values_b = jnp.broadcast_to(values, (*target_batch, n_interactions))
    zero = jnp.zeros((), dtype=values.dtype)
    total = jnp.sum(jnp.where(contains_b, values_b, zero), axis=-1)
    safe_sizes_b = jnp.broadcast_to(safe_sizes, target_batch)
    scaled = jnp.ldexp(total, safe_sizes_b)
    return jnp.where((safe_sizes_b & 1) == 0, scaled, -scaled)


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
        return (-1) ** jnp.sum(self.interactions & coalition[..., None, :], axis=-1)

    @override
    def empty_value(self) -> Shaped[JaxValue, "..."]:
        """Sum stored Fourier coefficients directly."""
        return cast("Shaped[JaxValue, '...']", jnp.sum(self.values, axis=-1))

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
