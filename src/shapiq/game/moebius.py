"""K-additive games stored as Möbius coefficients and explicit Möbius inversion helpers."""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, cast, override

import jax
import jax.numpy as jnp
from flextype import CastUnavailable, castmethod
from jax.core import Tracer
from jaxtyping import Bool, Float, Shaped
from ndlike import jax_functions as jf
from ndlike.jax_like import JaxLike

from shapiq.game.basis import BasisGame
from shapiq.game.discretederivative import powerset, subset_patterns
from shapiq.game.kadditive import AdditiveGame, KAdditiveGame
from shapiq.typing import typechecked
from shapiq.value.value import JaxValue


def _moebius_zero_like[P, V: JaxValue](
    game: KAdditiveGame[P, V],
    shape: tuple[int, ...],
) -> Shaped[V, "..."]:
    """Return zero Möbius coefficients matching ``_moebius_coefficients_by_eval`` dtypes."""
    zero = game._zeros_like_value(shape)  # noqa: SLF001 - internal KAdditiveGame hook
    return cast(
        "Shaped[V, '...']",
        jf.jax_sum(jf.jax_multiply(jf.jax_expand_dims(zero, axis=0), jnp.int32(1)), axis=0),
    )


@typechecked
def _moebius_coefficients_by_eval[P, V: JaxValue](
    game: Shaped[KAdditiveGame[P, V], "#*batch"],
    interactions: Bool[jax.Array, "#*batch {game.n_players}"],
    *,
    max_order: int | None = None,
) -> Shaped[V, "*batch"]:
    """Compute Möbius coefficients by subset inversion, zeroing above ``additivity``."""
    n_players = game.n_players
    bound = min(game.additivity, n_players)
    sizes = jnp.sum(interactions, axis=-1)
    target_batch = jnp.broadcast_shapes(game.shape, interactions.shape[:-1])

    if max_order is not None:
        if max_order > bound:
            return _moebius_zero_like(game, target_batch)
        cap = max_order
    elif not isinstance(sizes, Tracer):
        if sizes.size == 0:
            return _moebius_zero_like(game, target_batch)
        max_valid = int(jnp.max(jnp.where(sizes <= bound, sizes, -1)))
        if max_valid < 0:
            return _moebius_zero_like(game, target_batch)
        cap = max_valid
    else:
        cap = bound

    in_bound = sizes <= cap
    clamped = interactions & in_bound[..., None]
    clamped_b = jnp.broadcast_to(clamped, (*target_batch, n_players))
    sizes_b = jnp.broadcast_to(sizes, target_batch)
    in_bound_b = jnp.broadcast_to(in_bound, target_batch)

    patterns = subset_patterns(cap)
    subsets = powerset(clamped_b, cap)
    pattern_sizes = jnp.sum(patterns, axis=-1, dtype=sizes_b.dtype)
    if cap == 0:
        required_sizes = jnp.zeros((1,), dtype=sizes_b.dtype)
    else:
        slot_index = jnp.arange(1, cap + 1, dtype=sizes_b.dtype)
        required_sizes = jnp.max(jnp.where(patterns, slot_index, 0), axis=-1)

    evals = game.eval(subsets)
    lead_shape = (subsets.shape[0], *(1,) * len(target_batch))
    pattern_sizes_b = pattern_sizes.reshape(lead_shape)
    required_sizes_b = required_sizes.reshape(lead_shape)
    valid = in_bound_b[None, ...] & (required_sizes_b <= sizes_b[None, ...])
    parity = ((sizes_b[None, ...] - pattern_sizes_b) & 1) == 0

    signs = jnp.where(parity, 1, -1)
    zero = jf.jax_zeros_like(evals, shape=())
    signed = jf.jax_where(valid, jf.jax_multiply(evals, signs), zero)
    return cast("Shaped[V, '...']", jf.jax_sum(signed, axis=0))


@typechecked
def _moebius_lookup(
    values: Float[JaxValue, "#*batch n_interactions"],
    stored_interactions: Bool[jax.Array, "#*batch n_interactions n_players"],
    query_interactions: Bool[jax.Array, "#*batch n_players"],
) -> Float[JaxValue, "*batch"]:
    """Look up Möbius coefficients directly from stored interaction sets."""
    target_batch = jnp.broadcast_shapes(
        values.shape[:-1], stored_interactions.shape[:-2], query_interactions.shape[:-1]
    )
    n_interactions = values.shape[-1]
    matches = jnp.all(stored_interactions == query_interactions[..., None, :], axis=-1)
    matches_b = jnp.broadcast_to(matches, (*target_batch, n_interactions))
    values_b = jf.jax_broadcast_to(values, (*target_batch, n_interactions))
    zero = jf.jax_zeros_like(values, shape=())
    return jf.jax_sum(jf.jax_where(matches_b, values_b, zero), axis=-1)


@typechecked
def _moebius_contributions(
    values: Float[JaxValue, "*batch n_interactions"],
    stored_interactions: Bool[jax.Array, "#*batch n_interactions {n_players}"],
    n_players: int,
) -> Float[JaxValue, "*batch {n_players}"]:
    """Extract singleton Möbius coefficients in canonical player order directly from storage."""
    game_shape = values.shape[:-1]
    if n_players == 0:
        return jf.jax_zeros_like(values, shape=(*game_shape, 0))
    n_interactions = values.shape[-1]
    is_singleton = jnp.sum(stored_interactions, axis=-1) == 1
    mask = stored_interactions & is_singleton[..., None]
    mask_b = jnp.broadcast_to(mask, (*game_shape, n_interactions, n_players))
    values_b = jf.jax_expand_dims(jf.jax_broadcast_to(values, (*game_shape, n_interactions)), -1)
    zero = jf.jax_zeros_like(values, shape=())
    return jf.jax_sum(jf.jax_where(mask_b, values_b, zero), axis=-2)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class MoebiusGame[
    P,
    V: JaxLike[Any],
    OwnerT: "MoebiusGame[Any, Any, Any]" = "MoebiusGame[P, V, Any]",
](BasisGame[P, V, OwnerT]):
    """A batch of k-additive games with Möbius coefficients on explicitly stored sets.

    ``values[..., t]`` is the Möbius coefficient m(T_t) of ``interactions[..., t, :]``, and
    v(S) = sum over T ⊆ S of m(T). Sets not in ``interactions`` have coefficient zero.
    """

    interactions: Bool[jax.Array, "#*batch n_interactions {players.n_players}"] = field()
    additivity: int = field()
    protected_axes: ClassVar[dict[str, int]] = {"interactions": 2}
    broadcasted_batch: ClassVar[set[str]] = {"interactions"}

    @override
    def _basis(
        self, coalition: Bool[jax.Array, "#*batch {self.n_players}"]
    ) -> Shaped[jax.Array, "#*batch n_interactions"]:
        """Return whether every interaction T is contained in S (T ⊆ S)."""
        return ~jnp.any(self.interactions & ~coalition[..., None, :], axis=-1)

    @override
    def empty_value(self) -> Shaped[JaxValue, "..."]:
        """Read the empty-coalition Möbius coefficient directly from storage."""
        is_empty = ~jnp.any(self.interactions, axis=-1)
        is_empty_b = jnp.broadcast_to(is_empty, self.values.shape)
        zero = jf.jax_zeros_like(self.values, shape=())
        return cast(
            "Shaped[JaxValue, '...']",
            jf.jax_sum(jf.jax_where(is_empty_b, self.values, zero), axis=-1),
        )

    @override
    def _coefficients_for_interactions(
        self,
        interactions: Bool[jax.Array, "#*batch {self.n_players}"],
        *,
        max_order: int | None = None,
    ) -> Shaped[JaxValue, "*batch"]:
        """Look up Möbius coefficients directly from stored sets."""
        del max_order
        return cast(
            "Shaped[JaxValue, '...']",
            _moebius_lookup(self.values, self.interactions, interactions),
        )

    @castmethod(AdditiveGame)
    def _as_additive(self) -> AdditiveGame[P, JaxValue]:
        """View a ``MoebiusGame`` with ``additivity <= 1`` as an ``AdditiveGame``."""
        additivity = self.additivity
        if additivity > 1:
            msg = f"MoebiusGame with additivity {additivity} is not additive."
            raise CastUnavailable(msg)
        return AdditiveMoebiusGame(
            values=self.values,
            players=self.players,
            interactions=self.interactions,
            additivity=additivity,  # ty: ignore[invalid-argument-type]
        )


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class AdditiveMoebiusGame[P, V: JaxLike[Any]](
    MoebiusGame[P, V, "AdditiveMoebiusGame[P, V]"],
    AdditiveGame[P, JaxValue],
):
    """A batch of 1-additive (or 0-additive) games stored as Möbius coefficients on sets.

    Interactions may include the empty coalition (giving an arbitrary empty-coalition value)
    and any subset of singletons in any order. Sets not in ``interactions`` have coefficient
    zero.
    """

    additivity: Literal[0, 1] = field(default=1)

    def __post_init__(self) -> None:
        """Check that ``additivity`` is at most one (zero for zero-player games) and bounds interactions.

        Raises:
            TypeError: ``additivity`` is a boolean rather than an integer.
            ValueError: ``additivity`` is negative, exceeds one, or an interaction is larger
                than ``additivity``.
        """
        if isinstance(self.additivity, bool):
            msg = "Additivity must be an integer, not bool."
            raise TypeError(msg)
        if self.additivity > 1:
            msg = f"AdditiveMoebiusGame additivity must be at most 1, got {self.additivity}."
            raise ValueError(msg)
        super(AdditiveMoebiusGame, self).__post_init__()
        normalized = 0 if self.n_players == 0 or self.additivity == 0 else 1
        object.__setattr__(self, "additivity", normalized)

    @override
    def contributions(self) -> Shaped[JaxValue, "..."]:
        """Read singleton Möbius coefficients in canonical player order directly from storage."""
        return cast(
            "Shaped[JaxValue, '...']",
            _moebius_contributions(self.values, self.interactions, self.n_players),
        )
