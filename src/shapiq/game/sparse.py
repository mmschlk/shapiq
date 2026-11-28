"""Sparse coefficient-list representations for finite games across Coalition, Möbius, and Fourier bases."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, override

import jax
import jax.numpy as jnp
from flextype import (
    CastUnavailable,
    cast as flex_cast,
    caster,
    castmethod,
)
from jax.core import Tracer
from jaxtyping import Bool, Float, Shaped
from ndlike import jax_functions as jf
from ndlike.jax_like import JaxLike
from ndlike.protected_axis.jax import JaxAtMethod, JaxAxisProtected

from shapiq.players import FinitePlayers
from shapiq.typing import typechecked
from shapiq.value._dtype import cast_value, require_value_spec
from shapiq.value.value import JaxValue

from ._transforms import (
    _additive_to_coalition_values,
    _check_dense_n_players,
    _exact_match_lookup,
    _fourier_to_moebius_lookup,
    _moebius_contributions,
    _moebius_to_fourier_lookup,
    _require_dense_values,
    _scatter_basis_coefficients,
    subset_patterns,
)
from .dense import DenseCoalitionGame, DenseFourierGame, DenseMoebiusGame
from .finite import (
    Basis,
    CoalitionDefinedGame,
    FinitePlayerGame,
    FourierDefinedGame,
    KnownAdditivity,
    MoebiusDefinedGame,
    _FiniteGame,
)

if TYPE_CHECKING:
    from jax.sharding import Sharding
    from jax.typing import DTypeLike
    from numpy.typing import DTypeLike as NpDTypeLike

_SPARSE_DENSE_LAYOUT_COST = 2.0


def _validate_sparse_additivity(
    additivity: object,
    interactions: Bool[jax.Array, "#*batch n_interactions n_players"],
    n_players: int,
) -> int:
    """Validate and normalize a declared static additivity bound for a sparse game.

    Raises:
        TypeError: ``additivity`` is a boolean or not an integer.
        ValueError: The bound is negative or an interaction exceeds it.
    """
    if additivity is None:
        if n_players == 0 or interactions.shape[-2] == 0:
            return 0
        sizes = jnp.sum(interactions, axis=-1)
        if isinstance(sizes, Tracer):
            return n_players
        return int(jnp.max(sizes)) if sizes.size else 0
    if isinstance(additivity, bool):
        msg = "Additivity must be an integer, not bool."
        raise TypeError(msg)
    if not isinstance(additivity, int):
        msg = f"Additivity must be an integer, got {type(additivity).__name__}."
        raise TypeError(msg)
    bound = int(additivity)
    if bound < 0:
        msg = "Additivity must be nonnegative."
        raise ValueError(msg)
    sizes = jnp.sum(interactions, axis=-1)
    # Inside jit the interactions are traced and cannot be checked.
    if not isinstance(sizes, Tracer) and sizes.size and int(jnp.max(sizes)) > bound:
        msg = "An interaction is larger than the declared additivity."
        raise ValueError(msg)
    return 0 if n_players == 0 else bound


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class SparseGame[
    P,
    V: JaxLike[Any],
    OwnerT: "SparseGame[Any, Any, Any]" = "SparseGame[P, V, Any]",
](
    FinitePlayerGame[P, JaxValue],
    JaxAxisProtected[OwnerT, Any, _FiniteGame[P]],
):
    """Abstract base for finite games stored as coefficients on listed interaction sets.

    ``values[..., t]`` is the coefficient of ``interactions[..., t, :]`` in the subclass's
    native basis. Unlisted sets have coefficient zero; duplicate entries in ``interactions``
    contribute additively. ``interactions`` may be shared across the batch or batched along
    axes that broadcast into the game batch without enlarging it.
    """

    values: Float[JaxValue, "*batch n_interactions"]
    # A required slot replaces the inherited abstract players property.
    players: Shaped[FinitePlayers[P], "#*batch"] = field()
    interactions: Bool[jax.Array, "#*batch n_interactions {players.n_players}"] = field()
    protected_axes: ClassVar[dict[str, int]] = {"values": 1, "players": 0, "interactions": 2}
    broadcasted_batch: ClassVar[set[str]] = {"players", "interactions"}

    @classmethod
    def from_game(cls, game: FinitePlayerGame[Any, Any]) -> Self:
        """Convert ``game`` to this sparse game class via ``flextype.cast``."""
        return flex_cast(game, cls)

    @override
    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Return the coefficient dtype, strong after summing basis terms."""
        return require_value_spec(self.values).update(weak_type=False)

    @property
    @override
    def dtype(self) -> jnp.dtype:
        """Return the stored coefficient dtype."""
        return self.values.dtype

    @override
    def astype(
        self,
        dtype: "DTypeLike | NpDTypeLike | None",
        copy: bool = False,
        device: "jax.Device | Sharding | None" = None,
    ) -> Self:
        """Cast the stored coefficients."""
        game = self if device is None else self.to_device(device)
        target = None if dtype is None else jax.dtypes.canonicalize_dtype(jnp.dtype(dtype))
        values = cast_value(game.values, target, copy=copy, device=None)
        return game if values is game.values else replace(game, values=values)

    @override
    def _prepare_assignment_value(self, value: object, *, method: JaxAtMethod) -> object:
        """Require compatible sparse type (and bounded additivity when known) on assignment."""
        del method
        if isinstance(value, SparseGame):
            if not isinstance(value, type(self)):
                msg = f"Cannot assign {type(value).__name__} into {type(self).__name__}."
                raise TypeError(msg)
            if (
                isinstance(self, KnownAdditivity)
                and isinstance(value, KnownAdditivity)
                and value.additivity() > self.additivity()
            ):
                msg = (
                    f"Cannot assign {type(value).__name__} with additivity "
                    f"{value.additivity()} into {type(self).__name__} with additivity "
                    f"{self.additivity()}."
                )
                raise ValueError(msg)
        return value

    @override
    def _zeros_like_value(self, shape: tuple[int, ...]) -> Shaped[JaxValue, "..."]:
        """Return zeros of the coefficients' type without evaluating coalitions."""
        if isinstance(self.values, jax.Array):
            return cast("Shaped[JaxValue, '...']", jnp.zeros(shape, dtype=self.dtype))
        return cast("Shaped[JaxValue, '...']", jf.jax_zeros_like(self.values, shape=shape))


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class SparseCoalitionGame[P, V: JaxLike[Any]](
    SparseGame[P, V, "SparseCoalitionGame[P, V]"],
    CoalitionDefinedGame[P, JaxValue],
):
    """A batch of finite games stored as Coalition-basis values on listed subsets.

    ``values[..., t]`` is the value assigned to ``interactions[..., t, :]``; unlisted
    coalitions evaluate to zero and duplicates sum. Does not implement ``KnownAdditivity``
    because a sparse point-mass coalition function is in general ``n``-additive.
    """

    @override
    def empty_value(self) -> Shaped[JaxValue, "..."]:
        """Read the empty-coalition value directly from storage."""
        is_empty = ~jnp.any(self.interactions, axis=-1)
        is_empty_b = jnp.broadcast_to(is_empty, self.values.shape)
        zero = jf.jax_zeros_like(self.values, shape=())
        return cast(
            "Shaped[JaxValue, '...']",
            jf.jax_sum(jf.jax_where(is_empty_b, self.values, zero), axis=-1),
        )

    @override
    def _coalition_bool(
        self, coalition: Bool[jax.Array, "#*batch {self.n_players}"]
    ) -> Shaped[JaxValue, "*batch"]:
        """Look up coalition values directly from stored sets."""
        return cast(
            "Shaped[JaxValue, '...']",
            _exact_match_lookup(self.values, self.interactions, coalition),
        )

    @castmethod(DenseCoalitionGame, cost=_SPARSE_DENSE_LAYOUT_COST)
    def _as_dense_coalition(self) -> DenseCoalitionGame[P]:
        """Scatter stored coalition values into a ``DenseCoalitionGame``."""
        values = _require_dense_values(self.values, "DenseCoalitionGame")
        table = _scatter_basis_coefficients(values, self.interactions, self.n_players)
        return DenseCoalitionGame(table, self.players)


@typechecked
@dataclass(init=False, frozen=True, slots=True, weakref_slot=True)
class SparseMoebiusGame[P, V: JaxLike[Any]](
    SparseGame[P, V, "SparseMoebiusGame[P, V]"],
    MoebiusDefinedGame[P, JaxValue],
    KnownAdditivity[P, JaxValue],
):
    """A batch of finite games stored as Möbius coefficients on explicitly listed sets.

    ``values[..., t]`` is the Möbius coefficient ``m(T_t)`` of ``interactions[..., t, :]``,
    and ``v(S) = sum_{T <= S} m(T)``. Sets not in ``interactions`` have coefficient zero.
    Implements ``KnownAdditivity`` via ``_additivity``.
    """

    _additivity: int = field()

    @typechecked
    def __init__(
        self,
        values: Float[JaxValue, "*batch n_interactions"],
        players: Shaped[FinitePlayers[P], "#*batch"],
        interactions: Bool[jax.Array, "#*batch n_interactions {players.n_players}"],
        additivity: int | None = None,
        *,
        _additivity: int | None = None,
    ) -> None:
        """Initialize a sparse Möbius game with an optional static additivity bound.

        When ``additivity`` is supplied explicitly, every interaction must have
        cardinality ``<= additivity``. Eager construction validates this; inside
        ``jax.jit`` with traced ``interactions``, callers must uphold this
        invariant (or omit ``additivity`` to default to ``n_players``).
        """
        raw_bound = _additivity if _additivity is not None else additivity
        bound = _validate_sparse_additivity(raw_bound, interactions, players.n_players)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "players", players)
        object.__setattr__(self, "interactions", interactions)
        object.__setattr__(self, "_additivity", bound)

    @classmethod
    @override
    def _combine_sequence_metadata(
        cls,
        items: Sequence["SparseMoebiusGame[Any, Any]"],
        *,
        func: Callable[..., Any] | None,
    ) -> dict[str, Any]:
        """Combine declared additivity bounds order-independently via maximum."""
        del func
        return {"_additivity": max(item._additivity for item in items)}  # noqa: SLF001 - own private metadata field

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
    def contributions(self) -> Shaped[JaxValue, "..."]:
        """Read singleton Möbius coefficients in canonical player order directly from storage."""
        if self.n_players == 0 or self._additivity == 0:
            return self._zeros_like_value((*self.shape, self.n_players))
        return cast(
            "Shaped[JaxValue, '...']",
            _moebius_contributions(self.values, self.interactions, self.n_players),
        )

    @override
    def _coefficients_by_order(self, order: int, basis: Basis) -> Shaped[JaxValue, "..."]:
        """Use direct storage paths for Möbius order-0, order-1, and above-bound queries."""
        if basis is Basis.MOEBIUS:
            n_combos = math.comb(self.n_players, order)
            if order > self._additivity or n_combos == 0:
                return self._zeros_like_value((*self.shape, n_combos))
            if order == 0:
                return cast("Shaped[JaxValue, '...']", jf.jax_expand_dims(self.empty_value(), -1))
            if order == 1:
                return self.contributions()
        return super(SparseMoebiusGame, self)._coefficients_by_order(order, basis)

    @override
    def _coalition_bool(
        self, coalition: Bool[jax.Array, "#*batch {self.n_players}"]
    ) -> Shaped[JaxValue, "*batch"]:
        """Sum stored Möbius coefficients for every listed interaction ``T <= S``."""
        contains = ~jnp.any(self.interactions & ~coalition[..., None, :], axis=-1)
        return cast(
            "Shaped[JaxValue, '...']", jf.jax_sum(jf.jax_multiply(self.values, contains), axis=-1)
        )

    @override
    def _moebius_bool(
        self, interaction: Bool[jax.Array, "#*batch {self.n_players}"]
    ) -> Shaped[JaxValue, "*batch"]:
        """Look up Möbius coefficients directly from stored sets."""
        return cast(
            "Shaped[JaxValue, '...']",
            _exact_match_lookup(self.values, self.interactions, interaction),
        )

    @override
    def _fourier_bool(
        self, interaction: Bool[jax.Array, "#*batch {self.n_players}"]
    ) -> Shaped[JaxValue, "*batch"]:
        """Compute Fourier coefficients directly from stored Möbius terms in O(nnz)."""
        bound = min(self._additivity, self.n_players)
        return cast(
            "Shaped[JaxValue, '...']",
            _moebius_to_fourier_lookup(self.values, self.interactions, interaction, bound),
        )

    @castmethod(DenseMoebiusGame, cost=_SPARSE_DENSE_LAYOUT_COST)
    def _as_dense_moebius(self) -> DenseMoebiusGame[P]:
        """Scatter stored Möbius coefficients into a ``DenseMoebiusGame``."""
        values = _require_dense_values(self.values, "DenseMoebiusGame")
        table = _scatter_basis_coefficients(values, self.interactions, self.n_players)
        return DenseMoebiusGame(table, self.players)

    @castmethod(DenseCoalitionGame, cost=_SPARSE_DENSE_LAYOUT_COST)
    def _as_dense_coalition_additive(self) -> DenseCoalitionGame[P]:
        """Materialize a 0- or 1-additive ``SparseMoebiusGame`` in ``O(2**n_players)``."""
        if self._additivity > 1:
            msg = "O(2**n) additive shortcut requires additivity <= 1."
            raise CastUnavailable(msg)
        values = _require_dense_values(self.values, "DenseCoalitionGame")
        n_players = self.n_players
        index_dtype = _check_dense_n_players(n_players)
        n_coalitions = 1 << n_players
        n_interactions = values.shape[-1]
        if n_interactions == 0 or 0 in self.shape:
            return DenseCoalitionGame(
                jnp.zeros((*self.shape, n_coalitions), dtype=values.dtype), self.players
            )
        if n_players == 0 or self._additivity == 0:
            indices = jnp.zeros(self.interactions.shape[:-1], dtype=index_dtype)
            valid = (
                jnp.sum(self.interactions, axis=-1) == 0
                if n_players > 0
                else jnp.ones(self.interactions.shape[:-1], dtype=bool)
            )
            scatter_values = jnp.where(
                jnp.broadcast_to(valid, values.shape), values, jnp.zeros((), dtype=values.dtype)
            )
            n_slots = 1
        else:
            sizes = jnp.sum(self.interactions, axis=-1)
            is_empty = sizes == 0
            valid = sizes <= 1
            player_idx = jnp.argmax(self.interactions, axis=-1).astype(index_dtype)
            indices = jnp.where(is_empty, jnp.asarray(0, dtype=index_dtype), player_idx + 1)
            scatter_values = jnp.where(
                jnp.broadcast_to(valid, values.shape), values, jnp.zeros((), dtype=values.dtype)
            )
            n_slots = n_players + 1
        if math.prod(self.interactions.shape[:-2]) == 1:
            slots = jnp.zeros((*self.shape, n_slots), dtype=values.dtype)
            slots = slots.at[..., indices.reshape(n_interactions)].add(scatter_values)
        else:
            batch_size = math.prod(self.shape)
            batch_idx = jnp.arange(batch_size)[:, None]
            indices_flat = jnp.broadcast_to(indices, (*self.shape, n_interactions)).reshape(
                batch_size, n_interactions
            )
            values_flat = scatter_values.reshape(batch_size, n_interactions)
            slots_flat = jnp.zeros((batch_size, n_slots), dtype=values.dtype)
            slots = (
                slots_flat.at[batch_idx, indices_flat]
                .add(values_flat)
                .reshape(*self.shape, n_slots)
            )
        empty = slots[..., 0]
        if n_players == 0 or self._additivity == 0:
            return DenseCoalitionGame(
                jnp.broadcast_to(empty[..., None], (*self.shape, n_coalitions)),
                self.players,
            )
        table = _additive_to_coalition_values(empty, slots[..., 1:], n_players)
        return DenseCoalitionGame(table, self.players)


@typechecked
@dataclass(init=False, frozen=True, slots=True, weakref_slot=True)
class SparseFourierGame[P, V: JaxLike[Any]](
    SparseGame[P, V, "SparseFourierGame[P, V]"],
    FourierDefinedGame[P, JaxValue],
    KnownAdditivity[P, JaxValue],
):
    """A batch of finite games stored as Fourier coefficients on explicitly listed sets.

    ``values[..., t]`` is the Fourier coefficient ``f(T_t)`` of ``interactions[..., t, :]``,
    and ``v(S) = sum_T f(T) * (-1)**|T & S|``. Unlisted sets have coefficient zero.
    Fourier and Möbius degrees coincide, so ``_additivity`` bounds both.
    """

    _additivity: int = field()

    @typechecked
    def __init__(
        self,
        values: Float[JaxValue, "*batch n_interactions"],
        players: Shaped[FinitePlayers[P], "#*batch"],
        interactions: Bool[jax.Array, "#*batch n_interactions {players.n_players}"],
        additivity: int | None = None,
        *,
        _additivity: int | None = None,
    ) -> None:
        """Initialize a sparse Fourier game with an optional static additivity bound.

        When ``additivity`` is supplied explicitly, every interaction must have
        cardinality ``<= additivity``. Eager construction validates this; inside
        ``jax.jit`` with traced ``interactions``, callers must uphold this
        invariant (or omit ``additivity`` to default to ``n_players``).
        """
        raw_bound = _additivity if _additivity is not None else additivity
        bound = _validate_sparse_additivity(raw_bound, interactions, players.n_players)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "players", players)
        object.__setattr__(self, "interactions", interactions)
        object.__setattr__(self, "_additivity", bound)

    @classmethod
    @override
    def _combine_sequence_metadata(
        cls,
        items: Sequence["SparseFourierGame[Any, Any]"],
        *,
        func: Callable[..., Any] | None,
    ) -> dict[str, Any]:
        """Combine declared additivity bounds order-independently via maximum."""
        del func
        return {"_additivity": max(item._additivity for item in items)}  # noqa: SLF001 - own private metadata field

    @override
    def empty_value(self) -> Shaped[JaxValue, "..."]:
        """Sum stored Fourier coefficients directly."""
        return cast("Shaped[JaxValue, '...']", jf.jax_sum(self.values, axis=-1))

    @override
    def _coalition_bool(
        self, coalition: Bool[jax.Array, "#*batch {self.n_players}"]
    ) -> Shaped[JaxValue, "*batch"]:
        """Sum stored Fourier coefficients weighted by ``(-1)**|T & S|``."""
        parity = jnp.sum(self.interactions & coalition[..., None, :], axis=-1)
        signs = jnp.where((parity & 1) == 0, 1, -1)
        return cast(
            "Shaped[JaxValue, '...']", jf.jax_sum(jf.jax_multiply(self.values, signs), axis=-1)
        )

    @override
    def _fourier_bool(
        self, interaction: Bool[jax.Array, "#*batch {self.n_players}"]
    ) -> Shaped[JaxValue, "*batch"]:
        """Look up Fourier coefficients directly from stored sets."""
        return cast(
            "Shaped[JaxValue, '...']",
            _exact_match_lookup(self.values, self.interactions, interaction),
        )

    @override
    def _moebius_bool(
        self, interaction: Bool[jax.Array, "#*batch {self.n_players}"]
    ) -> Shaped[JaxValue, "*batch"]:
        """Compute Möbius coefficients directly from stored Fourier terms in O(nnz)."""
        bound = min(self._additivity, self.n_players)
        return cast(
            "Shaped[JaxValue, '...']",
            _fourier_to_moebius_lookup(self.values, self.interactions, interaction, bound),
        )

    @castmethod(DenseFourierGame, cost=_SPARSE_DENSE_LAYOUT_COST)
    def _as_dense_fourier(self) -> DenseFourierGame[P]:
        """Scatter stored Fourier coefficients into a ``DenseFourierGame``."""
        values = _require_dense_values(self.values, "DenseFourierGame")
        table = _scatter_basis_coefficients(values, self.interactions, self.n_players)
        return DenseFourierGame(table, self.players)

    @castmethod(DenseCoalitionGame, cost=_SPARSE_DENSE_LAYOUT_COST)
    def _as_dense_coalition_constant(self) -> DenseCoalitionGame[P]:
        """Materialize a 0-additive ``SparseFourierGame`` by broadcasting its empty sum."""
        if self._additivity > 0:
            msg = "Constant shortcut requires additivity == 0."
            raise CastUnavailable(msg)
        values = _require_dense_values(self.values, "DenseCoalitionGame")
        n_players = self.n_players
        _check_dense_n_players(n_players)
        n_coalitions = 1 << n_players
        if values.shape[-1] == 0 or 0 in self.shape:
            return DenseCoalitionGame(
                jnp.zeros((*self.shape, n_coalitions), dtype=values.dtype), self.players
            )
        empty = jnp.sum(values, axis=-1, dtype=values.dtype)
        return DenseCoalitionGame(
            jnp.broadcast_to(empty[..., None], (*self.shape, n_coalitions)),
            self.players,
        )


# --- Dense -> Sparse layout casts ---


@caster(cost=_SPARSE_DENSE_LAYOUT_COST)
def _dense_coalition_to_sparse[P](game: DenseCoalitionGame[P]) -> SparseCoalitionGame[P, jax.Array]:
    """Convert ``DenseCoalitionGame`` to ``SparseCoalitionGame`` over all subsets."""
    return SparseCoalitionGame(
        values=game.values,
        players=game.players,
        interactions=subset_patterns(game.n_players),
    )


@caster(cost=_SPARSE_DENSE_LAYOUT_COST)
def _dense_moebius_to_sparse[P](game: DenseMoebiusGame[P]) -> SparseMoebiusGame[P, jax.Array]:
    """Convert ``DenseMoebiusGame`` to ``SparseMoebiusGame`` over all subsets."""
    return SparseMoebiusGame(
        values=game.values,
        players=game.players,
        interactions=subset_patterns(game.n_players),
        additivity=game.n_players,
    )


@caster(cost=_SPARSE_DENSE_LAYOUT_COST)
def _dense_fourier_to_sparse[P](game: DenseFourierGame[P]) -> SparseFourierGame[P, jax.Array]:
    """Convert ``DenseFourierGame`` to ``SparseFourierGame`` over all subsets."""
    return SparseFourierGame(
        values=game.values,
        players=game.players,
        interactions=subset_patterns(game.n_players),
        additivity=game.n_players,
    )
