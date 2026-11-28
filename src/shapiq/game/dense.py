"""Dense 2**n-entry representations for finite games across Coalition, Möbius, and Fourier bases."""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Self, override

import jax
import jax.numpy as jnp
import numpy as np
from beartype.door import is_bearable
from flextype import (
    CastUnavailable,
    cast as flex_cast,
    caster,
)
from jax.core import Tracer
from jaxtyping import Bool, Float, Shaped, TypeCheckError, UInt8
from ndlike.protected_axis.jax import JaxAtMethod, JaxAxisProtected

from shapiq.coalition.coalition import pack_bool_coalition
from shapiq.players import FinitePlayers
from shapiq.typing import typechecked
from shapiq.value._dtype import cast_value

from ._transforms import (
    _fast_fourier_to_moebius_transform,
    _fast_moebius_to_fourier_transform,
    _fast_moebius_transform,
    _fast_walsh_hadamard_transform,
    _fast_zeta_transform,
    subset_patterns,
)
from .finite import (
    Basis,
    CoalitionDefinedGame,
    FinitePlayerGame,
    FourierDefinedGame,
    MoebiusDefinedGame,
    _FiniteGame,
)

if TYPE_CHECKING:
    from jax.sharding import Sharding
    from jax.typing import DTypeLike
    from numpy.typing import DTypeLike as NpDTypeLike

_DENSE_TRANSFORM_COST = 3.0
_DENSE_NATIVE_FALLBACK_COST = 1000.0
_DENSE_FALLBACK_COST = 1010.0


@typechecked
@dataclass(slots=True, frozen=True, weakref_slot=True)
class DenseGame[
    P,
    OwnerT: "DenseGame[Any, Any]" = "DenseGame[P, Any]",
](
    FinitePlayerGame[P, Float[jax.Array, "..."]],
    JaxAxisProtected[OwnerT, Any, _FiniteGame[P]],
):
    """Abstract base for finite games stored as dense ``(*batch, 2**n_players)`` tables.

    Entry ``i`` holds the coefficient in the subclass's native basis for the subset whose
    membership bits form the little-endian binary representation of ``i``. Bit ``j``
    refers to local position ``j`` in the sorted player set, not its latent ID.

    Subclasses bind ``_lookup_packed`` / ``_lookup_bool`` to their native basis hook.
    """

    values: Float[jax.Array, "*batch 2**{players.n_players}"]
    # A required slot replaces the inherited abstract players property.
    players: Shaped[FinitePlayers[P], "#*batch"] = field()
    protected_axes: ClassVar[dict[str, int]] = {"values": 1, "players": 0}
    broadcasted_batch: ClassVar[set[str]] = {"players"}

    @classmethod
    def from_game(cls, game: FinitePlayerGame[Any, Any]) -> Self:
        """Convert ``game`` to this dense game class via ``flextype.cast``."""
        return flex_cast(game, cls)

    @override
    def _prepare_assignment_value(self, value: object, *, method: JaxAtMethod) -> object:
        """Cast compatible finite-game replacements to ``type(self)`` before scattering."""
        if (
            method == "set"
            and isinstance(value, FinitePlayerGame)
            and not isinstance(value, type(self))
        ):
            return type(self).from_game(value)
        return value

    @override
    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Return lazy dtype metadata preserving ``weak_type``."""
        return jax.ShapeDtypeStruct((), self.values.dtype, weak_type=self.values.weak_type)

    @property
    @override
    def dtype(self) -> jnp.dtype:
        """Return the stored table dtype."""
        return self.values.dtype

    @override
    def _zeros_like_value(self, shape: tuple[int, ...]) -> Float[jax.Array, "..."]:
        """Return zeros of the stored dtype without evaluating coalitions."""
        return jnp.zeros(shape, dtype=self.values.dtype)

    @override
    def astype(
        self,
        dtype: "DTypeLike | NpDTypeLike | None",
        copy: bool = False,
        device: "jax.Device | Sharding | None" = None,
    ) -> Self:
        """Cast the stored table entries."""
        game = self if device is None else self.to_device(device)
        target = None if dtype is None else jax.dtypes.canonicalize_dtype(jnp.dtype(dtype))
        values = cast_value(game.values, target, copy=copy, device=None)
        return game if values is game.values else replace(game, values=values)

    @jax.jit
    @typechecked
    def _lookup_packed(
        self,
        coalition: UInt8[jax.Array, "#*batch ({self.n_players}+7)//8"],
    ) -> Float[jax.Array, "*batch"]:
        """Look up packed subsets in the dense table, ignoring unused padding bits.

        Raises:
            TypeCheckError: The game and coalition batch shapes are incompatible.
            ValueError: The player count exceeds the enabled index width.
        """
        if not is_bearable(self, Shaped[DenseGame[P, Any], "#*batch"]):
            msg = "Incompatible game and coalition batch shapes."
            raise TypeCheckError(msg)
        n_players = self.players.n_players
        index_dtype = jnp.uint64 if jax.config.x64_enabled else jnp.uint32
        if n_players > jnp.iinfo(index_dtype).bits:
            msg = "Dense game indices exceed the enabled integer width."
            raise ValueError(msg)
        shifts = 8 * jnp.arange(coalition.shape[-1], dtype=index_dtype)
        index = jnp.sum(coalition.astype(index_dtype) << shifts, axis=-1, dtype=index_dtype)
        index &= jnp.array((1 << n_players) - 1, dtype=index_dtype)
        shape = jnp.broadcast_shapes(self.shape, index.shape)
        table = jnp.broadcast_to(self.values, (*shape, self.values.shape[-1]))
        index = jnp.broadcast_to(index, shape)
        return jnp.take_along_axis(table, index[..., None], axis=-1)[..., 0]

    def _lookup_bool(
        self,
        coalition: Bool[jax.Array, "... {self.n_players}"],
    ) -> Float[jax.Array, "..."]:
        """Pack boolean subsets and delegate to the JIT-compiled table lookup."""
        return self._lookup_packed(pack_bool_coalition(coalition))


@typechecked
@dataclass(slots=True, frozen=True, weakref_slot=True)
class DenseCoalitionGame[P](
    DenseGame[P, "DenseCoalitionGame[P]"],
    CoalitionDefinedGame[P, Float[jax.Array, "..."]],
):
    """A batch of finite games stored as dense coalition-value tables.

    Entry ``i`` holds ``v(S_i)`` where ``S_i`` has little-endian membership bits ``i``.
    Entry zero is the empty coalition's value. Subclasses must preserve the contract
    that ``values`` directly holds the coalition values in canonical order so fast
    table casts can operate on ``values`` without invoking ``eval``.
    """

    @override
    def empty_value(self) -> Float[jax.Array, "..."]:
        """Read the empty-coalition value directly from index zero."""
        return self.values[..., 0]

    @override
    def _coalition_packed(
        self, coalition: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Float[jax.Array, "..."]:
        """Look up packed coalitions directly in the dense coalition table."""
        return self._lookup_packed(coalition)

    @override
    def _coalition_bool(
        self, coalition: Bool[jax.Array, "... {self.n_players}"]
    ) -> Float[jax.Array, "..."]:
        """Look up boolean coalitions via the JIT-compiled packed table lookup."""
        return self._lookup_bool(coalition)


@typechecked
@dataclass(slots=True, frozen=True, weakref_slot=True)
class DenseMoebiusGame[P](
    DenseGame[P, "DenseMoebiusGame[P]"],
    MoebiusDefinedGame[P, Float[jax.Array, "..."]],
):
    """A batch of finite games stored as dense Möbius-coefficient tables.

    Entry ``i`` holds ``m(T_i)`` where ``T_i`` has little-endian membership bits ``i``.
    """

    @override
    def empty_value(self) -> Float[jax.Array, "..."]:
        """Read ``v(empty) = m(empty)`` directly from index zero."""
        return self.values[..., 0]

    @override
    def contributions(self) -> Float[jax.Array, "..."]:
        """Read singleton Möbius coefficients directly from power-of-two indices."""
        if self.n_players == 0:
            return jnp.zeros((*self.shape, 0), dtype=self.values.dtype)
        idx = jnp.asarray(1 << np.arange(self.n_players))
        return self.values[..., idx]

    @override
    def _coefficients_by_order(self, order: int, basis: Basis) -> Float[jax.Array, "..."]:
        """Use direct O(1)/O(n) slices for order-0 and order-1 Möbius queries."""
        if basis is Basis.MOEBIUS:
            if order == 0:
                return self.values[..., :1]
            if order == 1:
                return self.contributions()
        return super(DenseMoebiusGame, self)._coefficients_by_order(order, basis)

    @override
    def _moebius_packed(
        self, interaction: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Float[jax.Array, "..."]:
        """Look up packed interactions directly in the dense Möbius table."""
        return self._lookup_packed(interaction)

    @override
    def _moebius_bool(
        self, interaction: Bool[jax.Array, "... {self.n_players}"]
    ) -> Float[jax.Array, "..."]:
        """Look up boolean interactions via the JIT-compiled packed table lookup."""
        return self._lookup_bool(interaction)


@typechecked
@dataclass(slots=True, frozen=True, weakref_slot=True)
class DenseFourierGame[P](
    DenseGame[P, "DenseFourierGame[P]"],
    FourierDefinedGame[P, Float[jax.Array, "..."]],
):
    """A batch of finite games stored as dense Fourier-coefficient tables.

    Entry ``i`` holds ``f(T_i)`` where ``T_i`` has little-endian membership bits ``i``.
    """

    @override
    def empty_value(self) -> Float[jax.Array, "..."]:
        """Compute ``v(empty) = sum_T f(T)`` directly across the stored table."""
        return jnp.sum(self.values, axis=-1, dtype=self.values.dtype)

    @override
    def _fourier_packed(
        self, interaction: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Float[jax.Array, "..."]:
        """Look up packed interactions directly in the dense Fourier table."""
        return self._lookup_packed(interaction)

    @override
    def _fourier_bool(
        self, interaction: Bool[jax.Array, "... {self.n_players}"]
    ) -> Float[jax.Array, "..."]:
        """Look up boolean interactions via the JIT-compiled packed table lookup."""
        return self._lookup_bool(interaction)


# --- Pairwise O(n * 2**n) dense basis transforms ---


@caster(cost=_DENSE_TRANSFORM_COST)
def _dense_coalition_to_moebius[P](game: DenseCoalitionGame[P]) -> DenseMoebiusGame[P]:
    """Convert ``DenseCoalitionGame`` to ``DenseMoebiusGame`` via fast Möbius transform."""
    return DenseMoebiusGame(_fast_moebius_transform(game.values, game.n_players), game.players)


@caster(cost=_DENSE_TRANSFORM_COST)
def _dense_moebius_to_coalition[P](game: DenseMoebiusGame[P]) -> DenseCoalitionGame[P]:
    """Convert ``DenseMoebiusGame`` to ``DenseCoalitionGame`` via fast zeta transform."""
    return DenseCoalitionGame(_fast_zeta_transform(game.values, game.n_players), game.players)


@caster(cost=_DENSE_TRANSFORM_COST)
def _dense_coalition_to_fourier[P](game: DenseCoalitionGame[P]) -> DenseFourierGame[P]:
    """Convert ``DenseCoalitionGame`` to ``DenseFourierGame`` via normalized WHT."""
    return DenseFourierGame(
        _fast_walsh_hadamard_transform(game.values, game.n_players, normalize=True),
        game.players,
    )


@caster(cost=_DENSE_TRANSFORM_COST)
def _dense_fourier_to_coalition[P](game: DenseFourierGame[P]) -> DenseCoalitionGame[P]:
    """Convert ``DenseFourierGame`` to ``DenseCoalitionGame`` via unnormalized WHT."""
    return DenseCoalitionGame(
        _fast_walsh_hadamard_transform(game.values, game.n_players, normalize=False),
        game.players,
    )


@caster(cost=_DENSE_TRANSFORM_COST)
def _dense_moebius_to_fourier[P](game: DenseMoebiusGame[P]) -> DenseFourierGame[P]:
    """Convert ``DenseMoebiusGame`` to ``DenseFourierGame`` via direct Kronecker transform."""
    return DenseFourierGame(
        _fast_moebius_to_fourier_transform(game.values, game.n_players),
        game.players,
    )


@caster(cost=_DENSE_TRANSFORM_COST)
def _dense_fourier_to_moebius[P](game: DenseFourierGame[P]) -> DenseMoebiusGame[P]:
    """Convert ``DenseFourierGame`` to ``DenseMoebiusGame`` via direct Kronecker transform."""
    return DenseMoebiusGame(
        _fast_fourier_to_moebius_transform(game.values, game.n_players),
        game.players,
    )


# --- Fallback 2**n evaluation materializers ---


def _materialize_dense_values[P](
    game: FinitePlayerGame[P, Any],
    basis: Basis,
    target_name: str,
) -> tuple[Float[jax.Array, "..."], FinitePlayers[P]]:
    """Evaluate ``game`` on every subset in ``basis`` to build a dense table and aligned players."""
    spec = game._value_spec()  # noqa: SLF001 - internal lazy abstract dtype hook
    if not spec.weak_type and not jnp.issubdtype(spec.dtype, jnp.floating):
        msg = f"{target_name} requires floating jax.Array values, got dtype={spec.dtype}."
        raise CastUnavailable(msg)
    n_players = game.n_players
    patterns = subset_patterns(n_players).reshape(2**n_players, *(1,) * game.ndim, n_players)
    raw_values = (
        game.eval(patterns)
        if basis is Basis.COALITION
        else game.coefficients(patterns, basis=basis)
    )
    if not isinstance(raw_values, jax.Array) or not jnp.issubdtype(raw_values.dtype, jnp.floating):
        msg = (
            f"{target_name} requires floating jax.Array values, got "
            f"{type(raw_values).__name__} (dtype={getattr(raw_values, 'dtype', None)})."
        )
        raise CastUnavailable(msg)
    values = jnp.moveaxis(raw_values, 0, -1)
    players = game.players
    if not isinstance(values, Tracer):
        devices = values.devices()
        if len(devices) == 1 and players.devices() != devices:
            players = players.to_device(next(iter(devices)))
    return values, players


# High fallback costs so any fast cast chain beats full 2**n evaluation, while native-basis
# materialization + O(n * 2**n) Kronecker transform (1003) beats generic coalition fallback (1010).
@caster(cost=_DENSE_NATIVE_FALLBACK_COST)
def _coalition_defined_to_dense_coalition[P](
    game: CoalitionDefinedGame[P, Any],
) -> DenseCoalitionGame[P]:
    """Evaluate a native Coalition game on every coalition to materialize a ``DenseCoalitionGame``."""
    values, players = _materialize_dense_values(game, Basis.COALITION, "DenseCoalitionGame")
    return DenseCoalitionGame(values, players)


@caster(cost=_DENSE_FALLBACK_COST)
def _finite_game_to_dense_coalition[P](game: FinitePlayerGame[P, Any]) -> DenseCoalitionGame[P]:
    """Evaluate ``game`` on every coalition to materialize a ``DenseCoalitionGame``."""
    values, players = _materialize_dense_values(game, Basis.COALITION, "DenseCoalitionGame")
    return DenseCoalitionGame(values, players)


@caster(cost=_DENSE_NATIVE_FALLBACK_COST)
def _moebius_defined_to_dense_moebius[P](game: MoebiusDefinedGame[P, Any]) -> DenseMoebiusGame[P]:
    """Evaluate a native Möbius game on every subset to materialize a ``DenseMoebiusGame``."""
    values, players = _materialize_dense_values(game, Basis.MOEBIUS, "DenseMoebiusGame")
    return DenseMoebiusGame(values, players)


@caster(cost=_DENSE_NATIVE_FALLBACK_COST)
def _fourier_defined_to_dense_fourier[P](game: FourierDefinedGame[P, Any]) -> DenseFourierGame[P]:
    """Evaluate a native Fourier game on every subset to materialize a ``DenseFourierGame``."""
    values, players = _materialize_dense_values(game, Basis.FOURIER, "DenseFourierGame")
    return DenseFourierGame(values, players)
