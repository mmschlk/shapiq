"""Lazy dtype and device conversion for evaluated coalition values."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, override

import jax
import jax.numpy as jnp
from flextype import CastUnavailable, castmethod, try_cast
from jaxtyping import Bool, Shaped, UInt8
from ndlike.protected_axis.jax import JaxAtMethod, JaxAxisProtected

from shapiq.players import FinitePlayers
from shapiq.typing import typechecked
from shapiq.value import Value
from shapiq.value._dtype import _resolve_dtype, cast_value, is_numeric_dtype

from .dense import DenseCoalitionGame
from .finite import Basis, CoalitionDefinedGame, FinitePlayerGame, _FiniteGame

if TYPE_CHECKING:
    from jax.sharding import Sharding
    from jax.typing import DTypeLike
    from numpy.typing import DTypeLike as NpDTypeLike


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class CastGame[P, V: Value, GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[P, V]](
    CoalitionDefinedGame[P, V],
    JaxAxisProtected["CastGame[P, V, GameT]", Any, _FiniteGame[P]],
):
    """Cast or place each evaluated value, keeping the child game on its original device.

    A target dtype is always strong. With ``target_dtype=None``, only device placement changes.
    """

    game: GameT
    target_dtype: Any
    device: Any = field(default=None, kw_only=True)
    protected_axes: ClassVar[dict[str, int]] = {"game": 0}
    permitted_at_methods: ClassVar[set[JaxAtMethod]] = set()

    def __post_init__(self) -> None:
        """Normalize an explicit target dtype under JAX's active dtype configuration."""
        if self.target_dtype is not None:
            target = _resolve_dtype(self.target_dtype, "CastGame target_dtype", allow_bool=True)
            object.__setattr__(self, "target_dtype", target)

    def _preserves_additivity(self) -> bool:
        """Return whether this cast preserves additivity linearly on Möbius coefficients.

        Floating/complex casts preserve structural additivity under ADR 0011's
        floating-point rounding policy; integer targets require a lossless
        widening or same-range integer conversion so modular wraparound cannot
        introduce higher-order interactions.
        """
        if self.target_dtype is None:
            return True
        if not is_numeric_dtype(self.target_dtype):
            return False
        src_dtype = self.game.dtype
        if jnp.issubdtype(self.target_dtype, jnp.integer):
            if not jnp.issubdtype(src_dtype, jnp.integer):
                return False
            src_info = jnp.iinfo(src_dtype)
            dst_info = jnp.iinfo(self.target_dtype)
            return dst_info.min <= src_info.min and dst_info.max >= src_info.max
        return True

    @property
    @override
    def players(self) -> FinitePlayers[P]:
        """Return the wrapped game's players."""
        return self.game.players

    @property
    @override
    def dtype(self) -> jnp.dtype:
        """Return the target dtype without evaluating the game."""
        return self._value_spec().dtype

    @override
    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Return the strong target dtype or the unchanged child value spec."""
        if self.target_dtype is None:
            return self.game._value_spec()  # noqa: SLF001 - internal lazy abstract dtype hook
        return jax.ShapeDtypeStruct((), self.target_dtype)

    @override
    def _known_additivity(self) -> int | None:
        """Propagate the child's known additivity bound when this cast preserves additivity."""
        if self.n_players == 0:
            return 0
        child_bound = self.game._known_additivity()  # noqa: SLF001 - internal static bound check
        if child_bound is None or not self._preserves_additivity():
            return None
        return child_bound

    @override
    def astype(
        self,
        dtype: "DTypeLike | NpDTypeLike | None",
        copy: bool = False,
        device: "jax.Device | Sharding | None" = None,
    ) -> Any:
        """Keep intermediate casts and apply further dtype or device changes after them."""
        if dtype is None and device is None:
            return self.copy() if copy else self
        if dtype is not None and not copy and device is None and self.device is None:
            target = _resolve_dtype(dtype, "CastGame target_dtype", allow_bool=True)
            if target == self.target_dtype:
                return self
        return CastGame(self.copy() if copy else self, dtype, device=device)

    def _cast_value(self, value: Shaped[V, "..."]) -> Shaped[V, "..."]:
        """Apply this game's cast to an evaluated value or coefficient."""
        return cast_value(value, self.target_dtype, copy=False, device=self.device)

    @override
    def empty_value(self) -> Shaped[V, "..."]:
        """Cast the child's empty-coalition value directly."""
        return self._cast_value(self.game.empty_value())

    @override
    def _zeros_like_value(self, shape: tuple[int, ...]) -> Shaped[V, "..."]:
        """Cast the child's lazy zero template directly."""
        if type(self.game)._zeros_like_value is not FinitePlayerGame._zeros_like_value:  # noqa: SLF001
            return self._cast_value(self.game._zeros_like_value(shape))  # noqa: SLF001
        return super(CastGame, self)._zeros_like_value(shape)

    @override
    def _coalition_bool(
        self, coalition: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        return self._cast_value(self.game.eval(coalition))

    @override
    def _coalition_packed(
        self, coalition: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        return self._cast_value(self.game.eval(coalition))

    @override
    def _moebius_bool(
        self, interaction: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        if self._preserves_additivity():
            return self._cast_value(self.game.moebius_coefficients(interaction))
        return super(CastGame, self)._moebius_bool(interaction)

    @override
    def _moebius_packed(
        self, interaction: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        if self._preserves_additivity():
            return self._cast_value(self.game.moebius_coefficients(interaction))
        return super(CastGame, self)._moebius_packed(interaction)

    @override
    def _fourier_bool(
        self, interaction: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        if self.target_dtype is None:
            return self._cast_value(self.game.fourier_coefficients(interaction))
        return super(CastGame, self)._fourier_bool(interaction)

    @override
    def _fourier_packed(
        self, interaction: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        if self.target_dtype is None:
            return self._cast_value(self.game.fourier_coefficients(interaction))
        return super(CastGame, self)._fourier_packed(interaction)

    @override
    def _coefficients_by_order(self, order: int, basis: Basis) -> Shaped[V, "..."]:
        if basis is Basis.MOEBIUS and self._preserves_additivity():
            return self._cast_value(self.game.moebius_coefficients(order=order))
        if basis is Basis.FOURIER and self.target_dtype is None:
            return self._cast_value(self.game.fourier_coefficients(order=order))
        return super(CastGame, self)._coefficients_by_order(order, basis)

    @castmethod(DenseCoalitionGame, cost=1.0)
    def _as_dense_coalition(self) -> DenseCoalitionGame[P]:
        """Cast the child's dense coalition table directly when it has a fast materialization."""
        if self.target_dtype is None or jnp.issubdtype(self.target_dtype, jnp.floating):
            result = try_cast(self.game, DenseCoalitionGame, max_cost=1.0)
            if result is not None:
                return result.value.astype(self.target_dtype, device=self.device)
        msg = "CastGame has no fast floating DenseCoalitionGame representation."
        raise CastUnavailable(msg)
