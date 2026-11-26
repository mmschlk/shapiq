"""Lazy dtype conversion for evaluated coalition values."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, override

import jax
import jax.numpy as jnp
from flextype import CastUnavailable, autocast, castmethod, try_cast
from jaxtyping import Bool, Shaped, UInt8
from ndlike.protected_axis.jax import JaxAtMethod, JaxAxisProtected

from shapiq.players import FinitePlayers
from shapiq.typing import typechecked
from shapiq.value import Value
from shapiq.value._dtype import _resolve_dtype, cast_value, is_numeric_dtype

from .finite import FinitePlayerGame, _FiniteGame
from .kadditive import AdditiveGame, KAdditiveGame
from .table import TableGame

if TYPE_CHECKING:
    from jax.sharding import Sharding
    from jax.typing import DTypeLike
    from numpy.typing import DTypeLike as NpDTypeLike


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class CastGame[P, V: Value, GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[P, V]](
    FinitePlayerGame[P, V],
    JaxAxisProtected["CastGame[P, V, GameT]", Any, _FiniteGame[P]],
    metaclass=autocast(KAdditiveGame, max_cost=0.0),
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
    def _eval_bool(self, coalition: Bool[jax.Array, "... {self.n_players}"]) -> Shaped[V, "..."]:
        return self._cast_value(self.game.eval(coalition))

    @override
    def _eval_packed(
        self, coalition: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        return self._cast_value(self.game.eval(coalition))

    def _refinable_child[T: KAdditiveGame](self, capability: type[T]) -> T:
        """Require a zero-cost child view and a cast that preserves additivity."""
        if self.target_dtype is None or is_numeric_dtype(self.target_dtype):
            result = try_cast(self.game, capability, max_cost=0.0)
            if result is not None and not (
                self.target_dtype is not None
                and jnp.issubdtype(result.value.dtype, jnp.inexact)
                and jnp.issubdtype(self.target_dtype, jnp.integer)
            ):
                return result.value
        msg = f"CastGame cannot preserve {capability.__name__} for this child and target dtype."
        raise CastUnavailable(msg)

    @castmethod(KAdditiveGame)
    def _as_kadditive(self) -> KAdditiveGame[P, V]:
        """Refine this cast of a k-additive game."""
        return _KAdditiveCastGame(
            self._refinable_child(KAdditiveGame), self.target_dtype, device=self.device
        )

    @castmethod(AdditiveGame)
    def _as_additive(self) -> AdditiveGame[P, V]:
        """Refine this cast of an additive game."""
        return _AdditiveCastGame(
            self._refinable_child(AdditiveGame), self.target_dtype, device=self.device
        )

    @castmethod(TableGame, cost=1.0)
    def _as_table(self) -> TableGame[P]:
        """Cast the child's table directly when it has a fast materialization."""
        if self.target_dtype is None or jnp.issubdtype(self.target_dtype, jnp.floating):
            result = try_cast(self.game, TableGame, max_cost=1.0)
            if result is not None:
                return result.value.astype(self.target_dtype, device=self.device)
        msg = "CastGame has no fast floating table representation."
        raise CastUnavailable(msg)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class _KAdditiveCastGame[P, V: Value, GameT: KAdditiveGame[Any, Any] = KAdditiveGame[P, V]](
    CastGame[P, V, GameT], KAdditiveGame[P, V]
):
    """Cast the values and Möbius coefficients of a k-additive game."""

    @property
    @override
    def additivity(self) -> int:
        return self.game.additivity

    @override
    def empty_value(self) -> Shaped[V, "..."]:
        return self._cast_value(self.game.empty_value())

    @override
    def _zeros_like_value(self, shape: tuple[int, ...]) -> Shaped[V, "..."]:
        return self._cast_value(self.game._zeros_like_value(shape))  # noqa: SLF001 - internal KAdditiveGame hook

    @override
    def _coefficients_by_order(self, order: int) -> Shaped[V, "..."]:
        return self._cast_value(self.game.coefficients(order=order))

    @override
    def _coefficients_for_interactions(
        self,
        interactions: Bool[jax.Array, "#*batch {self.n_players}"],
        *,
        max_order: int | None = None,
    ) -> Shaped[V, "..."]:
        return self._cast_value(
            self.game._coefficients_for_interactions(interactions, max_order=max_order)  # noqa: SLF001 - internal KAdditiveGame hook
        )


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class _AdditiveCastGame[P, V: Value](
    _KAdditiveCastGame[P, V, AdditiveGame[P, V]], AdditiveGame[P, V]
):
    """Cast the values and singleton contributions of an additive game."""

    @property
    @override
    def additivity(self) -> Literal[0, 1]:
        return 0 if self.n_players == 0 else self.game.additivity

    @override
    def contributions(self) -> Shaped[V, "..."]:
        return self._cast_value(self.game.contributions())
