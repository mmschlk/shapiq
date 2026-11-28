"""Private operator dispatch and scalar division/power adapters for finite games."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, cast, override

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from jaxtyping import Bool, Shaped, UInt8
from ndlike import jax_functions as jf
from ndlike.jax_like import JaxLike
from ndlike.tuple import BroadcastTuple
from ndlike.tuple.jax import JaxBroadcastTuple

from shapiq.coalition import FiniteCoalition
from shapiq.players import Players
from shapiq.typing import typechecked
from shapiq.value._dtype import (
    is_numeric_dtype,
    op_spec,
    promote_spec,
    unit_scalar,
    validate_numeric_value,
)
from shapiq.value.value import JaxValue

from .arithmetic import (
    AffineGame,
    ArithmeticGame,
    MappedGame,
    ProductGame,
    _ensure_shape,
    _zeros_from_arithmetic,
)
from .base import Game
from .finite import Basis, FinitePlayerGame


def _known_int_exponent(other: object) -> int | None:
    """Return a concrete integer scalar exponent (either sign), or ``None`` if general/dynamic."""
    if isinstance(other, (bool, np.bool_)):
        return None
    if isinstance(other, (int, np.integer)):
        return int(other)
    if isinstance(other, np.ndarray):
        if other.ndim == 0 and np.issubdtype(other.dtype, np.integer):
            return int(other.item())
        return None
    if isinstance(other, jax.Array):
        if (
            not isinstance(other, Tracer)
            and other.ndim == 0
            and jnp.issubdtype(other.dtype, jnp.integer)
        ):
            return int(np.asarray(other).item())
        return None
    return None


def _is_concrete_nonzero(
    divisor: JaxValue,
    raw: object = None,
    *,
    target_dtype: object = None,
) -> bool:
    """Return whether every leaf of ``divisor`` is concretely finite and nonzero at division dtype."""
    cast_dtype = None if target_dtype is None else jnp.dtype(cast("Any", target_dtype))
    if isinstance(raw, (int, float, complex, np.number, np.ndarray)) and isinstance(
        divisor, jax.Array
    ):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            arr = np.asarray(raw, dtype=divisor.dtype)
            if not bool(np.all(np.isfinite(arr) & (arr != 0))):
                return False
            if cast_dtype is not None and arr.dtype != cast_dtype:
                arr = np.asarray(arr, dtype=cast_dtype)
                if not bool(np.all(np.isfinite(arr) & (arr != 0))):
                    return False
        return True
    leaves = jax.tree.leaves(divisor)
    if not leaves:
        return False
    for leaf in leaves:
        if isinstance(leaf, Tracer):
            return False
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            arr = np.asarray(leaf)
            if not bool(np.all(np.isfinite(arr) & (arr != 0))):
                return False
            if cast_dtype is not None and arr.dtype != cast_dtype:
                arr = np.asarray(arr, dtype=cast_dtype)
                if not bool(np.all(np.isfinite(arr) & (arr != 0))):
                    return False
    return True


def _divide_values(values: tuple[JaxValue, ...]) -> JaxValue:
    """Top-level pure quotient for ``game / game`` (no closure, stable JAX static metadata)."""
    left, right = values
    return jf.jax_true_divide(left, right)


@jax.jit
def _jit_dynamic_power(left: Any, right: Any) -> Any:  # noqa: ANN401 - generic over structured JaxValue pytrees
    """JIT-traced fallback ensuring dynamic power semantics on structured ``JaxValue``s."""
    return jf.jax_power(left, right)


def _dynamic_power(left: JaxValue, right: JaxValue) -> JaxValue:
    """Evaluate ``left ** right`` along JAX's dynamic power path."""
    if isinstance(left, jax.Array) and isinstance(right, jax.Array):
        if (
            not isinstance(right, Tracer)
            and right.ndim == 0
            and jnp.issubdtype(right.dtype, jnp.integer)
        ):
            return jnp.power(jnp.expand_dims(left, 0), jnp.expand_dims(right, 0))[0]
        return jnp.power(left, right)
    return cast("JaxValue", _jit_dynamic_power(left, right))


def _power_values(values: tuple[JaxValue, ...]) -> JaxValue:
    """Top-level dynamic power for ``game ** game`` (no closure, uniform dynamic dtype)."""
    left, right = values
    return _dynamic_power(left, right)


# --- Private direct-division adapter (avoids reciprocal underflow; dynamic pytree constant) ---


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class _ScalarDivGame[
    P,
    V: JaxValue = JaxValue,
    GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[P, Any],
](ArithmeticGame[P, V, GameT, "_ScalarDivGame[P, V, GameT]"]):
    """Private direct ``true_divide`` adapter for ``game / constant`` or ``constant / game``."""

    constant: JaxValue
    reciprocal: bool = field(default=False, kw_only=True)
    known_nonzero: bool = field(default=False, kw_only=True)
    protected_axes: ClassVar[dict[str, int]] = {"games": 0, "constant": 0}
    broadcasted_batch: ClassVar[set[str]] = {"constant"}

    if TYPE_CHECKING:

        def __init__[P2, V2: JaxValue = JaxValue](
            self: "_ScalarDivGame[P2, V2, FinitePlayerGame[P2, Any]]",
            games: JaxBroadcastTuple[tuple[FinitePlayerGame[P2, Any], ...]],
            constant: JaxValue,
            *,
            reciprocal: bool = ...,
            known_nonzero: bool = ...,
        ) -> None: ...

    def __post_init__(self) -> None:
        super(_ScalarDivGame, self).__post_init__()
        if len(self.games.members) != 1:
            msg = "_ScalarDivGame requires exactly one game."
            raise ValueError(msg)
        validate_numeric_value(self.constant, "Division constant")
        self.validate()

    def _is_linear_division(self) -> bool:
        """Return whether this division scales the child linearly without zero/nonfinite division."""
        return not self.reciprocal and (
            self.known_nonzero or _is_concrete_nonzero(self.constant, target_dtype=self.dtype)
        )

    @override
    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Return exact JAX ``true_divide`` dtype and ``weak_type`` without evaluating."""
        lhs, rhs = (
            (self.constant, self.games.members[0])
            if self.reciprocal
            else (self.games.members[0], self.constant)
        )
        return op_spec(jnp.true_divide, lhs, rhs)

    @property
    @override
    def dtype(self) -> jnp.dtype:
        """Return exact JAX ``true_divide`` dtype without evaluating the child game."""
        return self._value_spec().dtype

    @override
    def _known_additivity(self) -> int | None:
        """Return the child's known additivity bound when dividing by a concrete nonzero constant."""
        if self.n_players == 0:
            return 0
        child_bound = self.games.members[0]._known_additivity()  # noqa: SLF001 - internal static bound check
        if child_bound is None or not self._is_linear_division():
            return None
        return child_bound

    @override
    def _zeros_like_value(self, shape: tuple[int, ...]) -> Shaped[V, "..."]:
        """Construct zero values lazily matching the quotient representation."""
        return cast(
            "Shaped[V, '...']",
            _zeros_from_arithmetic(self.games.members, (self.constant,), shape, self.dtype),
        )

    @override
    def empty_value(self) -> Shaped[V, "..."]:
        """Evaluate the empty coalition directly via child ``empty_value``."""
        return _ensure_shape(self._compute((self.games.members[0].empty_value(),)), self.shape)

    @classmethod
    @override
    def _combine_sequence_metadata(
        cls,
        items: Sequence["_ScalarDivGame[Any, Any, Any]"],
        *,
        func: Callable[..., Any] | None,
    ) -> dict[str, Any]:
        """Combine ``known_nonzero`` conservatively across stacked/concatenated divisions."""
        del func
        return {"known_nonzero": all(item.known_nonzero for item in items)}

    @override
    def _compute(self, values: tuple[JaxValue, ...]) -> Shaped[V, "..."]:
        lhs, rhs = (self.constant, values[0]) if self.reciprocal else (values[0], self.constant)
        return cast("Shaped[V, '...']", jf.jax_true_divide(lhs, rhs))

    def _divide_last_axis(self, term: JaxValue) -> Shaped[V, "..."]:
        result = jf.jax_true_divide(term, jf.jax_expand_dims(self.constant, axis=-1))
        return _ensure_shape(cast("Shaped[V, '...']", result), (*self.shape, term.shape[-1]))

    def _linear_basis_query(self, interaction: FiniteCoalition, basis: Basis) -> Shaped[V, "..."]:
        child = self.games.members[0]
        result = jf.jax_true_divide(child.coefficients(interaction, basis=basis), self.constant)
        target_batch = jnp.broadcast_shapes(self.shape, interaction.shape[:-1])
        return _ensure_shape(cast("Shaped[V, '...']", result), target_batch)

    @override
    def _moebius_bool(
        self, interaction: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        if self._is_linear_division():
            return self._linear_basis_query(interaction, Basis.MOEBIUS)
        return super(_ScalarDivGame, self)._moebius_bool(interaction)

    @override
    def _moebius_packed(
        self, interaction: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        if self._is_linear_division():
            return self._linear_basis_query(interaction, Basis.MOEBIUS)
        return super(_ScalarDivGame, self)._moebius_packed(interaction)

    @override
    def _fourier_bool(
        self, interaction: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        if self._is_linear_division():
            return self._linear_basis_query(interaction, Basis.FOURIER)
        return super(_ScalarDivGame, self)._fourier_bool(interaction)

    @override
    def _fourier_packed(
        self, interaction: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        if self._is_linear_division():
            return self._linear_basis_query(interaction, Basis.FOURIER)
        return super(_ScalarDivGame, self)._fourier_packed(interaction)

    @override
    def _coefficients_by_order(self, order: int, basis: Basis) -> Shaped[V, "..."]:
        if basis in (Basis.MOEBIUS, Basis.FOURIER) and self._is_linear_division():
            n_combos = math.comb(self.n_players, order)
            known = self._known_additivity()
            if n_combos == 0 or (known is not None and order > known):
                return self._zeros_like_value((*self.shape, n_combos))
            if order == 0 and basis is Basis.MOEBIUS:
                return cast("Shaped[V, '...']", jf.jax_expand_dims(self.empty_value(), axis=-1))
            child = self.games.members[0]
            return self._divide_last_axis(child.coefficients(order=order, basis=basis))
        return super(_ScalarDivGame, self)._coefficients_by_order(order, basis)


# --- Private direct-power adapters (static int exponent vs dynamic constant leaf) ---


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class _IntPowerGame[
    P,
    V: JaxValue = JaxValue,
    GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[P, Any],
](ArithmeticGame[P, V, GameT, "_IntPowerGame[P, V, GameT]"]):
    """Private ``game ** m`` adapter storing a concrete scalar integer ``exponent`` statically."""

    exponent: int

    if TYPE_CHECKING:

        def __init__[P2, V2: JaxValue = JaxValue](
            self: "_IntPowerGame[P2, V2, FinitePlayerGame[P2, Any]]",
            games: JaxBroadcastTuple[tuple[FinitePlayerGame[P2, Any], ...]],
            exponent: int,
        ) -> None: ...

    def __post_init__(self) -> None:
        super(_IntPowerGame, self).__post_init__()
        if len(self.games.members) != 1:
            msg = "_IntPowerGame requires exactly one game."
            raise ValueError(msg)
        if not isinstance(self.exponent, int) or isinstance(self.exponent, bool):
            msg = "_IntPowerGame exponent must be an integer."
            raise TypeError(msg)
        self.validate()

    @override
    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Return JAX's ``lax.integer_pow`` dtype and ``weak_type``."""
        return op_spec(lambda x: jnp.power(x, 0), self.games.members[0])

    @property
    @override
    def dtype(self) -> jnp.dtype:
        """Return JAX's ``lax.integer_pow`` dtype (preserves numeric base dtype)."""
        return self._value_spec().dtype

    @override
    def _known_additivity(self) -> int | None:
        """Return ``min(n_players, exponent * base._known_additivity())`` for ``exponent >= 0``."""
        if self.n_players == 0:
            return 0
        if self.exponent < 0:
            return None
        base_bound = self.games.members[0]._known_additivity()  # noqa: SLF001 - internal O(1) bound check
        if base_bound is None:
            return None
        return min(self.n_players, self.exponent * base_bound)

    @override
    def _zeros_like_value(self, shape: tuple[int, ...]) -> Shaped[V, "..."]:
        """Construct zero values lazily from the base game without evaluating coalitions."""
        return cast(
            "Shaped[V, '...']",
            _zeros_from_arithmetic(self.games.members, (), shape, self.dtype),
        )

    @override
    def _compute(self, values: tuple[JaxValue, ...]) -> Shaped[V, "..."]:
        return cast("Shaped[V, '...']", jf.jax_power(values[0], self.exponent))


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class _ScalarPowerGame[
    P,
    V: JaxValue = JaxValue,
    GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[P, Any],
](ArithmeticGame[P, V, GameT, "_ScalarPowerGame[P, V, GameT]"]):
    """Private dynamic-constant ``power`` adapter for general ``game ** c`` and ``c ** game``."""

    constant: JaxValue
    reflected: bool = field(default=False, kw_only=True)
    protected_axes: ClassVar[dict[str, int]] = {"games": 0, "constant": 0}
    broadcasted_batch: ClassVar[set[str]] = {"constant"}

    def __post_init__(self) -> None:
        super(_ScalarPowerGame, self).__post_init__()
        if len(self.games.members) != 1:
            msg = "_ScalarPowerGame requires exactly one game."
            raise ValueError(msg)
        validate_numeric_value(self.constant, "Power constant")
        self.validate()

    @override
    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Return exact dynamic JAX ``power`` dtype and ``weak_type`` without evaluating."""
        lhs, rhs = (
            (self.constant, self.games.members[0])
            if self.reflected
            else (self.games.members[0], self.constant)
        )
        return op_spec(jnp.power, lhs, rhs)

    @property
    @override
    def dtype(self) -> jnp.dtype:
        """Return exact dynamic JAX ``power`` dtype without evaluating the child game."""
        return self._value_spec().dtype

    @override
    def _compute(self, values: tuple[JaxValue, ...]) -> Shaped[V, "..."]:
        lhs, rhs = (self.constant, values[0]) if self.reflected else (values[0], self.constant)
        return cast("Shaped[V, '...']", _dynamic_power(lhs, rhs))


# --- Operator helpers invoked via FinitePlayerGame.__jax_function__ ---


def _as_numeric_constant(other: object) -> JaxValue | None:
    """Normalize a numeric scalar, array, or numeric ``JaxLike`` constant, or return ``None``."""
    if isinstance(other, (bool, np.bool_)):
        return None
    if isinstance(other, (int, float, complex, np.number)):
        return jnp.asarray(other)
    if isinstance(other, (jax.Array, np.ndarray)):
        return jnp.asarray(other) if is_numeric_dtype(other.dtype) else None
    if isinstance(other, JaxLike) and not isinstance(other, (Game, Players, BroadcastTuple)):
        return other if is_numeric_dtype(other.dtype) else None
    return None


def _dispatch_game_pair(
    func: Callable[..., Any],
    left: FinitePlayerGame[Any, Any],
    right: FinitePlayerGame[Any, Any],
) -> Any:  # noqa: ANN401 - returns NotImplemented for unsupported ops
    """Dispatch binary arithmetic between two finite games."""
    if func is jf.jax_add:
        return AffineGame(
            games=JaxBroadcastTuple((left, right)),
            weights=JaxBroadcastTuple((unit_scalar(1, left), unit_scalar(1, right))),
            bias=unit_scalar(0, left),
        )
    if func is jf.jax_subtract:
        promoted = promote_spec((left, right), left.dtype)
        return AffineGame(
            games=JaxBroadcastTuple((left, right)),
            weights=JaxBroadcastTuple((unit_scalar(1, left), unit_scalar(-1, promoted))),
            bias=unit_scalar(0, promoted),
        )
    if func is jf.jax_multiply:
        return ProductGame(games=JaxBroadcastTuple((left, right)))
    if func is jf.jax_true_divide:
        return MappedGame(
            games=JaxBroadcastTuple((left, right)),
            fn=_divide_values,
        )
    if func is jf.jax_power:
        return MappedGame(
            games=JaxBroadcastTuple((left, right)),
            fn=_power_values,
        )
    return NotImplemented


def _dispatch_constant_op(
    func: Callable[..., Any],
    game: FinitePlayerGame[Any, Any],
    other: object,
    *,
    reflected: bool,
) -> Any:  # noqa: ANN401 - returns NotImplemented for unsupported operands
    """Dispatch binary arithmetic between a finite game and a numeric constant."""
    if func is jf.jax_power and not reflected:
        exponent = _known_int_exponent(other)
        if exponent is not None:
            return _IntPowerGame(games=JaxBroadcastTuple((game,)), exponent=exponent)
    constant = _as_numeric_constant(other)
    if constant is None:
        return NotImplemented
    target_shape = jnp.broadcast_shapes(game.shape, constant.shape)
    aligned: FinitePlayerGame[Any, Any] = (
        game if target_shape == game.shape else jf.jax_broadcast_to(game, target_shape)
    )
    if func is jf.jax_add:
        return AffineGame(
            games=JaxBroadcastTuple((aligned,)),
            weights=JaxBroadcastTuple((unit_scalar(1, aligned),)),
            bias=constant,
        )
    if func is jf.jax_multiply:
        return AffineGame(
            games=JaxBroadcastTuple((aligned,)),
            weights=JaxBroadcastTuple((constant,)),
            bias=unit_scalar(0, aligned),
        )
    if func is jf.jax_subtract:
        promoted = promote_spec((aligned, constant), aligned.dtype)
        if reflected:
            return AffineGame(
                games=JaxBroadcastTuple((aligned,)),
                weights=JaxBroadcastTuple((unit_scalar(-1, promoted),)),
                bias=constant,
            )
        neg_constant = jf.jax_multiply(unit_scalar(-1, promoted), constant)
        return AffineGame(
            games=JaxBroadcastTuple((aligned,)),
            weights=JaxBroadcastTuple((unit_scalar(1, aligned),)),
            bias=neg_constant,
        )
    if func is jf.jax_true_divide:
        if reflected:
            return _ScalarDivGame(
                games=JaxBroadcastTuple((aligned,)), constant=constant, reciprocal=True
            )
        compute_dtype = op_spec(jnp.true_divide, aligned, constant).dtype
        return _ScalarDivGame(
            games=JaxBroadcastTuple((aligned,)),
            constant=constant,
            known_nonzero=_is_concrete_nonzero(constant, raw=other, target_dtype=compute_dtype),
        )
    if func is jf.jax_power:
        return _ScalarPowerGame(
            games=JaxBroadcastTuple((aligned,)), constant=constant, reflected=reflected
        )
    return NotImplemented


def _dispatch_game_arithmetic(func: Callable[..., Any], args: tuple[Any, ...]) -> Any:  # noqa: ANN401 - returns NotImplemented for non-arithmetic ops
    """Single dispatch entry point for Python operators and ndlike ``jax_*`` arithmetic."""
    if len(args) == 1 and isinstance(args[0], FinitePlayerGame):
        game = args[0]
        if func is jf.jax_positive:
            return game
        if func is jf.jax_negative:
            return AffineGame(
                games=JaxBroadcastTuple((game,)),
                weights=JaxBroadcastTuple((unit_scalar(-1, game),)),
                bias=unit_scalar(0, game),
            )
        return NotImplemented
    if len(args) != 2:
        return NotImplemented
    left, right = args
    if isinstance(left, FinitePlayerGame) and isinstance(right, FinitePlayerGame):
        return _dispatch_game_pair(func, left, right)
    if isinstance(left, FinitePlayerGame):
        return _dispatch_constant_op(func, left, right, reflected=False)
    if isinstance(right, FinitePlayerGame):
        return _dispatch_constant_op(func, right, left, reflected=True)
    return NotImplemented
