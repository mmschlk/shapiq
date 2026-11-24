"""Private operator dispatch and scalar division/power adapters for finite games."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, override

import jax
import jax.numpy as jnp
import numpy as np
from flextype import CastUnavailable, autocast, castmethod
from jax.core import Tracer
from jaxtyping import Bool, Shaped
from ndlike import jax_functions as jf
from ndlike.jax_like import JaxLike
from ndlike.tuple._common import BroadcastTuple
from ndlike.tuple.jax import JaxBroadcastTuple

from shapiq.players import Players
from shapiq.typing import typechecked
from shapiq.value.value import JaxValue

from .arithmetic import (
    AffineGame,
    ArithmeticGame,
    MappedGame,
    ProductGame,
    _abstract_value_spec,
    _ensure_shape,
    _is_numeric_dtype,
    _promote_spec,
    _require_refinable_children,
    _validate_numeric_value,
    _zeros_from_arithmetic,
)
from .finite import FinitePlayerGame
from .game import Game
from .kadditive import AdditiveGame, KAdditiveGame
from .moebius import _moebius_coefficients_by_eval, _moebius_zero_like


def _unit_scalar(value: int, reference: object) -> jax.Array:
    """Return a dynamic scalar ``0``, ``1``, or ``-1`` preserving ``reference``'s dtype/weakness."""
    spec = _abstract_value_spec(reference)
    if spec is None or not _is_numeric_dtype(spec.dtype):
        return jnp.asarray(value)
    if spec.weak_type:
        if value == -1:
            if jnp.issubdtype(spec.dtype, jnp.complexfloating):
                return jnp.asarray(-1.0 + 0j)
            if jnp.issubdtype(spec.dtype, jnp.floating):
                return jnp.asarray(-1.0)
        return jnp.asarray(value)
    dt = spec.dtype
    # Use ``-ones`` for -1 so unsigned dtypes wrap cleanly without NumPy/JAX conversion errors.
    return -jnp.ones((), dtype=dt) if value == -1 else jnp.full((), value, dtype=dt)


def _dtype_operand(value: object) -> jax.ShapeDtypeStruct:
    """Build a zero-cost abstract operand preserving ``weak_type`` on JAX arrays and games."""
    spec = _abstract_value_spec(value)
    if spec is not None:
        return spec
    msg = f"Cannot determine numeric dtype for {type(value).__name__}."
    raise TypeError(msg)


def _true_divide_spec(left: object, right: object) -> jax.ShapeDtypeStruct:
    """Compute JAX's exact ``true_divide`` result dtype and ``weak_type`` without evaluating."""
    out = jax.eval_shape(jnp.true_divide, _dtype_operand(left), _dtype_operand(right))
    return jax.ShapeDtypeStruct((), out.dtype, weak_type=out.weak_type)


def _power_spec(left: object, right: object) -> jax.ShapeDtypeStruct:
    """Compute JAX's dynamic ``power`` result dtype and ``weak_type`` without evaluating."""
    out = jax.eval_shape(jnp.power, _dtype_operand(left), _dtype_operand(right))
    return jax.ShapeDtypeStruct((), out.dtype, weak_type=out.weak_type)


def _integer_power_spec(base: object) -> jax.ShapeDtypeStruct:
    """Compute JAX's ``lax.integer_pow`` result dtype and ``weak_type``."""
    out = jax.eval_shape(lambda x: jnp.power(x, 0), _dtype_operand(base))
    return jax.ShapeDtypeStruct((), out.dtype, weak_type=out.weak_type)


def _known_int_exponent(other: object) -> int | None:
    """Return a concrete integer scalar exponent (either sign), or ``None`` if general/dynamic.

    Matches JAX's ``jnp.power`` concrete-integer specialization (which lowers scalar
    integer exponents to ``jax.lax.integer_pow`` without truncating through
    ``jnp.asarray``) while rejecting booleans as non-numeric. Concrete JAX scalars
    are checked against ``Tracer`` first so closed-over JAX arrays inside
    ``jax.jit`` are read without entering the active trace.
    """
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
    """Return whether every leaf of ``divisor`` is concretely finite and nonzero at division dtype.

    Host scalars/NumPy arrays are checked after normalization to ``divisor.dtype`` and
    again after conversion to ``target_dtype`` (the ``true_divide`` computation dtype),
    so values that underflow to zero or overflow to infinity under either conversion
    (e.g., Python ``1e-8`` or ``1e5`` with a ``float16`` game) never falsely refine.
    Concrete JAX leaves are inspected via host NumPy after a ``Tracer`` preflight so
    closed-over JAX arrays inside ``jax.jit`` never enter the active trace.
    """
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
    """Evaluate ``left ** right`` along JAX's dynamic power path.

    In eager mode, ``jnp.power(x, y)`` specializes concrete 0-d integer ``y`` to
    ``jax.lax.integer_pow`` (preserving ``x.dtype``), whereas batched and JIT-traced
    calls use JAX's dynamic power path (promoting mixed integer operands, e.g.
    ``int8 ** int32 -> int32``). Routing intrinsically dynamic powers (game-valued
    and general-constant operands) through this path keeps runtime dtypes uniform
    across scalar coalitions, batched coalitions, JIT, and lazy ``dtype`` inference.
    """
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
](
    ArithmeticGame[P, V, GameT, "_ScalarDivGame[P, V, GameT]"],
    metaclass=autocast(KAdditiveGame, max_cost=0.0),
):
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
        _validate_numeric_value(self.constant, "Division constant")
        self.validate()

    @override
    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Return exact JAX ``true_divide`` dtype and ``weak_type`` without evaluating."""
        lhs, rhs = (
            (self.constant, self.games.members[0])
            if self.reciprocal
            else (self.games.members[0], self.constant)
        )
        return _true_divide_spec(lhs, rhs)

    @property
    @override
    def dtype(self) -> jnp.dtype:
        """Return exact JAX ``true_divide`` dtype without evaluating the child game."""
        return self._value_spec().dtype

    @override
    def _compute(self, values: tuple[JaxValue, ...]) -> Shaped[V, "..."]:
        lhs, rhs = (self.constant, values[0]) if self.reciprocal else (values[0], self.constant)
        return cast("Shaped[V, '...']", jf.jax_true_divide(lhs, rhs))

    @castmethod
    def _as_additive(self) -> AdditiveGame[P, V]:
        """Refine division of an additive game by a nonzero divisor."""
        members = _require_refinable_div_children(self, AdditiveGame)
        return _AdditiveDivGame(
            games=JaxBroadcastTuple(members), constant=self.constant, known_nonzero=True
        )

    @castmethod
    def _as_kadditive(self) -> KAdditiveGame[P, V]:
        """Refine division of a k-additive game by a nonzero divisor."""
        members = _require_refinable_div_children(self, KAdditiveGame)
        return _KAdditiveDivGame(
            games=JaxBroadcastTuple(members), constant=self.constant, known_nonzero=True
        )


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class _KAdditiveDivGame[
    P,
    V: JaxValue = JaxValue,
    GameT: KAdditiveGame[Any, Any] = KAdditiveGame[P, Any],
](
    _ScalarDivGame[P, V, GameT],
    KAdditiveGame[P, V],
):
    """Private k-additive refinement of ``game / nonzero_constant``."""

    @property
    @override
    def additivity(self) -> int:
        return self.games.members[0].additivity

    @override
    def _zeros_like_value(self, shape: tuple[int, ...]) -> Shaped[V, "..."]:
        return cast(
            "Shaped[V, '...']",
            _zeros_from_arithmetic(self.games.members, (self.constant,), shape, self.dtype),
        )

    @override
    def empty_value(self) -> Shaped[V, "..."]:
        return _ensure_shape(self._compute((self.games.members[0].empty_value(),)), self.shape)

    def _divide_last_axis(self, term: JaxValue) -> Shaped[V, "..."]:
        result = jf.jax_true_divide(term, jf.jax_expand_dims(self.constant, axis=-1))
        return _ensure_shape(cast("Shaped[V, '...']", result), (*self.shape, term.shape[-1]))

    @override
    def _coefficients_by_order(self, order: int) -> Shaped[V, "..."]:
        n_combos = math.comb(self.n_players, order)
        if order > self.additivity or n_combos == 0:
            return self._zeros_like_value((*self.shape, n_combos))
        if order == 0:
            return cast("Shaped[V, '...']", jf.jax_expand_dims(self.empty_value(), axis=-1))
        child = self.games.members[0]
        return self._divide_last_axis(child.coefficients(order=order))

    @override
    def _coefficients_for_interactions(
        self,
        interactions: Bool[jax.Array, "#*batch {self.n_players}"],
        *,
        max_order: int | None = None,
    ) -> Shaped[V, "..."]:
        del max_order
        child = self.games.members[0]
        result = jf.jax_true_divide(child.coefficients(interactions), self.constant)
        target_batch = jnp.broadcast_shapes(self.shape, interactions.shape[:-1])
        return _ensure_shape(cast("Shaped[V, '...']", result), target_batch)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class _AdditiveDivGame[
    P,
    V: JaxValue = JaxValue,
    GameT: AdditiveGame[Any, Any] = AdditiveGame[P, Any],
](
    _KAdditiveDivGame[P, V, GameT],
    AdditiveGame[P, V],
):
    """Private additive refinement of ``game / nonzero_constant``."""

    @property
    @override
    def additivity(self) -> Literal[0, 1]:
        if self.n_players == 0:
            return 0
        return super(_AdditiveDivGame, self).additivity  # ty: ignore[invalid-explicit-return]

    @override
    def contributions(self) -> Shaped[V, "..."]:
        if self.n_players == 0 or self.additivity == 0:
            return self._zeros_like_value((*self.shape, self.n_players))
        child = self.games.members[0]
        return self._divide_last_axis(child.contributions())


def _require_refinable_div_children[T: FinitePlayerGame](
    game: _ScalarDivGame[Any, Any, Any],
    target: type[T],
) -> tuple[T, ...]:
    """Validate division refinement eligibility and zero-cost refine child games to ``target``."""
    if game.reciprocal or not (
        game.known_nonzero or _is_concrete_nonzero(game.constant, target_dtype=game.dtype)
    ):
        msg = "Division refinement requires dividing by a concrete nonzero constant."
        raise CastUnavailable(msg)
    return _require_refinable_children(game, target)


# --- Private direct-power adapters (static int exponent vs dynamic constant leaf) ---


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class _IntPowerGame[
    P,
    V: JaxValue = JaxValue,
    GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[P, Any],
](
    ArithmeticGame[P, V, GameT, "_IntPowerGame[P, V, GameT]"],
    metaclass=autocast(KAdditiveGame, max_cost=0.0),
):
    """Private ``game ** m`` adapter storing a concrete scalar integer ``exponent`` statically.

    Because ``exponent`` is static pytree metadata rather than a dynamic leaf, it is
    the single source of truth for both evaluation (JAX's ``lax.integer_pow``
    semantics, preserving base dtype and raising ``TypeError`` on negative powers of
    integers even under ``jax.jit``) and k-additive refinement (for ``exponent >= 0``
    and k-additive bases, yielding bound ``min(n_players, exponent * base.additivity)``).
    """

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
        return _integer_power_spec(self.games.members[0])

    @property
    @override
    def dtype(self) -> jnp.dtype:
        """Return JAX's ``lax.integer_pow`` dtype (preserves numeric base dtype)."""
        return self._value_spec().dtype

    @override
    def _compute(self, values: tuple[JaxValue, ...]) -> Shaped[V, "..."]:
        return cast("Shaped[V, '...']", jf.jax_power(values[0], self.exponent))

    @castmethod
    def _as_kadditive(self) -> KAdditiveGame[P, V]:
        """Refine a k-additive game raised to a nonnegative integer power."""
        if self.exponent < 0:
            msg = "Power refinement requires a nonnegative integer exponent."
            raise CastUnavailable(msg)
        members = _require_refinable_children(self, KAdditiveGame)
        return _KAdditivePowerGame(games=JaxBroadcastTuple(members), exponent=self.exponent)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class _KAdditivePowerGame[
    P,
    V: JaxValue = JaxValue,
    GameT: KAdditiveGame[Any, Any] = KAdditiveGame[P, Any],
](
    _IntPowerGame[P, V, GameT],
    KAdditiveGame[P, V],
):
    """Private k-additive refinement of ``game ** m`` for nonnegative static integer ``m``."""

    def __post_init__(self) -> None:
        super(_KAdditivePowerGame, self).__post_init__()
        if self.exponent < 0:
            msg = "K-additive power refinement requires a nonnegative integer exponent."
            raise ValueError(msg)

    @property
    @override
    def additivity(self) -> int:
        """Return ``min(n_players, exponent * base.additivity)`` without evaluating coalitions."""
        base_bound = self.games.members[0].additivity
        return min(self.n_players, self.exponent * base_bound)

    @override
    def _zeros_like_value(self, shape: tuple[int, ...]) -> Shaped[V, "..."]:
        """Construct zero values lazily from the base game without evaluating coalitions."""
        return cast(
            "Shaped[V, '...']",
            _zeros_from_arithmetic(self.games.members, (), shape, self.dtype),
        )

    @override
    def _coefficients_by_order(self, order: int) -> Shaped[V, "..."]:
        """Match Möbius-inversion dtypes on zero-shortcut order queries."""
        n_combos = math.comb(self.n_players, order)
        if order > self.additivity or n_combos == 0:
            return _moebius_zero_like(self, (*self.shape, n_combos))
        return super(_KAdditivePowerGame, self)._coefficients_by_order(order)

    @override
    def _coefficients_for_interactions(
        self,
        interactions: Bool[jax.Array, "#*batch {self.n_players}"],
        *,
        max_order: int | None = None,
    ) -> Shaped[V, "..."]:
        """Compute integer-power Möbius coefficients explicitly via subset inversion."""
        return _moebius_coefficients_by_eval(self, interactions, max_order=max_order)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class _ScalarPowerGame[
    P,
    V: JaxValue = JaxValue,
    GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[P, Any],
](
    ArithmeticGame[P, V, GameT, "_ScalarPowerGame[P, V, GameT]"],
):
    """Private dynamic-constant ``power`` adapter for general ``game ** c`` and ``c ** game``.

    Handles reflected, floating/complex, batched, and traced/dynamic exponents using
    a dynamic pytree ``constant`` leaf and JAX's uniform dynamic power path so
    metadata ``dtype`` and runtime dtypes agree across scalar, batched, sliced, and
    JIT evaluations. Does not infer a reduced k-additive bound.
    """

    constant: JaxValue
    reflected: bool = field(default=False, kw_only=True)
    protected_axes: ClassVar[dict[str, int]] = {"games": 0, "constant": 0}
    broadcasted_batch: ClassVar[set[str]] = {"constant"}

    def __post_init__(self) -> None:
        super(_ScalarPowerGame, self).__post_init__()
        if len(self.games.members) != 1:
            msg = "_ScalarPowerGame requires exactly one game."
            raise ValueError(msg)
        _validate_numeric_value(self.constant, "Power constant")
        self.validate()

    @override
    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Return exact dynamic JAX ``power`` dtype and ``weak_type`` without evaluating."""
        lhs, rhs = (
            (self.constant, self.games.members[0])
            if self.reflected
            else (self.games.members[0], self.constant)
        )
        return _power_spec(lhs, rhs)

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
        return jnp.asarray(other) if _is_numeric_dtype(other.dtype) else None
    if isinstance(other, JaxLike) and not isinstance(other, (Game, Players, BroadcastTuple)):
        return other if _is_numeric_dtype(other.dtype) else None
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
            weights=JaxBroadcastTuple((_unit_scalar(1, left), _unit_scalar(1, right))),
            bias=_unit_scalar(0, left),
        )
    if func is jf.jax_subtract:
        promoted = _promote_spec((left, right), left.dtype)
        return AffineGame(
            games=JaxBroadcastTuple((left, right)),
            weights=JaxBroadcastTuple((_unit_scalar(1, left), _unit_scalar(-1, promoted))),
            bias=_unit_scalar(0, promoted),
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
            weights=JaxBroadcastTuple((_unit_scalar(1, aligned),)),
            bias=constant,
        )
    if func is jf.jax_multiply:
        return AffineGame(
            games=JaxBroadcastTuple((aligned,)),
            weights=JaxBroadcastTuple((constant,)),
            bias=_unit_scalar(0, aligned),
        )
    if func is jf.jax_subtract:
        promoted = _promote_spec((aligned, constant), aligned.dtype)
        if reflected:
            return AffineGame(
                games=JaxBroadcastTuple((aligned,)),
                weights=JaxBroadcastTuple((_unit_scalar(-1, promoted),)),
                bias=constant,
            )
        neg_constant = jf.jax_multiply(_unit_scalar(-1, promoted), constant)
        return AffineGame(
            games=JaxBroadcastTuple((aligned,)),
            weights=JaxBroadcastTuple((_unit_scalar(1, aligned),)),
            bias=neg_constant,
        )
    if func is jf.jax_true_divide:
        if reflected:
            return _ScalarDivGame(
                games=JaxBroadcastTuple((aligned,)), constant=constant, reciprocal=True
            )
        compute_dtype = _true_divide_spec(aligned, constant).dtype
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
                weights=JaxBroadcastTuple((_unit_scalar(-1, game),)),
                bias=_unit_scalar(0, game),
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
