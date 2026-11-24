"""Lazy arithmetic wrappers and k-additive refinements for finite games."""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, override

import jax
import jax.numpy as jnp
import numpy as np
from flextype import CastUnavailable, autocast, castmethod, try_cast
from jaxtyping import Bool, Shaped, UInt8
from ndlike import jax_functions as jf
from ndlike.jax_like import JaxLike
from ndlike.protected_axis.jax import JaxAtMethod, JaxAxisProtected, JaxProtectedValue
from ndlike.tuple.jax import JaxBroadcastTuple

from shapiq.coalition import FiniteCoalition
from shapiq.players import FinitePlayers
from shapiq.players.finite import _TRUSTED_GATHER, _TRUSTED_SHAPE_FUNCS
from shapiq.typing import typechecked
from shapiq.value.value import JaxValue

from .finite import FinitePlayerGame, _FiniteGameOpResult
from .kadditive import AdditiveGame, KAdditiveGame
from .moebius import _moebius_coefficients_by_eval, _moebius_zero_like


def _zero_scalar() -> jax.Array:
    """Return a weakly typed zero-dimensional scalar array for default affine bias."""
    return jnp.asarray(0)


def _weak_zero_for_dtype(dtype: jnp.dtype) -> jax.Array:
    """Return a weakly typed zero scalar matching ``dtype``'s numeric kind."""
    if jnp.issubdtype(dtype, jnp.complexfloating):
        return jnp.asarray(0j)
    if jnp.issubdtype(dtype, jnp.floating):
        return jnp.asarray(0.0)
    return jnp.asarray(0)


def _is_numeric_dtype(dtype: object) -> bool:
    """Return whether ``dtype`` represents a numeric JAX/NumPy dtype."""
    try:
        return bool(jnp.issubdtype(cast("Any", dtype), jnp.number))
    except TypeError:
        return False


def _validate_numeric_value(value: object, name: str) -> None:
    """Validate that an arithmetic parameter carries a numeric dtype."""
    if not _is_numeric_dtype(getattr(value, "dtype", None)):
        msg = f"{name} must have a numeric dtype."
        raise TypeError(msg)


def _abstract_value_spec(value: object) -> jax.ShapeDtypeStruct | None:
    """Return lazy scalar ``ShapeDtypeStruct`` (dtype and ``weak_type``) without evaluating games."""
    if isinstance(value, jax.ShapeDtypeStruct):
        return jax.ShapeDtypeStruct((), value.dtype, weak_type=value.weak_type)
    if isinstance(value, jax.Array):
        return jax.ShapeDtypeStruct((), value.dtype, weak_type=value.weak_type)
    if isinstance(value, FinitePlayerGame):
        return value._value_spec()  # noqa: SLF001 - internal lazy abstract dtype hook
    if isinstance(value, JaxAxisProtected):
        rep = value._jax_protected_value()  # noqa: SLF001 - representative protected value leaf
        if rep is not value and getattr(rep, "dtype", None) == getattr(value, "dtype", None):
            return _abstract_value_spec(rep)
    if isinstance(value, (jnp.dtype, np.dtype, type)):
        try:
            return jax.ShapeDtypeStruct((), jnp.dtype(cast("Any", value)), weak_type=False)
        except TypeError:
            return None
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    try:
        return jax.ShapeDtypeStruct(
            (), jnp.dtype(cast("Any", dtype)), weak_type=bool(getattr(value, "weak_type", False))
        )
    except TypeError:
        return None


def _fallback_spec(fallback: object) -> jax.ShapeDtypeStruct:
    """Build a conservative fallback ``ShapeDtypeStruct``."""
    spec = _abstract_value_spec(fallback)
    if spec is not None:
        return spec
    return jax.ShapeDtypeStruct((), cast("jnp.dtype", fallback), weak_type=False)


def _promote_add(*xs: jax.Array) -> jax.Array:
    """Abstract addition chain used with ``jax.eval_shape`` to follow JAX weak-type promotion."""
    acc = xs[0]
    for x in xs[1:]:
        acc = jnp.add(acc, x)
    return acc


def _promote_spec(values: tuple[object, ...], fallback: object) -> jax.ShapeDtypeStruct:
    """Compute promoted JAX dtype and ``weak_type`` across operands without evaluating games."""
    specs: list[jax.ShapeDtypeStruct] = []
    for item in values:
        spec = _abstract_value_spec(item)
        if spec is None:
            return _fallback_spec(fallback)
        specs.append(spec)
    if not specs:
        return _fallback_spec(fallback)
    if len(specs) == 1:
        return specs[0]
    try:
        out = jax.eval_shape(_promote_add, *specs)
        return jax.ShapeDtypeStruct((), out.dtype, weak_type=out.weak_type)
    except TypeError:
        return _fallback_spec(fallback)


def _promote_dtype(values: tuple[object, ...], fallback: object) -> jnp.dtype:
    """Compute promoted JAX dtype across operands while preserving weak array/game dtypes."""
    return _promote_spec(values, fallback).dtype


def _match_promoted_dtype[T: JaxValue](
    value: Shaped[T, "..."], reference: object
) -> Shaped[T, "..."]:
    """Promote ``value`` to match the promoted dtype (and ``weak_type``) with ``reference``."""
    spec = _promote_spec((value, reference), getattr(value, "dtype", None))
    if isinstance(value, jax.Array):
        if value.dtype == spec.dtype and value.weak_type == spec.weak_type:
            return value
        if spec.weak_type:
            promoted = value + _weak_zero_for_dtype(spec.dtype)
            if promoted.dtype == spec.dtype and promoted.weak_type:
                return cast("Shaped[T, '...']", promoted)
        return cast("Shaped[T, '...']", value.astype(spec.dtype))
    if isinstance(value, JaxAxisProtected) and getattr(value, "dtype", None) != spec.dtype:
        return cast("Shaped[T, '...']", value.astype(spec.dtype))
    return value


def _ensure_shape[T: JaxValue](value: Shaped[T, "..."], shape: tuple[int, ...]) -> Shaped[T, "..."]:
    """Broadcast ``value`` to ``shape`` only when its logical shape differs."""
    return (
        value
        if value.shape == shape
        else cast("Shaped[T, '...']", jf.jax_broadcast_to(value, shape))
    )


def _scalar_zero_for_game(game: KAdditiveGame[Any, Any]) -> JaxValue:
    """Return a scalar-batch zero preserving ``game``'s structure, field dtypes, and weak type."""
    zero = game._zeros_like_value(())  # noqa: SLF001 - internal KAdditiveGame hook
    if isinstance(zero, jax.Array) and game._value_spec().weak_type:  # noqa: SLF001 - internal lazy abstract dtype hook
        return _weak_zero_for_dtype(zero.dtype)
    return zero


def _zeros_from_arithmetic(
    games: tuple[KAdditiveGame[Any, Any], ...],
    params: tuple[JaxValue, ...],
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> JaxValue:
    """Construct zero values lazily, preferring any structured parameter or child template."""
    structured: JaxValue | None = None
    structured_param_idx: int | None = None
    structured_game_idx: int | None = None
    for idx, param in enumerate(params):
        if not isinstance(param, jax.Array):
            structured = jf.jax_zeros_like(param, shape=shape)
            structured_param_idx = idx
            break
    first_zero: JaxValue | None = None
    if structured is None:
        for idx, game in enumerate(games):
            child_zero = game._zeros_like_value(shape)  # noqa: SLF001 - internal KAdditiveGame hook
            if not isinstance(child_zero, jax.Array):
                structured = child_zero
                structured_game_idx = idx
                break
            if first_zero is None:
                first_zero = child_zero
    if structured is not None:
        acc: JaxValue = structured
        if isinstance(acc, JaxAxisProtected) and jf.jax_add in type(acc).permitted_functions:
            for idx, param in enumerate(params):
                if idx == structured_param_idx:
                    continue
                acc = jf.jax_add(acc, jf.jax_zeros_like(param, shape=()))
            for idx, game in enumerate(games):
                if idx == structured_game_idx:
                    continue
                acc = jf.jax_add(acc, _scalar_zero_for_game(game))
        return _match_promoted_dtype(acc, dtype)
    return _match_promoted_dtype(
        jnp.zeros(shape, dtype=dtype) if first_zero is None else first_zero, dtype
    )


def _weighted_sum(
    weights: tuple[JaxValue, ...],
    terms: tuple[JaxValue, ...],
    *,
    expand_weight_axis: bool = False,
) -> JaxValue:
    """Compute ``sum(w_i * t_i)`` with logical-batch alignment."""
    w0 = jf.jax_expand_dims(weights[0], axis=-1) if expand_weight_axis else weights[0]
    result = jf.jax_multiply(w0, terms[0])
    for weight, term in zip(weights[1:], terms[1:], strict=True):
        w = jf.jax_expand_dims(weight, axis=-1) if expand_weight_axis else weight
        result = jf.jax_add(result, jf.jax_multiply(w, term))
    return result


def _require_refinable_children[T: FinitePlayerGame](
    game: "ArithmeticGame[Any, Any, Any, Any]",
    target: type[T],
) -> tuple[T, ...]:
    """Zero-cost cast every child game to ``target``."""
    converted: list[T] = []
    for member in game.games.members:
        if isinstance(member, target):
            converted.append(member)
            continue
        result = try_cast(member, target, max_cost=0.0)
        if result is None:
            msg = f"All component games of {type(game).__name__} must be {target.__name__}."
            raise CastUnavailable(msg)
        converted.append(result.value)
    return tuple(converted)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArithmeticGame[
    P,
    V: JaxValue = JaxValue,
    GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[P, Any],
    OwnerT: JaxLike[Any] = "ArithmeticGame[P, V, GameT, Any]",
](FinitePlayerGame[P, V], JaxAxisProtected[OwnerT, Any, _FiniteGameOpResult[P]], ABC):
    """Abstract lazy arithmetic game combining a nonempty tuple of finite games.

    ``games`` defines the logical batch shape via ``JaxBroadcastTuple`` and
    participates in ndlike shaping and JAX pytrees. Representation-level
    fieldwise arithmetic and indexed updates remain disabled.

    Subclasses implement ``_compute`` to combine the tuple of lazily evaluated
    child coalition values. ``GameT`` tracks the operand game type stored in
    ``games`` (broadly bounded by ``FinitePlayerGame[Any, Any]`` because PEP 695
    bounds cannot reference ``P``, while defaulting to ``FinitePlayerGame[P, Any]``).
    """

    games: JaxBroadcastTuple[tuple[GameT, ...]]
    protected_axes: ClassVar[dict[str, int]] = {"games": 0}
    permitted_at_methods: ClassVar[set[JaxAtMethod]] = set()

    def __post_init__(self) -> None:
        """Validate nonempty finite games and compatible player sets without evaluating them.

        Raises:
            TypeError: A member of ``games`` is not a ``FinitePlayerGame`` or
                distinct traced player IDs cannot be compared under JIT.
            ValueError: ``games`` is empty or its games have incompatible player sets.
        """
        if not self.games.members:
            msg = f"{type(self).__name__} requires at least one game."
            raise ValueError(msg)
        for game in self.games.members:
            if not isinstance(game, FinitePlayerGame):
                msg = f"{type(self).__name__} requires FinitePlayerGame members."
                raise TypeError(msg)
        batch_shape = self.games.shape
        first_players = self.games.members[0].players
        for game in self.games.members:
            if jnp.broadcast_shapes(game.players.shape, batch_shape) != batch_shape:
                msg = "Player-set batches must broadcast to the arithmetic game's batch shape."
                raise ValueError(msg)
        for game in self.games.members[1:]:
            if not first_players.is_compatible(game.players):
                msg = "ArithmeticGame requires compatible player sets."
                raise ValueError(msg)

    @override
    def _gather_protected_values(self, index: Any) -> Any:
        """Allow trusted reconstruction during read-only batch indexing on validated games."""
        token = _TRUSTED_GATHER.set(self)
        try:
            return super(ArithmeticGame, self)._gather_protected_values(index)
        finally:
            _TRUSTED_GATHER.reset(token)

    @override
    def with_protected_values(
        self,
        values: dict[str, JaxProtectedValue],
        func: Any = None,
    ) -> Any:
        """Skip value-dependent ID checks only for pure shape/gather ops on validated games."""
        if func in _TRUSTED_SHAPE_FUNCS or (func is None and _TRUSTED_GATHER.get() is self):
            obj = object.__new__(type(self))
            for field_info in fields(self):
                object.__setattr__(
                    obj,
                    field_info.name,
                    values.get(field_info.name, getattr(self, field_info.name)),
                )
            obj.validate()
            return obj
        return super(ArithmeticGame, self).with_protected_values(values, func)

    @property
    @override
    def players(self) -> FinitePlayers[P]:
        """Return the representative compatible player sets."""
        return self.games.members[0].players

    @override
    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Return promoted scalar dtype and ``weak_type`` across component games lazily."""
        return _promote_spec(self.games.members, super(ArithmeticGame, self).dtype)

    @property
    @override
    def dtype(self) -> jnp.dtype:
        """Return the promoted dtype across component games without evaluating them."""
        return self._value_spec().dtype

    @abstractmethod
    def _compute(self, values: tuple[JaxValue, ...]) -> Shaped[V, "..."]:
        """Combine lazily evaluated child coalition values into one batch of values."""

    def _eval_children(self, coalition: FiniteCoalition) -> Shaped[V, "..."]:
        """Evaluate every component game once and combine their results via ``_compute``."""
        return self._compute(tuple(game.eval(coalition) for game in self.games.members))

    @override
    def _eval_bool(self, coalition: Bool[jax.Array, "... {self.n_players}"]) -> Shaped[V, "..."]:
        """Forward boolean coalitions lazily to component games."""
        return self._eval_children(coalition)

    @override
    def _eval_packed(
        self, coalition: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        """Forward packed coalitions lazily to component games without unpacking."""
        return self._eval_children(coalition)


def _can_flatten_affine(outer_dtype: jnp.dtype, inner: "AffineGame[Any, Any, Any]") -> bool:
    """Return whether flattening ``inner`` into an ``outer_dtype`` affine game preserves semantics.

    Flattening is permitted when either:
    - ``inner`` and every inner ``(weight, game)`` term evaluate in an inexact
      (floating/complex) dtype, where no integer modular wraparound can be lost; or
    - ``inner``, every inner ``(weight, game)`` term, and the parent share the exact same
      integer dtype (same modular ring).

    Integer subexpressions whose dtype differs from the parent are preserved so widening
    cannot alter integer overflow or wraparound.
    """
    inner_dtype = inner.dtype
    if jnp.issubdtype(inner_dtype, jnp.inexact):
        return all(
            jnp.issubdtype(_promote_dtype((game, weight), inner_dtype), jnp.inexact)
            for game, weight in zip(inner.games.members, inner.weights.members, strict=True)
        )
    if jnp.issubdtype(inner_dtype, jnp.integer) and inner_dtype == outer_dtype:
        return all(
            _promote_dtype((game, weight), inner_dtype) == inner_dtype
            for game, weight in zip(inner.games.members, inner.weights.members, strict=True)
        )
    return False


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class AffineGame[
    P,
    V: JaxValue = JaxValue,
    GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[P, Any],
](
    ArithmeticGame[P, V, GameT, "AffineGame[P, V, GameT]"],
    metaclass=autocast(KAdditiveGame, max_cost=0.0),
):
    """A lazy affine combination ``sum(w_i * g_i(S)) + bias`` of compatible finite games.

    ``games`` defines the logical batch shape; ``weights`` and ``bias`` must
    broadcast into ``games.shape`` without enlarging it. Nested ``AffineGame``
    instances are normalized during construction by multiplying weights and
    accumulating scaled inner biases when all inner terms are inexact
    (floating/complex) or share the parent's exact integer ring; integer
    subexpressions that would widen under promotion are preserved intact.

    When all component games are zero-cost castable to ``AdditiveGame`` or
    ``KAdditiveGame``, construction automatically refines the instance via
    ``flextype.autocast``.
    """

    weights: JaxBroadcastTuple[tuple[JaxValue, ...]]
    bias: JaxValue = field(default_factory=_zero_scalar)
    protected_axes: ClassVar[dict[str, int]] = {"games": 0, "weights": 0, "bias": 0}
    broadcasted_batch: ClassVar[set[str]] = {"weights", "bias"}

    if TYPE_CHECKING:

        def __init__[P2, V2: JaxValue = JaxValue](
            self: "AffineGame[P2, V2, FinitePlayerGame[P2, Any]]",
            games: JaxBroadcastTuple[tuple[FinitePlayerGame[P2, Any], ...]],
            weights: JaxBroadcastTuple[tuple[JaxValue, ...]],
            bias: JaxValue = ...,
        ) -> None:
            """Construct an affine game with broadly typed stored operands."""

    def __post_init__(self) -> None:
        """Validate weights/bias shapes and normalize nested known affine wrappers.

        Raises:
            TypeError: Weights or bias do not carry numeric dtypes.
            ValueError: Weight count mismatches games or shapes fail to broadcast
                into ``games.shape``.
        """
        super(AffineGame, self).__post_init__()
        if len(self.weights.members) != len(self.games.members):
            msg = "AffineGame requires one weight per game."
            raise ValueError(msg)
        for weight in self.weights.members:
            _validate_numeric_value(weight, "AffineGame weights")
        _validate_numeric_value(self.bias, "AffineGame bias")
        self.validate()

        outer_spec = self._value_spec()
        outer_dtype = outer_spec.dtype
        if any(
            isinstance(game, AffineGame) and _can_flatten_affine(outer_dtype, game)
            for game in self.games.members
        ):
            flat_games: list[FinitePlayerGame[P, Any]] = []
            flat_weights: list[JaxValue] = []
            bias = _match_promoted_dtype(self.bias, outer_spec)
            for game, weight in zip(self.games.members, self.weights.members, strict=True):
                if isinstance(game, AffineGame) and _can_flatten_affine(outer_dtype, game):
                    scaled_weight = _match_promoted_dtype(weight, outer_spec)
                    flat_games.extend(game.games.members)
                    flat_weights.extend(
                        jf.jax_multiply(scaled_weight, inner_weight)
                        for inner_weight in game.weights.members
                    )
                    bias = jf.jax_add(bias, jf.jax_multiply(scaled_weight, game.bias))
                else:
                    flat_games.append(game)
                    flat_weights.append(weight)
            object.__setattr__(self, "games", JaxBroadcastTuple(tuple(flat_games)))
            object.__setattr__(self, "weights", JaxBroadcastTuple(tuple(flat_weights)))
            object.__setattr__(self, "bias", bias)

    @override
    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Return promoted scalar dtype and ``weak_type`` across games, weights, and bias."""
        return _promote_spec(
            (*self.games.members, *self.weights.members, self.bias),
            super(AffineGame, self)._value_spec(),
        )

    @property
    @override
    def dtype(self) -> jnp.dtype:
        """Return the promoted dtype across games, weights, and bias."""
        return self._value_spec().dtype

    @override
    def _compute(self, values: tuple[JaxValue, ...]) -> Shaped[V, "..."]:
        """Compute ``sum(weight * value) + bias`` with logical-batch alignment."""
        return cast(
            "Shaped[V, '...']",
            jf.jax_add(_weighted_sum(self.weights.members, values), self.bias),
        )

    @castmethod
    def _as_additive(self) -> AdditiveGame[P, V]:
        """Refine this affine game when every component is additive."""
        members = _require_refinable_children(self, AdditiveGame)
        return _AdditiveAffineGame(
            games=JaxBroadcastTuple(members), weights=self.weights, bias=self.bias
        )

    @castmethod
    def _as_kadditive(self) -> KAdditiveGame[P, V]:
        """Refine this affine game when every component is k-additive."""
        members = _require_refinable_children(self, KAdditiveGame)
        return _KAdditiveAffineGame(
            games=JaxBroadcastTuple(members), weights=self.weights, bias=self.bias
        )


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class _KAdditiveAffineGame[
    P,
    V: JaxValue = JaxValue,
    GameT: KAdditiveGame[Any, Any] = KAdditiveGame[P, Any],
](
    AffineGame[P, V, GameT],
    KAdditiveGame[P, V],
):
    """Private k-additive refinement of ``AffineGame`` with all k-additive components."""

    if TYPE_CHECKING:

        def __init__[P2, V2: JaxValue = JaxValue](
            self: "_KAdditiveAffineGame[P2, V2, KAdditiveGame[P2, Any]]",
            games: JaxBroadcastTuple[tuple[KAdditiveGame[P2, Any], ...]],
            weights: JaxBroadcastTuple[tuple[JaxValue, ...]],
            bias: JaxValue = ...,
        ) -> None: ...

    @property
    @override
    def additivity(self) -> int:
        """Return the maximum additivity bound across component games."""
        return max(game.additivity for game in self.games.members)

    @override
    def _zeros_like_value(self, shape: tuple[int, ...]) -> Shaped[V, "..."]:
        """Construct zero values preserving structured parameter/child representations lazily."""
        return cast(
            "Shaped[V, '...']",
            _zeros_from_arithmetic(
                self.games.members, (self.bias, *self.weights.members), shape, self.dtype
            ),
        )

    @override
    def empty_value(self) -> Shaped[V, "..."]:
        """Compute the affine combination of component empty values plus bias."""
        result = self._compute(tuple(game.empty_value() for game in self.games.members))
        return _ensure_shape(result, self.shape)

    def _linear_combine_last_axis(self, child_terms: tuple[JaxValue, ...]) -> Shaped[V, "..."]:
        """Combine child coefficient arrays along a shared trailing interaction axis."""
        combined = _weighted_sum(self.weights.members, child_terms, expand_weight_axis=True)
        result = _match_promoted_dtype(combined, self.bias)
        return _ensure_shape(
            cast("Shaped[V, '...']", result), (*self.shape, child_terms[0].shape[-1])
        )

    @override
    def _coefficients_by_order(self, order: int) -> Shaped[V, "..."]:
        """Combine order-``order`` Möbius coefficients linearly across component games."""
        n_combos = math.comb(self.n_players, order)
        if order > self.additivity or n_combos == 0:
            return self._zeros_like_value((*self.shape, n_combos))
        if order == 0:
            return cast("Shaped[V, '...']", jf.jax_expand_dims(self.empty_value(), axis=-1))
        terms = tuple(game.coefficients(order=order) for game in self.games.members)
        return self._linear_combine_last_axis(terms)

    @override
    def _coefficients_for_interactions(
        self,
        interactions: Bool[jax.Array, "#*batch {self.n_players}"],
        *,
        max_order: int | None = None,
    ) -> Shaped[V, "..."]:
        """Delegate explicit Möbius queries to child games and add ``bias`` on empty sets."""
        del max_order
        terms = tuple(game.coefficients(interactions) for game in self.games.members)
        result = _weighted_sum(self.weights.members, terms)
        is_empty = jnp.sum(interactions, axis=-1) == 0
        target_batch = jnp.broadcast_shapes(self.shape, interactions.shape[:-1])
        result_b = jf.jax_broadcast_to(result, target_batch)
        bias_b = jf.jax_broadcast_to(self.bias, target_batch)
        is_empty_b = jnp.broadcast_to(is_empty, target_batch)
        return cast(
            "Shaped[V, '...']",
            jf.jax_where(is_empty_b, jf.jax_add(result_b, bias_b), result_b),
        )


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class _AdditiveAffineGame[
    P,
    V: JaxValue = JaxValue,
    GameT: AdditiveGame[Any, Any] = AdditiveGame[P, Any],
](
    _KAdditiveAffineGame[P, V, GameT],
    AdditiveGame[P, V],
):
    """Private additive refinement of ``AffineGame`` sharing k-additive affine mechanics."""

    if TYPE_CHECKING:

        def __init__[P2, V2: JaxValue = JaxValue](
            self: "_AdditiveAffineGame[P2, V2, AdditiveGame[P2, Any]]",
            games: JaxBroadcastTuple[tuple[AdditiveGame[P2, Any], ...]],
            weights: JaxBroadcastTuple[tuple[JaxValue, ...]],
            bias: JaxValue = ...,
        ) -> None: ...

    @property
    @override
    def additivity(self) -> Literal[0, 1]:
        """Return ``0`` for zero-player games and the component maximum otherwise."""
        if self.n_players == 0:
            return 0
        return super(_AdditiveAffineGame, self).additivity  # ty: ignore[invalid-explicit-return]

    @override
    def contributions(self) -> Shaped[V, "..."]:
        """Combine component singleton contributions linearly without evaluating coalitions."""
        if self.n_players == 0 or self.additivity == 0:
            return self._zeros_like_value((*self.shape, self.n_players))
        terms = tuple(game.contributions() for game in self.games.members)
        return self._linear_combine_last_axis(terms)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class ProductGame[
    P,
    V: JaxValue = JaxValue,
    GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[P, Any],
](
    ArithmeticGame[P, V, GameT, "ProductGame[P, V, GameT]"],
    metaclass=autocast(KAdditiveGame, max_cost=0.0),
):
    """A lazy pointwise product ``prod_i g_i(S)`` of compatible finite games.

    Nesting is preserved to maintain exact multiplication order and floating-point
    behavior. When every component game is zero-cost castable to ``KAdditiveGame``,
    construction automatically refines the instance.
    """

    if TYPE_CHECKING:

        def __init__[P2, V2: JaxValue = JaxValue](
            self: "ProductGame[P2, V2, FinitePlayerGame[P2, Any]]",
            games: JaxBroadcastTuple[tuple[FinitePlayerGame[P2, Any], ...]],
        ) -> None:
            """Construct a product game with broadly typed stored operands."""

    @override
    def _compute(self, values: tuple[JaxValue, ...]) -> Shaped[V, "..."]:
        """Multiply component coalition values pointwise with logical-batch alignment."""
        result = values[0]
        for value in values[1:]:
            result = jf.jax_multiply(result, value)
        return cast("Shaped[V, '...']", result)

    @castmethod
    def _as_kadditive(self) -> KAdditiveGame[P, V]:
        """Refine this product when every component is k-additive."""
        members = _require_refinable_children(self, KAdditiveGame)
        return _KAdditiveProductGame(games=JaxBroadcastTuple(members))


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class _KAdditiveProductGame[
    P,
    V: JaxValue = JaxValue,
    GameT: KAdditiveGame[Any, Any] = KAdditiveGame[P, Any],
](
    ProductGame[P, V, GameT],
    KAdditiveGame[P, V],
):
    """Private k-additive refinement of ``ProductGame`` with capped sum of bounds."""

    @property
    @override
    def additivity(self) -> int:
        """Return ``min(n_players, sum(g.additivity))`` without evaluating coalitions."""
        total = sum(game.additivity for game in self.games.members)
        return min(self.n_players, total)

    @override
    def _zeros_like_value(self, shape: tuple[int, ...]) -> Shaped[V, "..."]:
        """Construct zero values lazily from component games without evaluating coalitions."""
        return cast(
            "Shaped[V, '...']", _zeros_from_arithmetic(self.games.members, (), shape, self.dtype)
        )

    @override
    def _coefficients_by_order(self, order: int) -> Shaped[V, "..."]:
        """Match Möbius-inversion dtypes on zero-shortcut order queries."""
        n_combos = math.comb(self.n_players, order)
        if order > self.additivity or n_combos == 0:
            return _moebius_zero_like(self, (*self.shape, n_combos))
        return super(_KAdditiveProductGame, self)._coefficients_by_order(order)

    @override
    def _coefficients_for_interactions(
        self,
        interactions: Bool[jax.Array, "#*batch {self.n_players}"],
        *,
        max_order: int | None = None,
    ) -> Shaped[V, "..."]:
        """Compute product Möbius coefficients explicitly via subset inversion."""
        return _moebius_coefficients_by_eval(self, interactions, max_order=max_order)


def _infer_mapped_spec(
    fn: Callable[[tuple[JaxValue, ...]], Any],
    games: tuple[FinitePlayerGame[Any, Any], ...],
    batch_shape: tuple[int, ...],
) -> jax.ShapeDtypeStruct:
    """Lazily infer ``fn``'s numeric output dtype and ``weak_type`` via ``jax.eval_shape``."""
    specs = tuple(
        jax.ShapeDtypeStruct(batch_shape, spec.dtype, weak_type=spec.weak_type)
        for spec in (game._value_spec() for game in games)  # noqa: SLF001 - internal lazy abstract dtype hook
    )
    try:
        abstract_out = jax.eval_shape(cast("Any", fn), specs)
    except Exception as error:
        msg = (
            "Cannot infer MappedGame output dtype lazily from abstract array specs; "
            "specify output_dtype explicitly."
        )
        raise TypeError(msg) from error
    out_dtype = getattr(abstract_out, "dtype", None)
    if out_dtype is None or not _is_numeric_dtype(out_dtype):
        msg = "MappedGame output dtype must be numeric."
        raise TypeError(msg)
    out_spec = _abstract_value_spec(abstract_out)
    weak = False if out_spec is None else out_spec.weak_type
    return jax.ShapeDtypeStruct((), jnp.dtype(out_dtype), weak_type=weak)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class MappedGame[
    P,
    V: JaxValue = JaxValue,
    GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[P, Any],
](ArithmeticGame[P, V, GameT, "MappedGame[P, V, GameT]"]):
    """A lazy finite game applying a pure batch-local callable to child coalition values.

    Args:
        games: Nonempty ``JaxBroadcastTuple`` of compatible finite games.
        fn: Pure batch-local callable receiving a tuple of broadcast-aligned
            child coalition value batches (one per game in ``games``) and
            returning a numeric value batch with the same logical batch shape.
            Stored as static pytree metadata.
        output_dtype: Optional explicit numeric output dtype metadata. When
            omitted, inferred lazily via ``jax.eval_shape`` on abstract child
            array specs without evaluating any child game; specify explicitly
            when ``fn`` expects structured ``JaxLike`` inputs that cannot be
            abstractly traced from plain array specs.
    """

    fn: Callable[[tuple[JaxValue, ...]], Shaped[V, "..."]]
    output_dtype: Any = field(default=None, kw_only=True)
    _output_weak_type: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate component games and resolve/validate the numeric output dtype lazily."""
        super(MappedGame, self).__post_init__()
        if self.output_dtype is None:
            resolved_spec = _infer_mapped_spec(self.fn, self.games.members, self.games.shape)
            resolved = resolved_spec.dtype
            weak_type = resolved_spec.weak_type
        else:
            try:
                resolved = jnp.dtype(self.output_dtype)
            except TypeError as error:
                msg = "MappedGame output_dtype must be a valid numeric dtype."
                raise TypeError(msg) from error
            if not _is_numeric_dtype(resolved):
                msg = "MappedGame output_dtype must be a numeric dtype."
                raise TypeError(msg)
            weak_type = False
        object.__setattr__(self, "output_dtype", resolved)
        object.__setattr__(self, "_output_weak_type", weak_type)

    @override
    def with_protected_values(
        self,
        values: dict[str, JaxProtectedValue],
        func: Any = None,
    ) -> Any:
        """Preserve inferred ``_output_weak_type`` across non-shape reconstructions such as stack."""
        result = super(MappedGame, self).with_protected_values(values, func)
        if isinstance(result, MappedGame) and result.output_dtype == self.output_dtype:
            object.__setattr__(result, "_output_weak_type", self._output_weak_type)
        return result

    @override
    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Return the validated numeric output dtype and inferred ``weak_type`` lazily."""
        return jax.ShapeDtypeStruct(
            (), cast("jnp.dtype", self.output_dtype), weak_type=self._output_weak_type
        )

    @property
    @override
    def dtype(self) -> jnp.dtype:
        """Return the validated numeric output dtype without evaluating child games."""
        return cast("jnp.dtype", self.output_dtype)

    @override
    def _compute(self, values: tuple[JaxValue, ...]) -> Shaped[V, "..."]:
        """Align child value batches, apply ``fn``, and verify the declared output dtype."""
        if len(values) > 1:
            target_shape = jnp.broadcast_shapes(*(value.shape for value in values))
            values = tuple(_ensure_shape(value, target_shape) for value in values)
        result = self.fn(values)
        result_dtype = getattr(result, "dtype", None)
        if result_dtype is None or jnp.dtype(cast("Any", result_dtype)) != self.output_dtype:
            msg = f"MappedGame fn returned dtype {result_dtype}, expected {self.output_dtype}."
            raise TypeError(msg)
        return result
