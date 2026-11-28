"""Lazy arithmetic wrappers for finite games."""

import math
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, cast, override

import jax
import jax.numpy as jnp
from flextype import CastUnavailable, castmethod, try_cast
from jaxtyping import Bool, Shaped, UInt8
from ndlike import jax_functions as jf
from ndlike.protected_axis.jax import JaxAtMethod, JaxAxisProtected
from ndlike.tuple.jax import JaxBroadcastTuple

from shapiq.coalition import FiniteCoalition
from shapiq.players import FinitePlayers
from shapiq.typing import typechecked
from shapiq.value._dtype import (
    match_promoted_dtype,
    op_spec,
    promote_dtype,
    promote_spec,
    require_value_spec,
    validate_numeric_dtype,
    validate_numeric_value,
    weak_zero,
    weak_zero_for_dtype,
)
from shapiq.value.value import JaxValue

from .dense import DenseCoalitionGame
from .finite import Basis, CoalitionDefinedGame, FinitePlayerGame, _FiniteGame


def _ensure_shape[T: JaxValue](value: Shaped[T, "..."], shape: tuple[int, ...]) -> Shaped[T, "..."]:
    """Broadcast ``value`` to ``shape`` only when its logical shape differs."""
    return (
        value
        if value.shape == shape
        else cast("Shaped[T, '...']", jf.jax_broadcast_to(value, shape))
    )


def _packed_is_empty(
    packed: UInt8[jax.Array, "... bytes"],
    n_players: int,
) -> Bool[jax.Array, "..."]:
    """Return whether each packed coalition is empty, ignoring unused padding bits."""
    if n_players == 0:
        return jnp.ones(packed.shape[:-1], dtype=jnp.bool_)
    rem = n_players % 8
    if rem != 0:
        mask = jnp.uint8((1 << rem) - 1)
        packed = packed.at[..., -1].set(packed[..., -1] & mask)
    return jnp.all(packed == 0, axis=-1)


def _scalar_zero_for_game(game: FinitePlayerGame[Any, Any]) -> JaxValue:
    """Return a scalar-batch zero preserving ``game``'s structure, field dtypes, and weak type."""
    zero = game._zeros_like_value(())  # noqa: SLF001 - internal lazy zero hook
    if isinstance(zero, jax.Array) and game._value_spec().weak_type:  # noqa: SLF001 - internal lazy abstract dtype hook
        return weak_zero_for_dtype(zero.dtype)
    return zero


def _zeros_from_arithmetic(
    games: tuple[FinitePlayerGame[Any, Any], ...],
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
            child_zero = game._zeros_like_value(shape)  # noqa: SLF001 - internal lazy zero hook
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
        return match_promoted_dtype(acc, dtype)
    return match_promoted_dtype(
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


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArithmeticGame[
    P,
    V: JaxValue = JaxValue,
    GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[P, Any],
    OwnerT: "ArithmeticGame[Any, Any, Any, Any]" = "ArithmeticGame[P, V, GameT, Any]",
](
    CoalitionDefinedGame[P, V],
    JaxAxisProtected[OwnerT, Any, _FiniteGame[P]],
):
    """Abstract lazy arithmetic game combining a nonempty tuple of finite games.

    ``games`` defines the logical batch shape via ``JaxBroadcastTuple`` and
    participates in ndlike shaping and JAX pytrees. Representation-level
    fieldwise arithmetic and indexed updates remain disabled.

    Subclasses implement ``_compute`` to combine the tuple of lazily evaluated
    child coalition values (and fast ``DenseCoalitionGame`` casting delegates
    directly to ``_compute``).
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

    @property
    @override
    def players(self) -> FinitePlayers[P]:
        """Return the representative compatible player sets."""
        return self.games.members[0].players

    @override
    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Return promoted scalar dtype and ``weak_type`` across component games lazily."""
        return promote_spec(self.games.members, self.games.members[0])

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
    def _coalition_bool(
        self, coalition: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        """Forward boolean coalitions lazily to component games."""
        return self._eval_children(coalition)

    @override
    def _coalition_packed(
        self, coalition: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        """Forward packed coalitions lazily to component games without unpacking."""
        return self._eval_children(coalition)

    @castmethod(DenseCoalitionGame, cost=1.0)
    def _as_dense_coalition(self) -> DenseCoalitionGame[P]:
        """Materialize directly on raw dense tables when every leaf is a ``DenseCoalitionGame``."""
        n_coalitions = 2**self.n_players
        child_tables: list[jax.Array] = []
        for member in self.games.members:
            if isinstance(member, DenseCoalitionGame):
                table = member
            else:
                result = try_cast(member, DenseCoalitionGame, max_cost=1.0)
                if result is None:
                    msg = "Fast DenseCoalitionGame cast requires DenseCoalitionGame leaves."
                    raise CastUnavailable(msg)
                table = result.value
            # Align trailing batch rank with self.ndim while keeping the 2**n axis at axis 0.
            moved = jnp.moveaxis(table.values, -1, 0)
            pad = self.ndim - table.ndim
            if pad > 0:
                moved = moved.reshape((n_coalitions, *(1,) * pad, *table.shape))
            child_tables.append(moved)
        computed = self._compute(tuple(child_tables))
        if not isinstance(computed, jax.Array) or not jnp.issubdtype(computed.dtype, jnp.floating):
            msg = "DenseCoalitionGame requires floating jax.Array values."
            raise CastUnavailable(msg)
        values = jnp.moveaxis(computed, 0, -1)
        expected_shape = (*self.shape, n_coalitions)
        if values.shape != expected_shape:
            values = jnp.broadcast_to(values, expected_shape)
        return DenseCoalitionGame(values, self.players)


def _can_flatten_affine(outer_dtype: jnp.dtype, inner: "AffineGame[Any, Any, Any]") -> bool:
    """Return whether flattening ``inner`` into an ``outer_dtype`` affine game preserves semantics."""
    inner_dtype = inner.dtype
    if jnp.issubdtype(inner_dtype, jnp.inexact):
        return all(
            jnp.issubdtype(promote_dtype((game, weight), inner_dtype), jnp.inexact)
            for game, weight in zip(inner.games.members, inner.weights.members, strict=True)
        )
    if jnp.issubdtype(inner_dtype, jnp.integer) and inner_dtype == outer_dtype:
        return all(
            promote_dtype((game, weight), inner_dtype) == inner_dtype
            for game, weight in zip(inner.games.members, inner.weights.members, strict=True)
        )
    return False


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class AffineGame[
    P,
    V: JaxValue = JaxValue,
    GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[P, Any],
](ArithmeticGame[P, V, GameT, "AffineGame[P, V, GameT]"]):
    """A lazy affine combination ``sum(w_i * g_i(S)) + bias`` of compatible finite games.

    ``games`` defines the logical batch shape; ``weights`` and ``bias`` must
    broadcast into ``games.shape`` without enlarging it. Nested ``AffineGame``
    instances are normalized during construction when safe for dtype semantics.

    Möbius and Fourier queries forward linearly to component games (adding
    ``bias`` on the empty interaction), and ``additivity()`` propagates known
    child bounds in ``O(1)``.
    """

    weights: JaxBroadcastTuple[tuple[JaxValue, ...]]
    bias: JaxValue = field(default_factory=weak_zero)
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
            validate_numeric_value(weight, "AffineGame weights")
        validate_numeric_value(self.bias, "AffineGame bias")
        self.validate()

        outer_spec = self._value_spec()
        outer_dtype = outer_spec.dtype
        if any(
            isinstance(game, AffineGame) and _can_flatten_affine(outer_dtype, game)
            for game in self.games.members
        ):
            flat_games: list[FinitePlayerGame[P, Any]] = []
            flat_weights: list[JaxValue] = []
            bias = match_promoted_dtype(self.bias, outer_spec)
            for game, weight in zip(self.games.members, self.weights.members, strict=True):
                if isinstance(game, AffineGame) and _can_flatten_affine(outer_dtype, game):
                    scaled_weight = match_promoted_dtype(weight, outer_spec)
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
        return promote_spec(
            (*self.games.members, *self.weights.members, self.bias),
            super(AffineGame, self)._value_spec(),
        )

    @property
    @override
    def dtype(self) -> jnp.dtype:
        """Return the promoted dtype across games, weights, and bias."""
        return self._value_spec().dtype

    @override
    def _known_additivity(self) -> int | None:
        """Return the maximum known child additivity bound if every child has one."""
        if self.n_players == 0:
            return 0
        bounds: list[int] = []
        for game in self.games.members:
            bound = game._known_additivity()  # noqa: SLF001 - internal O(1) bound check
            if bound is None:
                return None
            bounds.append(bound)
        return max(bounds)

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

    @override
    def _compute(self, values: tuple[JaxValue, ...]) -> Shaped[V, "..."]:
        """Compute ``sum(weight * value) + bias`` with logical-batch alignment."""
        return cast(
            "Shaped[V, '...']",
            jf.jax_add(_weighted_sum(self.weights.members, values), self.bias),
        )

    def _linear_combine_last_axis(self, child_terms: tuple[JaxValue, ...]) -> Shaped[V, "..."]:
        """Combine child coefficient arrays along a shared trailing interaction axis."""
        combined = _weighted_sum(self.weights.members, child_terms, expand_weight_axis=True)
        zero_bias = jf.jax_zeros_like(self.bias, shape=(*self.bias.shape, 1))
        result = jf.jax_add(combined, zero_bias)
        return _ensure_shape(
            cast("Shaped[V, '...']", result), (*self.shape, child_terms[0].shape[-1])
        )

    def _linear_basis_query(
        self,
        interaction: FiniteCoalition,
        basis: Basis,
        is_empty: Bool[jax.Array, "..."],
    ) -> Shaped[V, "..."]:
        """Forward a Möbius or Fourier query linearly to children and add ``bias`` on empty sets."""
        terms = tuple(game.coefficients(interaction, basis=basis) for game in self.games.members)
        result = _weighted_sum(self.weights.members, terms)
        target_batch = jnp.broadcast_shapes(self.shape, interaction.shape[:-1])
        result_b = jf.jax_broadcast_to(result, target_batch)
        bias_b = jf.jax_broadcast_to(self.bias, target_batch)
        is_empty_b = jnp.broadcast_to(is_empty, target_batch)
        return cast(
            "Shaped[V, '...']",
            jf.jax_where(is_empty_b, jf.jax_add(result_b, bias_b), result_b),
        )

    @override
    def _moebius_bool(
        self, interaction: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        """Forward boolean Möbius queries linearly to component games."""
        return self._linear_basis_query(
            interaction, Basis.MOEBIUS, jnp.sum(interaction, axis=-1) == 0
        )

    @override
    def _moebius_packed(
        self, interaction: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        """Forward packed Möbius queries linearly to component games without unpacking."""
        return self._linear_basis_query(
            interaction, Basis.MOEBIUS, _packed_is_empty(interaction, self.n_players)
        )

    @override
    def _fourier_bool(
        self, interaction: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        """Forward boolean Fourier queries linearly to component games."""
        return self._linear_basis_query(
            interaction, Basis.FOURIER, jnp.sum(interaction, axis=-1) == 0
        )

    @override
    def _fourier_packed(
        self, interaction: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        """Forward packed Fourier queries linearly to component games without unpacking."""
        return self._linear_basis_query(
            interaction, Basis.FOURIER, _packed_is_empty(interaction, self.n_players)
        )

    @override
    def _coefficients_by_order(self, order: int, basis: Basis) -> Shaped[V, "..."]:
        """Combine order-``order`` Möbius or Fourier coefficients linearly across component games."""
        if basis in (Basis.MOEBIUS, Basis.FOURIER):
            n_combos = math.comb(self.n_players, order)
            known = self._known_additivity()
            if n_combos == 0 or (known is not None and order > known):
                return self._zeros_like_value((*self.shape, n_combos))
            if order == 0:
                if basis is Basis.MOEBIUS:
                    return cast("Shaped[V, '...']", jf.jax_expand_dims(self.empty_value(), axis=-1))
                terms = tuple(game.fourier_coefficients(order=0) for game in self.games.members)
                combined = self._linear_combine_last_axis(terms)
                return cast(
                    "Shaped[V, '...']",
                    jf.jax_add(combined, jf.jax_expand_dims(self.bias, axis=-1)),
                )
            terms = tuple(
                game.coefficients(order=order, basis=basis) for game in self.games.members
            )
            return self._linear_combine_last_axis(terms)
        return super(AffineGame, self)._coefficients_by_order(order, basis)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class ProductGame[
    P,
    V: JaxValue = JaxValue,
    GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[P, Any],
](ArithmeticGame[P, V, GameT, "ProductGame[P, V, GameT]"]):
    """A lazy pointwise product ``prod_i g_i(S)`` of compatible finite games.

    Nesting is preserved to maintain exact multiplication order and floating-point
    behavior. When every component game has a known additivity bound, ``additivity()``
    returns ``min(n_players, sum(g.additivity()))`` in ``O(1)``.
    """

    if TYPE_CHECKING:

        def __init__[P2, V2: JaxValue = JaxValue](
            self: "ProductGame[P2, V2, FinitePlayerGame[P2, Any]]",
            games: JaxBroadcastTuple[tuple[FinitePlayerGame[P2, Any], ...]],
        ) -> None:
            """Construct a product game with broadly typed stored operands."""

    @override
    def _known_additivity(self) -> int | None:
        """Return ``min(n_players, sum(g._known_additivity()))`` when all children are known."""
        if self.n_players == 0:
            return 0
        total = 0
        for game in self.games.members:
            bound = game._known_additivity()  # noqa: SLF001 - internal O(1) bound check
            if bound is None:
                return None
            total += bound
        return min(self.n_players, total)

    @override
    def _zeros_like_value(self, shape: tuple[int, ...]) -> Shaped[V, "..."]:
        """Construct zero values lazily from component games without evaluating coalitions."""
        return cast(
            "Shaped[V, '...']", _zeros_from_arithmetic(self.games.members, (), shape, self.dtype)
        )

    @override
    def _compute(self, values: tuple[JaxValue, ...]) -> Shaped[V, "..."]:
        """Multiply component coalition values pointwise with logical-batch alignment."""
        result = values[0]
        for value in values[1:]:
            result = jf.jax_multiply(result, value)
        return cast("Shaped[V, '...']", result)


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
        spec = op_spec(fn, specs)
    except Exception as error:
        msg = (
            "Cannot infer MappedGame output dtype lazily from abstract array specs; "
            "specify output_dtype explicitly."
        )
        raise TypeError(msg) from error
    validate_numeric_dtype(spec.dtype, "MappedGame output dtype")
    return spec


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class MappedGame[
    P,
    V: JaxValue = JaxValue,
    GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[P, Any],
](ArithmeticGame[P, V, GameT, "MappedGame[P, V, GameT]"]):
    """A lazy finite game applying a pure batch-local callable to child coalition values."""

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
            resolved = validate_numeric_dtype(self.output_dtype, "MappedGame output_dtype")
            weak_type = False
        object.__setattr__(self, "output_dtype", resolved)
        object.__setattr__(self, "_output_weak_type", weak_type)

    @classmethod
    @override
    def _combine_sequence_metadata(
        cls,
        items: Sequence["MappedGame[Any, Any, Any]"],
        *,
        func: Callable[..., Any] | None,
    ) -> dict[str, Any]:
        """Combine ``_output_weak_type`` conservatively across stacked/concatenated mapped games."""
        del func
        return {"_output_weak_type": all(item._output_weak_type for item in items)}  # noqa: SLF001 - own private metadata field

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
        if not self._output_weak_type and require_value_spec(result).weak_type:
            return cast("Shaped[V, '...']", result.astype(self.output_dtype, copy=False))
        return result
