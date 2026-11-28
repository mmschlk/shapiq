"""Lazy reduction wrappers for finite games."""

import inspect
import math
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, overload, override

import jax
import jax.numpy as jnp
import numpy as np
from flextype import CastUnavailable, castmethod, try_cast
from jax.core import Tracer
from jaxtyping import Bool, Shaped, UInt8
from ndlike import jax_functions as jf
from ndlike.protected_axis.jax import JaxAtMethod, JaxAxisProtected

from shapiq.coalition import FiniteCoalition
from shapiq.players import FinitePlayers
from shapiq.typing import typechecked
from shapiq.value import Value
from shapiq.value._dtype import op_spec, validate_numeric_dtype, validate_numeric_value

from ._transforms import _scale_by_pow2
from .arithmetic import _ensure_shape
from .base import Game
from .dense import DenseCoalitionGame
from .finite import (
    Basis,
    CoalitionDefinedGame,
    FinitePlayerGame,
    MoebiusDefinedGame,
    _FiniteGame,
)

if TYPE_CHECKING:
    from jax.typing import DTypeLike
    from ndlike.jax_like import JaxLikeImplementation


def _normalize_reduction_axes(axis: object, ndim: int) -> tuple[int, ...]:
    """Normalize and validate reduction ``axis`` into sorted unique axes in ``0..ndim-1``."""
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, (bool, np.bool_)):
        msg = "Reduction axis must be None, an int, or a sequence of ints."
        raise TypeError(msg)
    if isinstance(axis, (int, np.integer)):
        raw_axes: tuple[int, ...] = (int(axis),)
    elif isinstance(axis, Sequence) and not isinstance(axis, (str, bytes, bytearray)):
        parsed: list[int] = []
        for item in axis:
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                msg = "Reduction axis must be None, an int, or a sequence of ints."
                raise TypeError(msg)
            parsed.append(int(item))
        raw_axes = tuple(parsed)
    else:
        msg = "Reduction axis must be None, an int, or a sequence of ints."
        raise TypeError(msg)

    normalized: list[int] = []
    seen: set[int] = set()
    for raw in raw_axes:
        norm = raw + ndim if raw < 0 else raw
        if norm < 0 or norm >= ndim:
            msg = f"axis {raw} is out of bounds for game of dimension {ndim}."
            raise ValueError(msg)
        if norm in seen:
            msg = f"duplicate value in 'axis': {axis!r}."
            raise ValueError(msg)
        seen.add(norm)
        normalized.append(norm)
    return tuple(sorted(normalized))


def _validate_and_reduce_players[P](
    players: FinitePlayers[P],
    game_shape: tuple[int, ...],
    reduced_axes: tuple[int, ...],
    out_shape: tuple[int, ...],
    *,
    keepdims: bool,
) -> FinitePlayers[P]:
    """Validate player-set compatibility along ``reduced_axes`` and return surviving players."""
    if jnp.broadcast_shapes(players.shape, game_shape) != game_shape:
        msg = "Player-set batch must broadcast to the game's batch shape."
        raise ValueError(msg)

    offset = len(game_shape) - players.ndim
    reduced_set = set(reduced_axes)
    p_reduced_axes = tuple(pk for pk in range(players.ndim) if (offset + pk) in reduced_set)
    if not p_reduced_axes:
        return players

    p_reduced_set = set(p_reduced_axes)
    p_shape = players.shape

    # Size-1 reduced axes are shared by broadcasting: compatible by construction (JIT-safe).
    if all(p_shape[pk] == 1 for pk in p_reduced_axes):
        if keepdims:
            return players
        return players.squeeze(axis=p_reduced_axes)

    # An empty surviving batch has no reduction groups in which player IDs could disagree.
    if 0 in out_shape:
        surviving_p_shape = tuple(
            1 if pk in p_reduced_set else p_shape[pk]
            for pk in range(players.ndim)
            if keepdims or pk not in p_reduced_set
        )
        target_p_shape = surviving_p_shape if 0 in surviving_p_shape else out_shape
        empty_ids = jnp.zeros((*target_p_shape, players.n_players), dtype=players.ids.dtype)
        return FinitePlayers(
            empty_ids,
            players=players.players,
            latent_n_players=players.latent_n_players,
            sorted=True,
        )

    # Zero-length reduced player axes carry no IDs to select a representative from.
    if any(p_shape[pk] == 0 for pk in p_reduced_axes):
        surviving_p_shape = tuple(
            1 if pk in p_reduced_set else p_shape[pk]
            for pk in range(players.ndim)
            if keepdims or pk not in p_reduced_set
        )
        if players.n_players == 0 or 0 in surviving_p_shape:
            empty_ids = jnp.zeros((*surviving_p_shape, players.n_players), dtype=players.ids.dtype)
            return FinitePlayers(
                empty_ids,
                players=players.players,
                latent_n_players=players.latent_n_players,
                sorted=True,
            )
        msg = "Cannot infer surviving player sets when reducing a zero-length player-set axis."
        raise ValueError(msg)

    # Non-broadcast, nonempty reduced axes: all player sets in each group must agree.
    if players.n_players > 0 and players.size > 0:
        if isinstance(players.ids, Tracer):
            msg = "Cannot validate player compatibility across traced player IDs under JIT."
            raise TypeError(msg)
        ref_slice = tuple(
            slice(0, 1) if pk in p_reduced_set else slice(None) for pk in range(players.ndim)
        )
        with jax.ensure_compile_time_eval():
            ref_ids = players.ids[(*ref_slice, slice(None))]
            compatible = bool(jnp.all(players.ids == ref_ids))
        if not compatible:
            msg = "Reduction requires compatible player sets along reduced axes."
            raise ValueError(msg)

    select_index = tuple(
        (slice(0, 1) if keepdims else 0) if pk in p_reduced_set else slice(None)
        for pk in range(players.ndim)
    )
    return players[select_index]


def _rearrange_game_for_reduction[GameT: FinitePlayerGame[Any, Any]](
    game: GameT,
    reduced_axes: tuple[int, ...],
    out_shape: tuple[int, ...],
) -> GameT:
    """Move ``reduced_axes`` to the end and flatten them into one trailing axis."""
    if not reduced_axes:
        return jf.jax_expand_dims(game, axis=-1)
    reduced_set = set(reduced_axes)
    surviving_axes = tuple(i for i in range(game.ndim) if i not in reduced_set)
    perm = (*surviving_axes, *reduced_axes)
    rearranged: FinitePlayerGame[Any, Any] = game
    if perm != tuple(range(game.ndim)):
        rearranged = jf.jax_transpose(rearranged, axes=perm)
    n_reduced = math.prod(game.shape[i] for i in reduced_axes)
    target_shape = (*out_shape, n_reduced)
    if rearranged.shape != target_shape:
        rearranged = jf.jax_reshape(rearranged, target_shape)
    return cast("GameT", rearranged)


def _sum_spec(operand: object, output_dtype: jnp.dtype | None) -> jax.ShapeDtypeStruct:
    """Compute JAX's exact scalar result spec for summing a batch lazily."""
    return op_spec(lambda x: jnp.sum(x, dtype=output_dtype), operand)


def _child_value_leaf_specs(child: FinitePlayerGame[Any, Any]) -> tuple[jax.ShapeDtypeStruct, ...]:
    """Inspect every leaf dtype and ``weak_type`` of ``child``'s values without calling ``eval``."""
    if type(child)._zeros_like_value is not FinitePlayerGame._zeros_like_value:  # noqa: SLF001 - internal lazy zero hook
        template: object = child._zeros_like_value(())  # noqa: SLF001 - internal lazy zero hook
    elif type(child).empty_value not in (
        FinitePlayerGame.empty_value,
        MoebiusDefinedGame.empty_value,
    ):
        try:
            template = jax.eval_shape(child.empty_value)
        except Exception:  # noqa: BLE001 - eager-only games may fail abstract tracing
            return ()
    elif type(child)._value_spec is not Game._value_spec:  # noqa: SLF001 - internal lazy spec hook
        return (child._value_spec(),)  # noqa: SLF001 - internal lazy spec hook
    else:
        return ()
    specs: list[jax.ShapeDtypeStruct] = []
    for leaf in jax.tree.leaves(template):
        if isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct)):
            specs.append(jax.ShapeDtypeStruct((), leaf.dtype, weak_type=leaf.weak_type))
        else:
            return ()
    return tuple(specs)


def _preserves_child_dtypes_for_sum(
    child: FinitePlayerGame[Any, Any],
    output_dtype: jnp.dtype | None,
) -> bool:
    """Return whether summing ``child`` with ``output_dtype`` keeps every leaf dtype unchanged."""
    specs = _child_value_leaf_specs(child)
    if not specs:
        return False
    for spec in specs:
        summed = _sum_spec(spec, output_dtype)
        if summed.dtype != spec.dtype:
            return False
    return True


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class ReductionGame[
    P,
    V: Value = Value,
    GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[Any, Any],
    OwnerT: "ReductionGame[Any, Any, Any, Any]" = "ReductionGame[Any, Any, Any, Any]",
](
    CoalitionDefinedGame[P, V],
    JaxAxisProtected[OwnerT, Any, _FiniteGame[P]],
):
    """Abstract lazy reduction game protecting one flattened trailing reduction axis."""

    game: GameT
    axis: InitVar[int | Sequence[int] | None] = field(default=-1, kw_only=True)
    keepdims: InitVar[bool] = field(default=False, kw_only=True)
    _players: FinitePlayers[P] | None = field(default=None, repr=False, kw_only=True)
    protected_axes: ClassVar[dict[str, int]] = {"game": 1, "_players": 0}
    broadcasted_batch: ClassVar[set[str]] = {"_players"}
    permitted_at_methods: ClassVar[set[JaxAtMethod]] = set()

    def __post_init__(
        self,
        axis: int | Sequence[int] | None,
        keepdims: bool,
    ) -> None:
        """Validate ``game`` and players along ``axis``, then flatten reduced axes to axis ``-1``."""
        if not isinstance(self.game, FinitePlayerGame):
            msg = f"{type(self).__name__} requires a FinitePlayerGame."
            raise TypeError(msg)
        if not isinstance(keepdims, (bool, np.bool_)):
            msg = "keepdims must be a bool."
            raise TypeError(msg)
        keepdims_flag = bool(keepdims)
        reduced_axes = _normalize_reduction_axes(axis, self.game.ndim)
        reduced_set = set(reduced_axes)
        if keepdims_flag:
            out_shape = tuple(
                1 if i in reduced_set else self.game.shape[i] for i in range(self.game.ndim)
            )
        else:
            out_shape = tuple(
                self.game.shape[i] for i in range(self.game.ndim) if i not in reduced_set
            )

        if self._players is not None:
            if not isinstance(self._players, FinitePlayers):
                msg = f"{type(self).__name__} _players must be a FinitePlayers instance."
                raise TypeError(msg)
            if jnp.broadcast_shapes(self._players.shape, out_shape) != out_shape:
                msg = "Player-set batch must broadcast to the reduction game's batch shape."
                raise ValueError(msg)
            game_players = self.game.players
            if (
                self._players.n_players != game_players.n_players
                or self._players.latent_n_players != game_players.latent_n_players
                or self._players.players is not game_players.players
            ):
                msg = "Supplied players are incompatible with the reduced game's player sets."
                raise ValueError(msg)
            offset = self.game.ndim - game_players.ndim
            p_reduced_axes = tuple(
                pk for pk in range(game_players.ndim) if (offset + pk) in reduced_set
            )
            if any(game_players.shape[pk] == 0 for pk in p_reduced_axes):
                final_players = self._players
            else:
                reduced_players = _validate_and_reduce_players(
                    game_players,
                    self.game.shape,
                    reduced_axes,
                    out_shape,
                    keepdims=keepdims_flag,
                )
                if not reduced_players.is_compatible(self._players):
                    msg = "Supplied players are incompatible with the reduced game's player sets."
                    raise ValueError(msg)
                final_players = self._players
        else:
            final_players = _validate_and_reduce_players(
                self.game.players,
                self.game.shape,
                reduced_axes,
                out_shape,
                keepdims=keepdims_flag,
            )

        rearranged = _rearrange_game_for_reduction(self.game, reduced_axes, out_shape)
        object.__setattr__(self, "game", rearranged)
        object.__setattr__(self, "_players", final_players)
        self.validate()

    @property
    @override
    def players(self) -> FinitePlayers[P]:
        """Return the surviving compatible player sets."""
        players = self._players
        if players is None:  # pragma: no cover - invariant enforced by __post_init__
            msg = f"{type(self).__name__} players were not initialized."
            raise RuntimeError(msg)
        return players

    @property
    def n_reduced(self) -> int:
        """Return the number of terms combined along the protected reduction axis."""
        return self.game.shape[-1]

    @override
    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Return lazy scalar abstract dtype metadata without evaluating the child game."""
        return self.game._value_spec()  # noqa: SLF001 - internal lazy abstract dtype hook

    @property
    @override
    def dtype(self) -> jnp.dtype:
        """Return the reduced game's value dtype without evaluating the child game."""
        return self._value_spec().dtype

    @property
    @override
    def weak_type(self) -> bool:
        """Return whether the reduced game's values are weakly typed without evaluating."""
        return self._value_spec().weak_type

    @abstractmethod
    def _reduce(self, values: Shaped[Value, "..."]) -> Shaped[V, "..."]:
        """Reduce evaluated child values along their final logical axis ``axis=-1``."""

    def _eval_child(self, coalition: FiniteCoalition) -> Shaped[V, "..."]:
        """Evaluate the underlying game over the protected axis and reduce along ``axis=-1``."""
        return self._reduce(self.game.eval(coalition[..., None, :]))

    @override
    def _coalition_bool(
        self, coalition: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        """Forward boolean coalitions lazily across the protected axis and reduce."""
        return self._eval_child(coalition)

    @override
    def _coalition_packed(
        self, coalition: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        """Forward packed coalitions lazily across the protected axis without unpacking."""
        return self._eval_child(coalition)

    @classmethod
    @override
    def _combine_sequence_metadata(
        cls,
        items: Sequence[Self],
        *,
        func: Callable[..., Any] | None,
    ) -> dict[str, Any]:
        """Reject combining reductions whose child value dtypes differ."""
        del func
        if len(items) <= 1:
            return {}
        signatures = {
            (item.dtype, item.game._value_spec())  # noqa: SLF001 - internal lazy abstract dtype hook
            for item in items
        }
        if len(signatures) > 1:
            msg = "Cannot combine ReductionGame instances with differing value dtypes."
            raise ValueError(msg)
        return {}

    @castmethod(DenseCoalitionGame, cost=1.0)
    def _as_dense_coalition(self) -> DenseCoalitionGame[P]:
        """Materialize directly on raw tables when the child supports fast dense casting."""
        if isinstance(self.game, DenseCoalitionGame):
            table = self.game
        else:
            result = try_cast(self.game, DenseCoalitionGame, max_cost=1.0)
            if result is None:
                msg = "Fast DenseCoalitionGame cast requires a DenseCoalitionGame leaf."
                raise CastUnavailable(msg)
            table = result.value
        # Keep the 2**n table axis at axis 0 so axis -1 remains the protected reduction axis.
        moved = jnp.moveaxis(table.values, -1, 0)
        computed = self._reduce(moved)
        if not isinstance(computed, jax.Array) or not jnp.issubdtype(computed.dtype, jnp.floating):
            msg = "DenseCoalitionGame requires floating jax.Array values."
            raise CastUnavailable(msg)
        values = jnp.moveaxis(computed, 0, -1)
        expected_shape = (*self.shape, 2**self.n_players)
        if values.shape != expected_shape:
            values = jnp.broadcast_to(values, expected_shape)
        return DenseCoalitionGame(values, self.players)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class SumReductionGame[
    P,
    V: Value = Value,
    GameT: FinitePlayerGame[Any, Any] = FinitePlayerGame[Any, Any],
](ReductionGame[P, V, GameT, "SumReductionGame[P, V, GameT]"]):
    """A lazy pointwise sum reduction over one or more batch axes of a finite game.

    When summing with ``output_dtype`` preserves every value leaf dtype, Möbius and
    Fourier queries forward linearly across the reduced axis and ``additivity()``
    propagates the child's known bound in ``O(1)``.
    """

    output_dtype: Any = field(default=None, kw_only=True)

    if TYPE_CHECKING:

        @overload
        def __init__[P2](
            self: "SumReductionGame[P2, jax.Array, FinitePlayerGame[Any, Any]]",
            game: FinitePlayerGame[P2, jax.Array],
            *,
            axis: int | Sequence[int] | None = ...,
            keepdims: bool = ...,
            output_dtype: "DTypeLike | None" = ...,
        ) -> None: ...

        @overload
        def __init__[P2, ResultT: Value](
            self: "SumReductionGame[P2, ResultT, FinitePlayerGame[Any, Any]]",
            game: FinitePlayerGame[
                P2, JaxLikeImplementation[Any, Any, ResultT, Any, Any, Any, Any]
            ],
            *,
            axis: int | Sequence[int] | None = ...,
            keepdims: bool = ...,
            output_dtype: "DTypeLike | None" = ...,
        ) -> None: ...

        @overload
        def __init__[P2](
            self: "SumReductionGame[P2, Value, FinitePlayerGame[Any, Any]]",
            game: FinitePlayerGame[P2, Value],
            *,
            axis: int | Sequence[int] | None = ...,
            keepdims: bool = ...,
            output_dtype: "DTypeLike | None" = ...,
        ) -> None: ...

        def __init__(
            self,
            game: FinitePlayerGame[Any, Any],
            *,
            axis: int | Sequence[int] | None = ...,
            keepdims: bool = ...,
            output_dtype: "DTypeLike | None" = ...,
        ) -> None:
            """Construct a lazy sum reduction over ``axis`` of ``game``."""

    def __post_init__(
        self,
        axis: int | Sequence[int] | None,
        keepdims: bool,
    ) -> None:
        """Validate ``output_dtype`` and prepare the reduced game layout."""
        resolved_dtype = (
            None
            if self.output_dtype is None
            else validate_numeric_dtype(self.output_dtype, "SumReductionGame output_dtype")
        )
        object.__setattr__(self, "output_dtype", resolved_dtype)
        super(SumReductionGame, self).__post_init__(axis, keepdims)
        validate_numeric_value(self.game, "SumReductionGame game")

    def _is_linear_sum(self) -> bool:
        """Return whether summing preserves every child leaf dtype (exact linearity)."""
        return _preserves_child_dtypes_for_sum(
            self.game, cast("jnp.dtype | None", self.output_dtype)
        )

    @override
    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Return exact JAX ``jnp.sum`` dtype and ``weak_type`` without evaluating."""
        return _sum_spec(self.game, cast("jnp.dtype | None", self.output_dtype))

    @property
    @override
    def dtype(self) -> jnp.dtype:
        """Return the summed value dtype without evaluating the underlying game."""
        return self._value_spec().dtype

    @override
    def _known_additivity(self) -> int | None:
        """Return ``0`` for zero-player or empty reductions, or the child's known bound when linear."""
        if self.n_players == 0 or self.n_reduced == 0:
            return 0
        child_bound = self.game._known_additivity()  # noqa: SLF001 - internal static bound check
        if child_bound is None or not self._is_linear_sum():
            return None
        return child_bound

    def _sum_values(
        self,
        values: Shaped[Value, "..."],
        *,
        axis: int,
        dtype: jnp.dtype | None = None,
    ) -> Shaped[V, "..."]:
        """Sum ``values`` along logical ``axis`` with ``dtype``."""
        if isinstance(values, jax.Array):
            return cast(
                "Shaped[V, '...']",
                jnp.sum(values, axis=axis, dtype=dtype),
            )
        return cast(
            "Shaped[V, '...']",
            jf.jax_sum(values, axis=axis, dtype=dtype),
        )

    def _basis_sum_dtype(self, basis: Basis) -> jnp.dtype | None:
        """Return the reduction dtype for ``basis``, avoiding integer truncation on Fourier."""
        if (
            basis is Basis.FOURIER
            and self.output_dtype is not None
            and not jnp.issubdtype(self.output_dtype, jnp.inexact)
        ):
            return None
        return self.output_dtype

    @override
    def _reduce(self, values: Shaped[Value, "..."]) -> Shaped[V, "..."]:
        """Sum evaluated child coalition values over the protected trailing axis."""
        return self._sum_values(values, axis=-1, dtype=self.output_dtype)

    @override
    def _zeros_like_value(self, shape: tuple[int, ...]) -> Shaped[V, "..."]:
        """Construct zero values lazily matching the reduced value structure and dtype."""
        if type(self.game)._zeros_like_value is not FinitePlayerGame._zeros_like_value:  # noqa: SLF001
            child_zero = self.game._zeros_like_value((*shape, 1))  # noqa: SLF001
            return self._sum_values(child_zero, axis=-1, dtype=self.output_dtype)
        return super(SumReductionGame, self)._zeros_like_value(shape)

    @override
    def empty_value(self) -> Shaped[V, "..."]:
        """Sum empty-coalition values directly without calling ``eval``."""
        child_empty = _ensure_shape(self.game.empty_value(), self.game.shape)
        return _ensure_shape(self._reduce(child_empty), self.shape)

    def _linear_basis_query(self, interaction: FiniteCoalition, basis: Basis) -> Shaped[V, "..."]:
        target_batch = jnp.broadcast_shapes(self.shape, interaction.shape[:-1])
        if self.n_reduced == 0:
            zero = self._zeros_like_value(target_batch)
            if basis is Basis.FOURIER:
                return cast("Shaped[V, '...']", _scale_by_pow2(cast("Any", zero), 0))
            return zero
        child_coeffs = _ensure_shape(
            self.game.coefficients(interaction[..., None, :], basis=basis),
            (*target_batch, self.n_reduced),
        )
        return _ensure_shape(
            self._sum_values(child_coeffs, axis=-1, dtype=self._basis_sum_dtype(basis)),
            target_batch,
        )

    @override
    def _moebius_bool(
        self, interaction: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        if self._is_linear_sum():
            return self._linear_basis_query(interaction, Basis.MOEBIUS)
        return super(SumReductionGame, self)._moebius_bool(interaction)

    @override
    def _moebius_packed(
        self, interaction: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        if self._is_linear_sum():
            return self._linear_basis_query(interaction, Basis.MOEBIUS)
        return super(SumReductionGame, self)._moebius_packed(interaction)

    @override
    def _fourier_bool(
        self, interaction: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        if self._is_linear_sum():
            return self._linear_basis_query(interaction, Basis.FOURIER)
        return super(SumReductionGame, self)._fourier_bool(interaction)

    @override
    def _fourier_packed(
        self, interaction: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        if self._is_linear_sum():
            return self._linear_basis_query(interaction, Basis.FOURIER)
        return super(SumReductionGame, self)._fourier_packed(interaction)

    @override
    def _coefficients_by_order(self, order: int, basis: Basis) -> Shaped[V, "..."]:
        if basis in (Basis.MOEBIUS, Basis.FOURIER) and self._is_linear_sum():
            n_combos = math.comb(self.n_players, order)
            known = self._known_additivity()
            if n_combos == 0 or self.n_reduced == 0 or (known is not None and order > known):
                zero = self._zeros_like_value((*self.shape, n_combos))
                if basis is Basis.FOURIER:
                    return cast("Shaped[V, '...']", _scale_by_pow2(cast("Any", zero), 0))
                return zero
            if order == 0 and basis is Basis.MOEBIUS:
                return cast("Shaped[V, '...']", jf.jax_expand_dims(self.empty_value(), axis=-1))
            child_coeffs = _ensure_shape(
                self.game.coefficients(order=order, basis=basis),
                (*self.game.shape, n_combos),
            )
            return _ensure_shape(
                self._sum_values(child_coeffs, axis=-2, dtype=self._basis_sum_dtype(basis)),
                (*self.shape, n_combos),
            )
        return super(SumReductionGame, self)._coefficients_by_order(order, basis)


_JAX_SUM_SIGNATURE = inspect.signature(jnp.sum)


def _dispatch_game_sum(
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
) -> Any:  # noqa: ANN401 - returns NotImplemented for non-game operands
    """Dispatch ``ndlike.jax_functions.jax_sum`` on a ``FinitePlayerGame``."""
    bound = _JAX_SUM_SIGNATURE.bind(*args, **(kwargs or {}))
    bound.apply_defaults()
    operand = bound.arguments["a"]
    if not isinstance(operand, FinitePlayerGame):
        return NotImplemented
    if bound.arguments["out"] is not None:
        msg = "Game sum reduction does not support the 'out' argument."
        raise NotImplementedError(msg)
    if bound.arguments["initial"] is not None:
        msg = "Game sum reduction does not support the 'initial' argument."
        raise NotImplementedError(msg)
    if bound.arguments["where"] is not None:
        msg = "Game sum reduction does not support the 'where' argument."
        raise NotImplementedError(msg)
    if bound.arguments["promote_integers"] is not True:
        msg = "Game sum reduction does not support custom 'promote_integers'."
        raise NotImplementedError(msg)
    return SumReductionGame(
        operand,
        axis=bound.arguments["axis"],
        keepdims=bound.arguments["keepdims"],
        output_dtype=bound.arguments["dtype"],
    )
