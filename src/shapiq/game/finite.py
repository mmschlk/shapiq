"""Games with boolean and packed finite-coalition evaluation hooks."""

from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, overload, override

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Shaped, TypeCheckError, UInt8
from ndlike import jax_functions as jf
from ndlike.jax_like import JaxLike, JaxLikeImplementation

from shapiq.coalition import FiniteCoalition
from shapiq.coalition.coalition import (
    pack_bool_coalition,
    unpack_bool_coalition,
)
from shapiq.players import FinitePlayers
from shapiq.typing import typechecked
from shapiq.value import Value

from .game import Game

if TYPE_CHECKING:
    from jax.sharding import Sharding
    from jax.typing import DTypeLike
    from numpy.typing import DTypeLike as NpDTypeLike

    from .arithmetic import AffineGame, ArithmeticGame
    from .reduction import SumReductionGame

type _FiniteGame[P] = FinitePlayerGame[P, Value]  # noqa: PYI047 - imported by finite-game subclasses


class FinitePlayerGame[P, V: Value](Game[FiniteCoalition, V]):
    """A game accepting boolean and little-endian packed finite coalitions.

    Implement both evaluation hooks for native support of both representations,
    or derive from ``BoolFinitePlayerGame`` or ``PackedFinitePlayerGame`` to
    implement just the corresponding native hook and inherit a conversion for
    the other representation. Hooks own value-batch broadcasting; public
    ``eval`` validates coalitions against the players and game batch and checks
    that the returned value's logical shape is the broadcasted batch. Packed
    implementations must ignore unused bits in the final input byte.
    Evaluation checks these contracts directly; ``is_valid`` is an explicit
    predicate and is not called during evaluation.
    """

    # Defer NumPy ufuncs (e.g., ``np.array(1.0) + game``) to game reflected operators.
    __array_ufunc__: ClassVar[None] = None

    @property
    @abstractmethod
    @override
    def players(self) -> FinitePlayers[P]:
        """Return finite player sets whose logical shape broadcasts to ``shape``."""

    @property
    def n_players(self) -> int:
        """Return the number of players in each game, which is constant across the batch."""
        return self.players.n_players

    @override
    def astype(
        self,
        dtype: "DTypeLike | NpDTypeLike | None",
        copy: bool = False,
        device: "jax.Device | Sharding | None" = None,
    ) -> Any:
        """Return this finite game or a lazy cast of its evaluated values."""
        from .cast import CastGame  # noqa: PLC0415 - CastGame depends on FinitePlayerGame

        if dtype is None:
            game = self if device is None else self.to_device(device)
            return game.copy() if copy else game
        return CastGame(self.copy() if copy else self, dtype, device=device)

    def empty_value(self) -> Shaped[V, "..."]:
        """Evaluate the game on the empty coalition for every game in the batch."""
        return self.eval(self.players.empty_coalition())

    def full_value(self) -> Shaped[V, "..."]:
        """Evaluate the game on the full coalition for every game in the batch."""
        return self.eval(self.players.full_coalition())

    @overload
    def sum(
        self: "FinitePlayerGame[P, jax.Array]",
        axis: int | Sequence[int] | None = None,
        dtype: "DTypeLike | None" = None,
        *,
        keepdims: bool = False,
    ) -> "SumReductionGame[P, jax.Array]": ...

    @overload
    def sum[ResultT: Value](
        self: "FinitePlayerGame[P, JaxLikeImplementation[Any, Any, ResultT, Any, Any, Any, Any]]",
        axis: int | Sequence[int] | None = None,
        dtype: "DTypeLike | None" = None,
        *,
        keepdims: bool = False,
    ) -> "SumReductionGame[P, ResultT]": ...

    @overload
    def sum(
        self,
        axis: int | Sequence[int] | None = None,
        dtype: "DTypeLike | None" = None,
        *,
        keepdims: bool = False,
    ) -> "SumReductionGame[P, Value]": ...

    def sum(
        self,
        axis: int | Sequence[int] | None = None,
        dtype: "DTypeLike | None" = None,
        *,
        keepdims: bool = False,
    ) -> "SumReductionGame[P, Any]":
        """Sum this game batch pointwise over ``axis`` via ``ndlike.jax_functions.jax_sum``."""
        return jf.jax_sum(
            self,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
        )

    @override
    def __pos__(self) -> Self:
        """Return this game."""
        return self

    @override
    def __neg__(self) -> "AffineGame[P, Value]":
        """Negate the game pointwise."""
        return cast("AffineGame[P, Value]", super().__neg__())

    @override
    def __add__(self, other: object, /) -> "AffineGame[P, Value]":
        """Add the right operand pointwise."""
        return cast("AffineGame[P, Value]", super().__add__(other))

    @override
    def __radd__(self, other: object, /) -> "AffineGame[P, Value]":
        """Add the game to the left operand pointwise."""
        return cast("AffineGame[P, Value]", super().__radd__(other))

    @override
    def __sub__(self, other: object, /) -> "AffineGame[P, Value]":
        """Subtract the right operand pointwise."""
        return cast("AffineGame[P, Value]", super().__sub__(other))

    @override
    def __rsub__(self, other: object, /) -> "AffineGame[P, Value]":
        """Subtract the game from the left operand pointwise."""
        return cast("AffineGame[P, Value]", super().__rsub__(other))

    @override
    def __mul__(self, other: object, /) -> "ArithmeticGame[P, Value]":
        """Multiply the operands pointwise."""
        return cast("ArithmeticGame[P, Value]", super().__mul__(other))

    @override
    def __rmul__(self, other: object, /) -> "ArithmeticGame[P, Value]":
        """Multiply the left operand by the game pointwise."""
        return cast("ArithmeticGame[P, Value]", super().__rmul__(other))

    @override
    def __truediv__(self, other: object, /) -> "ArithmeticGame[P, Value]":
        """Divide by the right operand pointwise."""
        return cast("ArithmeticGame[P, Value]", super().__truediv__(other))

    @override
    def __rtruediv__(self, other: object, /) -> "ArithmeticGame[P, Value]":
        """Divide the left operand by the game pointwise."""
        return cast("ArithmeticGame[P, Value]", super().__rtruediv__(other))

    @override
    def __pow__(self, other: object, modulo: object = None) -> "ArithmeticGame[P, Value]":
        """Raise the game to the right operand pointwise."""
        if modulo is not None:
            return NotImplemented
        return cast("ArithmeticGame[P, Value]", super().__pow__(other))

    @override
    def __rpow__(self, other: object, modulo: object = None) -> "ArithmeticGame[P, Value]":
        """Raise the left operand to the game pointwise."""
        if modulo is not None:
            return NotImplemented
        return cast("ArithmeticGame[P, Value]", super().__rpow__(other))

    @classmethod
    def __jax_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401 - ndlike dispatch protocol returns arbitrary result or NotImplemented
        """Route numeric arithmetic and reductions through lazy game wrappers; delegate other ops to super."""
        if func is jf.jax_sum:
            from .reduction import _dispatch_game_sum  # noqa: PLC0415

            result = _dispatch_game_sum(args, kwargs)
            if result is not NotImplemented:
                return result
        elif not kwargs:
            from ._arithmetic_ops import _dispatch_game_arithmetic  # noqa: PLC0415

            result = _dispatch_game_arithmetic(func, args)
            if result is not NotImplemented:
                return result
        super_fn = getattr(super(), "__jax_function__", None)
        if (
            super_fn is None or super_fn.__func__ is JaxLikeImplementation.__jax_function__.__func__  # type: ignore[attr-defined]
        ):
            return NotImplemented
        return super_fn(func, types, args, kwargs)

    @typechecked
    @override
    def _eval(
        self,
        coalition: Bool[jax.Array, "... {self.n_players}"]
        | UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"],
    ) -> Shaped[V, "..."]:
        """Validate and dispatch finite coalitions to the matching evaluation hook.

        Raises:
            ValueError: The coalition batch is inadmissible for this game.
            TypeCheckError: The coalition representation or returned value is invalid.
        """
        # The game batch is independent of the dtype of its evaluated values.
        try:
            shape = jnp.broadcast_shapes(self.shape, coalition.shape[:-1])
        except ValueError as error:
            msg = "Inadmissible coalition batch."
            raise ValueError(msg) from error
        value = (
            self._eval_bool(coalition)
            if coalition.dtype == jnp.bool_
            else self._eval_packed(coalition)
        )
        if isinstance(value, (jax.Array, JaxLike)) and value.shape != shape:
            msg = f"Invalid return value shape {value.shape}; expected {shape}."
            raise TypeCheckError(msg)
        return value

    @abstractmethod
    def _eval_bool(self, coalition: Bool[jax.Array, "... {self.n_players}"]) -> Shaped[V, "..."]:
        """Evaluate boolean membership, broadcasting with the game batch."""

    @abstractmethod
    def _eval_packed(
        self, coalition: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        """Evaluate packed membership, ignoring unused input padding bits."""


class BoolFinitePlayerGame[P, V: Value](FinitePlayerGame[P, V]):
    """A finite-player game with a native ``_eval_bool`` implementation.

    Packed input is unpacked automatically. Subclasses may also override
    ``_eval_packed`` to support both representations natively.
    """

    @override
    def _eval_packed(
        self, coalition: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        """Unpack exactly the local player count and defer to boolean evaluation."""
        return self._eval_bool(unpack_bool_coalition(coalition, self.n_players))


class PackedFinitePlayerGame[P, V: Value](FinitePlayerGame[P, V]):
    """A finite-player game with a native ``_eval_packed`` implementation.

    Boolean input is packed automatically. Subclasses may also override
    ``_eval_bool`` to support both representations natively.
    """

    @override
    def _eval_bool(self, coalition: Bool[jax.Array, "... {self.n_players}"]) -> Shaped[V, "..."]:
        """Pack boolean membership and defer to packed evaluation."""
        return self._eval_packed(pack_bool_coalition(coalition))
