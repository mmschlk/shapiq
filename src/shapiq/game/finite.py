"""Games with boolean and packed finite-coalition evaluation hooks."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, override

import jax
import jax.numpy as jnp
from beartype.door import is_bearable
from jaxtyping import Bool, Shaped, UInt8
from ndlike.jax_like import JaxLikeImplementation

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
    from shapiq.value.value import JaxValue

    from .arithmetic import AffineGame, ArithmeticGame

type _FiniteGame[P] = FinitePlayerGame[P, Value]


class FinitePlayerGame[P, V: Value, OwnerT: "FinitePlayerGame[Any, Any, Any]" = Any](
    Game[FiniteCoalition, V, OwnerT, _FiniteGame[P]], ABC
):
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

    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Return lazy scalar abstract dtype metadata without evaluating the game.

        Defaults conservatively to ``weak_type=False`` because unknown custom
        implementations may not preserve JAX weak typing across evaluation.
        """
        return jax.ShapeDtypeStruct((), jnp.dtype(self.dtype), weak_type=False)

    def empty_value(self) -> Shaped[V, "..."]:
        """Evaluate the game on the empty coalition for every game in the batch."""
        return self.eval(self.players.empty_coalition())

    def full_value(self) -> Shaped[V, "..."]:
        """Evaluate the game on the full coalition for every game in the batch."""
        return self.eval(self.players.full_coalition())

    @override
    def __pos__(self) -> Self:
        """Return this game."""
        return self

    @override
    def __neg__(self) -> "AffineGame[P, JaxValue]":
        """Negate the game pointwise."""
        return cast("AffineGame[P, JaxValue]", super().__neg__())

    @override
    def __add__(self, other: object, /) -> "AffineGame[P, JaxValue]":
        """Add the right operand pointwise."""
        return cast("AffineGame[P, JaxValue]", super().__add__(other))

    @override
    def __radd__(self, other: object, /) -> "AffineGame[P, JaxValue]":
        """Add the game to the left operand pointwise."""
        return cast("AffineGame[P, JaxValue]", super().__radd__(other))

    @override
    def __sub__(self, other: object, /) -> "AffineGame[P, JaxValue]":
        """Subtract the right operand pointwise."""
        return cast("AffineGame[P, JaxValue]", super().__sub__(other))

    @override
    def __rsub__(self, other: object, /) -> "AffineGame[P, JaxValue]":
        """Subtract the game from the left operand pointwise."""
        return cast("AffineGame[P, JaxValue]", super().__rsub__(other))

    @override
    def __mul__(self, other: object, /) -> "ArithmeticGame[P, JaxValue]":
        """Multiply the operands pointwise."""
        return cast("ArithmeticGame[P, JaxValue]", super().__mul__(other))

    @override
    def __rmul__(self, other: object, /) -> "ArithmeticGame[P, JaxValue]":
        """Multiply the left operand by the game pointwise."""
        return cast("ArithmeticGame[P, JaxValue]", super().__rmul__(other))

    @override
    def __truediv__(self, other: object, /) -> "ArithmeticGame[P, JaxValue]":
        """Divide by the right operand pointwise."""
        return cast("ArithmeticGame[P, JaxValue]", super().__truediv__(other))

    @override
    def __rtruediv__(self, other: object, /) -> "ArithmeticGame[P, JaxValue]":
        """Divide the left operand by the game pointwise."""
        return cast("ArithmeticGame[P, JaxValue]", super().__rtruediv__(other))

    @override
    def __pow__(self, other: object, modulo: object = None) -> "ArithmeticGame[P, JaxValue]":
        """Raise the game to the right operand pointwise."""
        return cast("ArithmeticGame[P, JaxValue]", super().__pow__(other, modulo))

    @override
    def __rpow__(self, other: object, modulo: object = None) -> "ArithmeticGame[P, JaxValue]":
        """Raise the left operand to the game pointwise."""
        return cast("ArithmeticGame[P, JaxValue]", super().__rpow__(other, modulo))

    @classmethod
    def __jax_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401 - ndlike dispatch protocol returns arbitrary result or NotImplemented
        """Route numeric arithmetic through lazy game wrappers; delegate other ops to super."""
        if not kwargs:
            from ._arithmetic_ops import _dispatch_game_arithmetic  # noqa: PLC0415

            result = _dispatch_game_arithmetic(func, args)
            if result is not NotImplemented:
                return result
        super_fn = super().__jax_function__
        if super_fn.__func__ is JaxLikeImplementation.__jax_function__.__func__:  # type: ignore[attr-defined]
            return NotImplemented
        return super_fn(func, types, args, kwargs)

    @typechecked
    @override
    def _eval(
        self,
        coalition: Bool[jax.Array, "#*batch {self.n_players}"]
        | UInt8[jax.Array, "#*batch ({self.n_players}+7)//8"],
    ) -> Shaped[V, "*batch"]:
        """Validate and dispatch finite coalitions to the matching evaluation hook.

        Raises:
            ValueError: The coalition batch is inadmissible for this game.
            TypeCheckError: The coalition representation or returned value is invalid.
        """
        if not is_bearable(self, Shaped[FinitePlayerGame[P, V], "#*batch"]):
            msg = "Inadmissible coalition batch."
            raise ValueError(msg)
        if coalition.dtype == jnp.bool_:
            return self._eval_bool(coalition)
        return self._eval_packed(coalition)

    @abstractmethod
    def _eval_bool(self, coalition: Bool[jax.Array, "... {self.n_players}"]) -> Shaped[V, "..."]:
        """Evaluate boolean membership, broadcasting with the game batch."""

    @abstractmethod
    def _eval_packed(
        self, coalition: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        """Evaluate packed membership, ignoring unused input padding bits."""


class BoolFinitePlayerGame[P, V: Value, OwnerT: "BoolFinitePlayerGame[Any, Any, Any]" = Any](
    FinitePlayerGame[P, V, OwnerT]
):
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


class PackedFinitePlayerGame[P, V: Value, OwnerT: "PackedFinitePlayerGame[Any, Any, Any]" = Any](
    FinitePlayerGame[P, V, OwnerT]
):
    """A finite-player game with a native ``_eval_packed`` implementation.

    Boolean input is packed automatically. Subclasses may also override
    ``_eval_bool`` to support both representations natively.
    """

    @override
    def _eval_bool(self, coalition: Bool[jax.Array, "... {self.n_players}"]) -> Shaped[V, "..."]:
        """Pack boolean membership and defer to packed evaluation."""
        return self._eval_packed(pack_bool_coalition(coalition))
