"""Finite-player games, discrete game bases, native-basis base classes, and additivity."""

import math
from abc import abstractmethod
from collections.abc import Callable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, overload, override

import jax
import jax.numpy as jnp
from flextype import try_cast
from jaxtyping import Bool, Shaped, TypeCheckError, UInt8
from ndlike import jax_functions as jf
from ndlike.jax_like import JaxLike, JaxLikeImplementation

from shapiq.coalition import FiniteCoalition
from shapiq.coalition.coalition import unpack_bool_coalition
from shapiq.players import FinitePlayers
from shapiq.typing import typechecked
from shapiq.value import Value

from ._transforms import (
    _coalition_by_fourier,
    _coalition_by_moebius,
    _fourier_by_moebius,
    _fourier_coefficients_by_eval,
    _moebius_by_fourier,
    _moebius_coefficients_by_eval,
    _moebius_zero_like,
    _order_interactions,
    subset_patterns,
)
from .base import Game

if TYPE_CHECKING:
    from jax.sharding import Sharding
    from jax.typing import DTypeLike
    from numpy.typing import DTypeLike as NpDTypeLike

    from .arithmetic import AffineGame, ArithmeticGame
    from .reduction import SumReductionGame

type _FiniteGame[P] = FinitePlayerGame[P, Value]  # noqa: PYI047 - imported by finite-game subclasses


class Basis(Enum):
    """Supported discrete linear bases for finite cooperative games."""

    COALITION = "coalition"
    MOEBIUS = "moebius"
    FOURIER = "fourier"


class FinitePlayerGame[P, V: Value](Game[FiniteCoalition, V]):
    """A finite-player game supporting Coalition, Möbius, and Fourier basis queries.

    Subclasses implement the three boolean basis hooks ``_coalition_bool``,
    ``_moebius_bool``, and ``_fourier_bool``, or inherit from one of the three
    native-basis base classes (``CoalitionDefinedGame``, ``MoebiusDefinedGame``,
    ``FourierDefinedGame``) to implement one native hook and inherit conversions
    for the other two. Packed ``uint8`` hooks default to unpacking and calling
    the matching boolean hook; representations with native packed support (such
    as ``DenseGame`` or lazy wrappers) may override them.

    Hooks own value-batch broadcasting; public entry points validate coalitions
    and interactions against the players and game batch and check that the
    returned value's logical shape is the broadcasted batch. Packed
    implementations must ignore unused bits in the final input byte.
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

    def _zeros_like_value(self, shape: tuple[int, ...]) -> Shaped[V, "..."]:
        """Construct zero values of logical ``shape`` matching this game's value type."""
        return cast("Shaped[V, '...']", jf.jax_zeros_like(self.empty_value(), shape=shape))

    def _known_additivity(self) -> int | None:
        """Return a statically known nonnegative additivity upper bound, or ``None``."""
        return 0 if self.n_players == 0 else None

    def additivity(self) -> int:
        """Return an upper bound on this game's Möbius interaction order.

        Games implementing ``KnownAdditivity`` (and compatible lazy wrappers
        around them) return their known static bound in ``O(1)`` without
        evaluating coalitions. Otherwise, this computes the exact additivity
        from the game's Möbius coefficients, which may evaluate up to
        ``2**n_players`` coalitions.
        """
        known = self._known_additivity()
        if known is not None:
            return known
        from .dense import DenseMoebiusGame  # noqa: PLC0415 - avoids circular import

        # Prefer a single O(n * 2**n) Möbius transform when the dense representation is available.
        dense_result = try_cast(self, DenseMoebiusGame)
        if dense_result is not None:
            table = dense_result.value.values
            if table.size == 0:
                return 0
            nonzero = jnp.any(table != 0, axis=tuple(range(table.ndim - 1)))
            if not bool(jnp.any(nonzero)):
                return 0
            orders = jnp.sum(subset_patterns(self.n_players), axis=-1)
            return int(jnp.max(jnp.where(nonzero, orders, 0)))

        for order in range(self.n_players, 0, -1):
            coeffs = self.moebius_coefficients(order=order)
            if any(bool(jnp.any(leaf != 0)) for leaf in jax.tree.leaves(coeffs)):
                return order
        return 0

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

    def contributions(self) -> Shaped[V, "..."]:
        """Return singleton Möbius coefficients in canonical player order.

        Returns an array-like value with logical shape
        ``(*self.shape, self.n_players)``, equivalent to
        ``self.moebius_coefficients(order=1)``.
        """
        return self.moebius_coefficients(order=1)

    @overload
    def coefficients(
        self,
        interaction: FiniteCoalition,
        *,
        order: None = None,
        basis: Basis = Basis.MOEBIUS,
    ) -> Shaped[V, "..."]: ...

    @overload
    def coefficients(
        self,
        interaction: None = None,
        *,
        order: int,
        basis: Basis = Basis.MOEBIUS,
    ) -> Shaped[V, "..."]: ...

    def _coefficients(
        self,
        interaction: FiniteCoalition | None,
        order: int | None,
        basis: Basis,
    ) -> Shaped[V, "..."]:
        """Validate and route coefficient queries for ``basis``."""
        if not isinstance(basis, Basis):
            msg = f"basis must be a Basis member, got {type(basis).__name__}."
            raise TypeError(msg)
        if (interaction is None) == (order is None):
            msg = "Specify either interaction or order, not both."
            raise TypeError(msg)
        if order is not None:
            if not isinstance(order, int) or isinstance(order, bool):
                msg = "order must be an integer."
                raise TypeError(msg)
            if order < 0:
                msg = "order must be nonnegative."
                raise ValueError(msg)
            return self._coefficients_by_order(order, basis)
        if interaction is None:  # pragma: no cover - unreachable guard for type narrowing
            msg = "Specify either interaction or order."
            raise TypeError(msg)
        return self._validate_and_dispatch(interaction, basis)

    def coefficients(
        self,
        interaction: FiniteCoalition | None = None,
        *,
        order: int | None = None,
        basis: Basis = Basis.MOEBIUS,
    ) -> Shaped[V, "..."]:
        """Return coefficients in ``basis`` for explicit interactions or all interactions of ``order``.

        Pass either ``interaction`` (boolean or packed ``uint8``, possibly
        mixed-order) or keyword ``order`` (nonnegative integer), not both.
        Defaults to ``Basis.MOEBIUS``.

        Raises:
            TypeError: Both or neither of ``interaction`` and ``order`` were
                supplied, ``order`` is not an integer, or ``basis`` is not a
                ``Basis`` member.
            ValueError: ``order`` is negative or ``interaction`` has an
                inadmissible batch shape.
        """
        return self._coefficients(interaction, order, basis)

    @overload
    def moebius_coefficients(
        self,
        interaction: FiniteCoalition,
        *,
        order: None = None,
    ) -> Shaped[V, "..."]: ...

    @overload
    def moebius_coefficients(
        self,
        interaction: None = None,
        *,
        order: int,
    ) -> Shaped[V, "..."]: ...

    def moebius_coefficients(
        self,
        interaction: FiniteCoalition | None = None,
        *,
        order: int | None = None,
    ) -> Shaped[V, "..."]:
        """Return Möbius coefficients for explicit interactions or all interactions of ``order``."""
        return self._coefficients(interaction, order, Basis.MOEBIUS)

    @overload
    def fourier_coefficients(
        self,
        interaction: FiniteCoalition,
        *,
        order: None = None,
    ) -> Shaped[V, "..."]: ...

    @overload
    def fourier_coefficients(
        self,
        interaction: None = None,
        *,
        order: int,
    ) -> Shaped[V, "..."]: ...

    def fourier_coefficients(
        self,
        interaction: FiniteCoalition | None = None,
        *,
        order: int | None = None,
    ) -> Shaped[V, "..."]:
        """Return Fourier coefficients for explicit interactions or all interactions of ``order``."""
        return self._coefficients(interaction, order, Basis.FOURIER)

    def _coefficients_by_order(self, order: int, basis: Basis) -> Shaped[V, "..."]:
        """Compute all coefficients of static size ``order`` in ``basis``."""
        n_combos = math.comb(self.n_players, order)
        if n_combos == 0:
            return self._zeros_like_value((*self.shape, 0))
        known = self._known_additivity()
        if basis in (Basis.MOEBIUS, Basis.FOURIER) and known is not None and order > known:
            return self._zeros_like_value((*self.shape, n_combos))
        patterns = _order_interactions(self.n_players, order).reshape(
            n_combos, *(1,) * self.ndim, self.n_players
        )
        coeffs = self._validate_and_dispatch(patterns, basis)
        return cast("Shaped[V, '...']", jf.jax_moveaxis(coeffs, 0, -1))

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
    def _validate_and_dispatch(
        self,
        coalition: Bool[jax.Array, "... {self.n_players}"]
        | UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"],
        basis: Basis,
    ) -> Shaped[V, "..."]:
        """Validate and dispatch finite coalitions/interactions to the matching basis hook.

        Raises:
            ValueError: The coalition/interaction batch is inadmissible for this game.
            TypeCheckError: The representation or returned value shape is invalid.
        """
        try:
            shape = jnp.broadcast_shapes(self.shape, coalition.shape[:-1])
        except ValueError as error:
            msg = "Inadmissible coalition batch."
            raise ValueError(msg) from error
        is_bool = coalition.dtype == jnp.bool_
        if basis is Basis.COALITION:
            value = (
                self._coalition_bool(coalition) if is_bool else self._coalition_packed(coalition)
            )
        elif basis is Basis.MOEBIUS:
            value = self._moebius_bool(coalition) if is_bool else self._moebius_packed(coalition)
        else:
            value = self._fourier_bool(coalition) if is_bool else self._fourier_packed(coalition)
        if isinstance(value, (jax.Array, JaxLike)) and value.shape != shape:
            msg = f"Invalid return value shape {value.shape}; expected {shape}."
            raise TypeCheckError(msg)
        return value

    @override
    def _eval(self, coalition: FiniteCoalition) -> Shaped[V, "..."]:
        """Evaluate coalitions in the Coalition (standard) basis."""
        return self._validate_and_dispatch(coalition, Basis.COALITION)

    @abstractmethod
    def _coalition_bool(
        self, coalition: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        """Evaluate boolean coalitions in the Coalition basis, broadcasting with the game batch."""

    def _coalition_packed(
        self, coalition: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        """Evaluate packed coalitions in the Coalition basis; defaults to unpacking."""
        return self._coalition_bool(unpack_bool_coalition(coalition, self.n_players))

    @abstractmethod
    def _moebius_bool(
        self, interaction: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        """Compute Möbius coefficients for boolean interactions, broadcasting with the game batch."""

    def _moebius_packed(
        self, interaction: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        """Compute Möbius coefficients for packed interactions; defaults to unpacking."""
        return self._moebius_bool(unpack_bool_coalition(interaction, self.n_players))

    @abstractmethod
    def _fourier_bool(
        self, interaction: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        """Compute Fourier coefficients for boolean interactions, broadcasting with the game batch."""

    def _fourier_packed(
        self, interaction: UInt8[jax.Array, "... bytes=({self.n_players}+7)//8"]
    ) -> Shaped[V, "..."]:
        """Compute Fourier coefficients for packed interactions; defaults to unpacking."""
        return self._fourier_bool(unpack_bool_coalition(interaction, self.n_players))


class _KnownAdditivityBase[P, V: Value](FinitePlayerGame[P, V]):
    """Shared implementation for finite games storing a static ``_additivity`` bound."""

    @property
    @abstractmethod
    def _additivity(self) -> int:
        """Return the stored nonnegative additivity upper bound."""

    @override
    def _known_additivity(self) -> int:
        """Return the stored nonnegative additivity upper bound."""
        return self._additivity

    @override
    def additivity(self) -> int:
        """Return the stored nonnegative additivity upper bound without evaluating coalitions."""
        return self._additivity


if TYPE_CHECKING:

    class KnownAdditivity[P, V: Value](_KnownAdditivityBase[P, V]):
        """A finite-player game carrying an ``O(1)`` statically known additivity upper bound.

        Direct subclasses store ``_additivity: int`` and inherit ``additivity()`` and
        ``_known_additivity()``. Lazy wrappers (such as ``AffineGame``, ``ProductGame``,
        ``SumReductionGame``, and ``CastGame``) also satisfy ``isinstance(game, KnownAdditivity)``
        whenever their propagated ``_known_additivity()`` is not ``None``.
        """
else:

    class _KnownAdditivityMeta(type(FinitePlayerGame)):
        """Enable ``isinstance(game, KnownAdditivity)`` for wrappers with known static bounds."""

        @override
        def __instancecheck__(self, instance: object) -> bool:
            if self is KnownAdditivity and isinstance(instance, FinitePlayerGame):
                return instance._known_additivity() is not None  # noqa: SLF001 - internal static bound hook
            return super().__instancecheck__(instance)

    class KnownAdditivity[P, V: Value](_KnownAdditivityBase[P, V], metaclass=_KnownAdditivityMeta):
        """A finite-player game carrying an ``O(1)`` statically known additivity upper bound.

        Direct subclasses store ``_additivity: int`` and inherit ``additivity()`` and
        ``_known_additivity()``. Lazy wrappers (such as ``AffineGame``, ``ProductGame``,
        ``SumReductionGame``, and ``CastGame``) also satisfy ``isinstance(game, KnownAdditivity)``
        whenever their propagated ``_known_additivity()`` is not ``None``.
        """


class CoalitionDefinedGame[P, V: Value](FinitePlayerGame[P, V]):
    """A finite-player game whose native representation is the Coalition basis.

    Leaves ``_coalition_bool`` abstract and supplies default Möbius (subset
    inversion) and Fourier (Walsh-Hadamard sum) conversions.
    """

    @override
    def _moebius_bool(
        self, interaction: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        """Compute Möbius coefficients by subset inversion."""
        return cast(
            "Shaped[V, '...']",
            _moebius_coefficients_by_eval(
                cast("FinitePlayerGame[P, Any]", self),
                interaction,
                bound=self._known_additivity(),
            ),
        )

    @override
    def _fourier_bool(
        self, interaction: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        """Compute Fourier coefficients by summing all coalition evaluations."""
        return cast(
            "Shaped[V, '...']",
            _fourier_coefficients_by_eval(cast("FinitePlayerGame[P, Any]", self), interaction),
        )

    @override
    def _coefficients_by_order(self, order: int, basis: Basis) -> Shaped[V, "..."]:
        """Cap subset inversion statically at ``order`` for Möbius order queries."""
        if basis is Basis.MOEBIUS:
            n_combos = math.comb(self.n_players, order)
            known = self._known_additivity()
            bound = self.n_players if known is None else min(known, self.n_players)
            if order > bound or n_combos == 0:
                return cast(
                    "Shaped[V, '...']",
                    _moebius_zero_like(
                        cast("FinitePlayerGame[P, Any]", self), (*self.shape, n_combos)
                    ),
                )
            patterns = _order_interactions(self.n_players, order).reshape(
                n_combos, *(1,) * self.ndim, self.n_players
            )
            coeffs = _moebius_coefficients_by_eval(
                cast("FinitePlayerGame[P, Any]", self),
                patterns,
                bound=known,
                max_order=order,
            )
            return cast("Shaped[V, '...']", jf.jax_moveaxis(coeffs, 0, -1))
        return super()._coefficients_by_order(order, basis)


class MoebiusDefinedGame[P, V: Value](FinitePlayerGame[P, V]):
    """A finite-player game whose native representation is the Möbius basis.

    Leaves ``_moebius_bool`` abstract and supplies default Coalition (subset
    sum) and Fourier conversions.
    """

    @override
    def empty_value(self) -> Shaped[V, "..."]:
        """Return ``v(empty) = m(empty)`` directly from the native Möbius hook."""
        return self.moebius_coefficients(jnp.zeros((self.n_players,), dtype=bool))

    @override
    def _coalition_bool(
        self, coalition: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        """Evaluate coalitions by summing subset Möbius coefficients."""
        return cast(
            "Shaped[V, '...']",
            _coalition_by_moebius(
                cast("FinitePlayerGame[P, Any]", self),
                coalition,
                bound=self._known_additivity(),
            ),
        )

    @override
    def _fourier_bool(
        self, interaction: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        """Compute Fourier coefficients from superset Möbius coefficients."""
        return cast(
            "Shaped[V, '...']",
            _fourier_by_moebius(
                cast("FinitePlayerGame[P, Any]", self),
                interaction,
                bound=self._known_additivity(),
            ),
        )


class FourierDefinedGame[P, V: Value](FinitePlayerGame[P, V]):
    """A finite-player game whose native representation is the Fourier basis.

    Leaves ``_fourier_bool`` abstract and supplies default Coalition and Möbius
    conversions.
    """

    @override
    def _coalition_bool(
        self, coalition: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        """Evaluate coalitions by summing parity-weighted Fourier coefficients."""
        return cast(
            "Shaped[V, '...']",
            _coalition_by_fourier(
                cast("FinitePlayerGame[P, Any]", self),
                coalition,
                bound=self._known_additivity(),
            ),
        )

    @override
    def _moebius_bool(
        self, interaction: Bool[jax.Array, "... {self.n_players}"]
    ) -> Shaped[V, "..."]:
        """Compute Möbius coefficients from superset Fourier coefficients."""
        return cast(
            "Shaped[V, '...']",
            _moebius_by_fourier(
                cast("FinitePlayerGame[P, Any]", self),
                interaction,
                bound=self._known_additivity(),
            ),
        )
