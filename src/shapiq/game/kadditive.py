"""Nominal k-additive and additive interfaces for finite cooperative games."""

import math
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Any, Literal, cast, overload, override

import jax
import jax.numpy as jnp
import numpy as np
from beartype.door import is_bearable
from jaxtyping import Bool, Shaped, UInt8
from ndlike import jax_functions as jf

from shapiq.coalition import FiniteCoalition
from shapiq.coalition.coalition import unpack_bool_coalition
from shapiq.typing import typechecked
from shapiq.value.value import JaxValue

from .finite import FinitePlayerGame


def _order_interactions(
    n_players: int, order: int
) -> Bool[jax.Array, "n_interactions {n_players}"]:
    """Return all interactions of size ``order`` in little-endian bitmask order."""
    if order == 0:
        return jnp.zeros((1, n_players), dtype=jnp.bool_)
    if order == 1:
        return jnp.eye(n_players, dtype=jnp.bool_)
    if order > n_players:
        return jnp.zeros((0, n_players), dtype=jnp.bool_)
    if order == n_players:
        return jnp.ones((1, n_players), dtype=jnp.bool_)
    # Reverse-lexicographic sorting matches little-endian bitmask order.
    combos = sorted(combinations(range(n_players), order), key=lambda combo: combo[::-1])
    rows = np.zeros((len(combos), n_players), dtype=bool)
    for row_index, combo in enumerate(combos):
        rows[row_index, combo] = True
    return jnp.asarray(rows)


class KAdditiveGame[
    P,
    V: JaxValue = JaxValue,
    OwnerT: "FinitePlayerGame[Any, Any, Any]" = Any,
](FinitePlayerGame[P, V, OwnerT], ABC):
    """Nominal interface for k-additive finite cooperative games.

    All Möbius coefficients above ``additivity`` vanish. The bound is
    nonnegative and need not be tight; a nonzero empty-coalition value is
    allowed. Nominal inheritance supports ``issubclass`` and ``flextype``
    routing where structural protocols with properties cannot.

    Concrete subclasses own their Möbius coefficient lookup by implementing
    ``_coefficients_for_interactions``. Structured ``JaxAxisProtected`` value
    types used with k-additive games must opt into ``jax_where`` (and
    ``jax_sum`` / ``jax_multiply`` for subset-inversion games, plus ``jax_add``
    for affine wrappers) in ``permitted_functions``.
    """

    @property
    @abstractmethod
    def additivity(self) -> int:
        """Return a nonnegative upper bound on the nonzero Möbius interaction order."""

    def _zeros_like_value(self, shape: tuple[int, ...]) -> Shaped[V, "..."]:
        """Construct zero coefficients of logical ``shape`` matching this game's value type."""
        return cast("Shaped[V, '...']", jf.jax_zeros_like(self.empty_value(), shape=shape))

    @overload
    def coefficients(
        self,
        interaction: FiniteCoalition,
        *,
        order: None = None,
    ) -> Shaped[V, "..."]: ...

    @overload
    def coefficients(
        self,
        interaction: None = None,
        *,
        order: int,
    ) -> Shaped[V, "..."]: ...

    def coefficients(
        self,
        interaction: FiniteCoalition | None = None,
        *,
        order: int | None = None,
    ) -> Shaped[V, "..."]:
        """Return Möbius coefficients for explicit interactions or all interactions of ``order``.

        Pass either ``interaction`` (boolean or packed ``uint8``, possibly
        mixed-order) or keyword ``order`` (nonnegative integer), not both.
        Coefficients for interactions above ``additivity`` are zero.

        Raises:
            TypeError: Both or neither of ``interaction`` and ``order`` were
                supplied, or ``order`` is not an integer.
            ValueError: ``order`` is negative or ``interaction`` has an
                inadmissible batch shape.
        """
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
            return self._coefficients_by_order(order)
        if interaction is None:  # pragma: no cover - unreachable guard for type narrowing
            msg = "Specify either interaction or order."
            raise TypeError(msg)
        return self._validate_and_coefficients(interaction)

    @typechecked
    def _validate_and_coefficients(
        self,
        interaction: Bool[jax.Array, "#*batch {self.n_players}"]
        | UInt8[jax.Array, "#*batch ({self.n_players}+7)//8"],
    ) -> Shaped[V, "*batch"]:
        """Validate interaction storage and batch shape before computing coefficients."""
        if not is_bearable(self, Shaped[FinitePlayerGame[P, V], "#*batch"]):
            msg = "Inadmissible interaction batch."
            raise ValueError(msg)
        if interaction.dtype == jnp.uint8:
            interaction = unpack_bool_coalition(interaction, self.n_players)
        return self._coefficients_for_interactions(interaction)

    @abstractmethod
    def _coefficients_for_interactions(
        self,
        interactions: Bool[jax.Array, "#*batch {self.n_players}"],
        *,
        max_order: int | None = None,
    ) -> Shaped[V, "..."]:
        """Compute Möbius coefficients for validated boolean interactions."""

    def _coefficients_by_order(self, order: int) -> Shaped[V, "..."]:
        """Compute all Möbius coefficients of static size ``order``."""
        n_combos = math.comb(self.n_players, order)
        if order > self.additivity or n_combos == 0:
            return self._zeros_like_value((*self.shape, n_combos))
        patterns = _order_interactions(self.n_players, order).reshape(
            n_combos, *(1,) * self.ndim, self.n_players
        )
        coeffs = self._coefficients_for_interactions(patterns, max_order=order)
        return cast("Shaped[V, '...']", jf.jax_moveaxis(coeffs, 0, -1))


class AdditiveGame[
    P,
    V: JaxValue = JaxValue,
    OwnerT: "FinitePlayerGame[Any, Any, Any]" = Any,
](KAdditiveGame[P, V, OwnerT], ABC):
    """Nominal interface for 1-additive finite games, allowing a nonzero empty value.

    Coalition values satisfy
    ``v(S) = v(empty) + sum_{i in S} contributions()[..., i]``, where
    ``contributions()`` orders singleton Möbius coefficients by canonical
    sorted player position.

    Concrete subclasses own ``contributions()``; ``coefficients()`` derives
    directly from ``empty_value()`` and ``contributions()``.
    """

    @property
    @override
    def additivity(self) -> Literal[0, 1]:
        """Return ``0`` for zero-player games and ``1`` otherwise."""
        return 0 if self.n_players == 0 else 1

    @abstractmethod
    def contributions(self) -> Shaped[V, "..."]:
        """Return singleton Möbius coefficients in canonical player order.

        Returns an array-like value with logical shape
        ``(*self.shape, self.n_players)``.
        """

    @override
    def _coefficients_by_order(self, order: int) -> Shaped[V, "..."]:
        """Return empty, singleton, or zero higher-order coefficients directly."""
        if order == 0:
            return cast("Shaped[V, '...']", jf.jax_expand_dims(self.empty_value(), axis=-1))
        if order == 1 and self.additivity >= 1:
            return self.contributions()
        n_combos = math.comb(self.n_players, order)
        return self._zeros_like_value((*self.shape, n_combos))

    @override
    def _coefficients_for_interactions(
        self,
        interactions: Bool[jax.Array, "#*batch {self.n_players}"],
        *,
        max_order: int | None = None,
    ) -> Shaped[V, "..."]:
        """Evaluate additive Möbius coefficients from ``empty_value`` and ``contributions``."""
        del max_order
        target_batch = jnp.broadcast_shapes(self.shape, interactions.shape[:-1])
        sizes = jnp.broadcast_to(jnp.sum(interactions, axis=-1), target_batch)
        empty = jf.jax_broadcast_to(self.empty_value(), target_batch)
        if self.n_players == 0:
            return cast("Shaped[V, '...']", empty)
        zero = jf.jax_zeros_like(empty, shape=())
        if self.additivity == 0:
            return cast("Shaped[V, '...']", jf.jax_where(sizes == 0, empty, zero))
        contribs = self.contributions()
        contribs_b = jf.jax_broadcast_to(contribs, (*target_batch, self.n_players))
        player_index = jnp.broadcast_to(jnp.argmax(interactions, axis=-1), target_batch)
        singleton = cast(
            "Any", jf.jax_take_along_axis(contribs_b, player_index[..., None], axis=-1)
        )[..., 0]
        return cast(
            "Shaped[V, '...']",
            jf.jax_where(sizes == 0, empty, jf.jax_where(sizes == 1, singleton, zero)),
        )
