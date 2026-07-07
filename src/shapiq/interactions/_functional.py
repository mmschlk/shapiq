"""Coalition functionals: derived linear-functional representations of indices.

Every cardinal interaction index and generalized value is a linear functional
of the game whose coefficient on ``v(K)`` depends only on the cardinalities
``|K & S|`` and ``|K|``. Deriving that coefficient table mechanically from an
index's declared weights is what lets exact contraction and unbiased
importance sampling fall out of one declaration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import TYPE_CHECKING, ClassVar

import jax.numpy as jnp

from shapiq._shape import ensure_bool, validate_int
from shapiq.interactions._indices import (
    BV,
    SV,
    CardinalInteractionIndex,
    GeneralizedValueIndex,
)
from shapiq.interactions._iteration import interaction_masks

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jax import Array

    from shapiq.interactions._indices import InteractionIndex
    from shapiq.interactions._types import (
        InteractionIndexName,
        InteractionOrientation,
        OrderSemantics,
    )


@dataclass(frozen=True)
class CoalitionFunctional:
    """The linear-functional representation of an interaction index.

    A coalition functional assigns to every represented interaction size
    ``s`` a coefficient table of shape ``(s + 1, n_players + 1)`` whose entry
    ``[l, k]`` is the coefficient of ``v(K)`` in the attribution of an
    interaction ``S`` with ``|S| = s``, ``|K & S| = l``, and ``|K| = k``. The
    attribution of ``S`` is the sum of these coefficients times the game
    values over all coalitions. Exact explainers contract the table densely;
    Monte Carlo approximators importance-sample it, with the proposal derived
    from the table's mass profile.
    """

    n_players: int
    tables: dict[int, Array]

    @property
    def interaction_sizes(self) -> tuple[int, ...]:
        """Return the represented interaction sizes in increasing order."""
        return tuple(sorted(self.tables))

    def coefficient_matrix(self, masks: Array, size: int) -> Array:
        """Return per-coalition, per-interaction coefficients for one size.

        Args:
            masks: Dense coalition masks of shape ``(..., n_coalitions,
                n_players)``; leading axes broadcast.
            size: Represented interaction size to look up.

        Returns:
            Coefficients of shape ``(..., n_coalitions, n_interactions)``
            where interactions of the size are ordered lexicographically.
        """
        members = interaction_masks(self.n_players, size)
        intersections = jnp.einsum(
            "...cn,in->...ci",
            masks.astype(jnp.int32),
            members.astype(jnp.int32),
        )
        coalition_sizes = jnp.sum(masks, axis=-1).astype(jnp.int32)[..., None]
        return self.tables[size][intersections, coalition_sizes]

    def size_mass(self) -> Array:
        """Return the total absolute coefficient mass per coalition size.

        The entry for size ``k`` sums ``|coefficient|`` over every coalition
        of size ``k`` and every represented interaction; it is the natural
        importance proposal for sampled estimation, and sizes with zero mass
        never need to be evaluated.
        """
        n = self.n_players
        mass = [0.0] * (n + 1)
        for size, table in self.tables.items():
            for shared in range(size + 1):
                for coalition_size in range(n + 1):
                    outside = coalition_size - shared
                    if outside < 0 or outside > n - size:
                        continue
                    n_coalitions = comb(size, shared) * comb(n - size, outside)
                    n_interactions = comb(n, size)
                    coefficient = float(table[shared, coalition_size])
                    mass[coalition_size] += n_interactions * n_coalitions * abs(coefficient)
        return jnp.asarray(mass)


def derive_functional(
    index: InteractionIndex,
    n_players: int,
    order: int,
) -> CoalitionFunctional:
    """Derive the coalition functional of a linear-functional index.

    Args:
        index: An index carrying discrete-derivative weights (cardinal
            interaction index) or bloc-marginal weights (generalized value).
        n_players: Number of players of the explained game.
        order: Resolved maximum interaction order.

    Returns:
        The derived coalition functional.

    Raises:
        TypeError: If the index declares neither weight formalism.
        ValueError: If a declared weight profile has the wrong length.
    """
    if isinstance(index, CardinalInteractionIndex):
        tables = {
            size: _cardinal_table(
                n_players,
                size,
                _checked_weights(index, index.derivative_weights(n_players, size), n_players, size),
            )
            for size in range(index.min_interaction_size, order + 1)
        }
        return CoalitionFunctional(n_players=n_players, tables=tables)
    if isinstance(index, GeneralizedValueIndex):
        tables = {
            size: _marginal_table(
                n_players,
                size,
                _checked_weights(index, index.marginal_weights(n_players, size), n_players, size),
            )
            for size in range(1, order + 1)
        }
        return CoalitionFunctional(n_players=n_players, tables=tables)
    name = getattr(index, "name", type(index).__name__)
    msg = (
        f"no coalition functional is derivable for {name!r}: the index declares "
        "neither discrete-derivative weights nor bloc-marginal weights"
    )
    raise TypeError(msg)


def aggregate_supersets(
    blocks: dict[int, Array],
    coefficients: Sequence[float],
    n_players: int,
) -> dict[int, Array]:
    """Aggregate per-size attribution blocks over supersets.

    The aggregated attribution of an interaction ``S`` sums the base
    attributions of its supersets ``T`` among the represented sizes, weighted
    by ``coefficients[|T| - |S|]``. Aggregation is linear, so it preserves
    both exactness and unbiasedness of the base blocks.
    """
    members = {size: interaction_masks(n_players, size) for size in blocks}
    aggregated: dict[int, Array] = {}
    for size, base_block in blocks.items():
        block = coefficients[0] * base_block
        for larger, larger_block in blocks.items():
            if larger <= size:
                continue
            counts = members[size].astype(jnp.int32) @ members[larger].T.astype(jnp.int32)
            subsets = 1.0 * (counts == size)
            block = block + coefficients[larger - size] * jnp.einsum(
                "...j,ij->...i",
                larger_block,
                subsets,
            )
        aggregated[size] = block
    return aggregated


def define_cardinal_index(
    name: InteractionIndexName,
    *,
    weights: Callable[[int, int], Array],
    order: int = 2,
    order_semantics: OrderSemantics = "coverage",
    includes_empty_interaction: bool = False,
    min_interaction_size: int = 1,
    generalizes: SV | BV | None = None,
) -> InteractionIndex:
    """Define a new cardinal interaction index from its weight formalism.

    The returned index carries the declared discrete-derivative weights and
    works with every explainer that consumes the cardinal capability: exact
    contraction and derived Monte Carlo estimation come with the definition.

    Args:
        name: Name recorded on explanations produced for the index.
        weights: Function mapping ``(n_players, interaction_size)`` to one
            discrete-derivative weight per outside-coalition size
            ``0..n_players - interaction_size``.
        order: Maximum interaction order of the index.
        order_semantics: Whether order is explanation coverage or identity.
        includes_empty_interaction: Whether explanations carry an order-0
            attribution.
        min_interaction_size: Smallest represented interaction size.
        generalizes: Probabilistic value the order-1 restriction equals, when
            the definition declares one.

    Returns:
        An immutable interaction index carrying the declared formalism.
    """
    _validate_definition(name, order=order, order_semantics=order_semantics)
    validate_int("min_interaction_size", min_interaction_size, minimum=0)
    return _DefinedCardinalIndex(
        name=name,
        order=order,
        order_semantics=order_semantics,
        includes_empty_interaction=ensure_bool(
            "includes_empty_interaction",
            includes_empty_interaction,
        ),
        min_interaction_size=min_interaction_size,
        generalizes=generalizes,
        weights=weights,
    )


def define_generalized_value(
    name: InteractionIndexName,
    *,
    weights: Callable[[int, int], Array],
    order: int = 2,
    order_semantics: OrderSemantics = "coverage",
    includes_empty_interaction: bool = False,
    generalizes: SV | BV | None = None,
) -> InteractionIndex:
    """Define a new generalized value from its bloc-marginal weight formalism.

    Args:
        name: Name recorded on explanations produced for the index.
        weights: Function mapping ``(n_players, interaction_size)`` to one
            bloc-marginal weight per outside-coalition size
            ``0..n_players - interaction_size``.
        order: Maximum interaction order of the index.
        order_semantics: Whether order is explanation coverage or identity.
        includes_empty_interaction: Whether explanations carry an order-0
            attribution.
        generalizes: Probabilistic value the order-1 restriction equals, when
            the definition declares one.

    Returns:
        An immutable interaction index carrying the declared formalism.
    """
    _validate_definition(name, order=order, order_semantics=order_semantics)
    return _DefinedGeneralizedValue(
        name=name,
        order=order,
        order_semantics=order_semantics,
        includes_empty_interaction=ensure_bool(
            "includes_empty_interaction",
            includes_empty_interaction,
        ),
        generalizes=generalizes,
        weights=weights,
    )


def define_regression_index(
    name: InteractionIndexName,
    *,
    kernel: Callable[[int], Array],
    order: int = 2,
    order_semantics: OrderSemantics = "identity",
    generalizes: SV | BV | None = None,
) -> InteractionIndex:
    """Define a new regression index from its kernel formalism.

    The returned index is the best ``order``-additive approximation of the
    game under the declared kernel, interpolating the empty and grand
    coalition exactly as constraints. It works with every explainer that
    consumes the regression capability: the exact solve and the
    kernel-matched sampled ``Regression`` estimator come with the
    definition. Its order-0 attribution is the empty-coalition value, as for
    every constrained-interpolation regression index.

    Args:
        name: Name recorded on explanations produced for the index.
        kernel: Function mapping ``n_players`` to one nonnegative kernel
            weight per coalition size ``0..n_players``. The empty and grand
            coalition must carry weight zero; they are interpolated exactly
            as constraints rather than weighted.
        order: Maximum interaction order of the index.
        order_semantics: Whether order is explanation coverage or identity;
            a best ``order``-additive fit changes with the order, so the
            default is identity.
        generalizes: Probabilistic value the order-1 restriction equals, when
            the definition declares one.

    Returns:
        An immutable interaction index carrying the declared formalism.
    """
    _validate_definition(name, order=order, order_semantics=order_semantics)
    return _DefinedRegressionIndex(
        name=name,
        order=order,
        order_semantics=order_semantics,
        generalizes=generalizes,
        kernel=kernel,
    )


@dataclass(frozen=True)
class _DefinedCardinalIndex:
    """A cardinal interaction index declared through its weight formalism."""

    name: InteractionIndexName
    order: int
    order_semantics: OrderSemantics
    includes_empty_interaction: bool
    min_interaction_size: int
    generalizes: SV | BV | None
    weights: Callable[[int, int], Array] = field(repr=False)

    orientation: ClassVar[InteractionOrientation] = "undirected"

    def derivative_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return the declared discrete-derivative weights per outside size."""
        return jnp.asarray(self.weights(n_players, interaction_size), dtype=jnp.float32)


@dataclass(frozen=True)
class _DefinedGeneralizedValue:
    """A generalized value declared through its bloc-marginal weight formalism."""

    name: InteractionIndexName
    order: int
    order_semantics: OrderSemantics
    includes_empty_interaction: bool
    generalizes: SV | BV | None
    weights: Callable[[int, int], Array] = field(repr=False)

    orientation: ClassVar[InteractionOrientation] = "undirected"

    def marginal_weights(self, n_players: int, interaction_size: int) -> Array:
        """Return the declared bloc-marginal weights per outside size."""
        return jnp.asarray(self.weights(n_players, interaction_size), dtype=jnp.float32)


@dataclass(frozen=True)
class _DefinedRegressionIndex:
    """A regression index declared through its kernel formalism."""

    name: InteractionIndexName
    order: int
    order_semantics: OrderSemantics
    generalizes: SV | BV | None
    kernel: Callable[[int], Array] = field(repr=False)

    orientation: ClassVar[InteractionOrientation] = "undirected"
    includes_empty_interaction: ClassVar[bool] = True

    def regression_kernel(self, n_players: int) -> Array:
        """Return the declared kernel weights per coalition size."""
        return jnp.asarray(self.kernel(n_players), dtype=jnp.float32)


def _validate_definition(
    name: InteractionIndexName,
    *,
    order: int,
    order_semantics: OrderSemantics,
) -> None:
    """Validate the shared metadata of a defined index."""
    if not isinstance(name, str) or not name:
        msg = "defined index names must be non-empty strings"
        raise ValueError(msg)
    validate_int("order", order, minimum=1)
    if order_semantics not in {"coverage", "identity"}:
        msg = f"unsupported order semantics: {order_semantics!r}"
        raise ValueError(msg)


def _checked_weights(
    index: InteractionIndex,
    weights: Array,
    n_players: int,
    size: int,
) -> Array:
    """Validate a declared weight profile before deriving from it."""
    profile = jnp.asarray(weights)
    expected = n_players - size + 1
    if profile.shape != (expected,):
        msg = (
            f"{index.name!r} declares a weight profile of shape {profile.shape} for "
            f"interaction size {size}, expected one weight per outside-coalition "
            f"size 0..{n_players - size}, shape ({expected},)"
        )
        raise ValueError(msg)
    return profile


def _cardinal_table(n_players: int, size: int, weights: Array) -> Array:
    """Return the coefficient table of a discrete-derivative weight profile.

    Expanding the discrete derivative of ``S`` at ``T`` over game values gives
    coefficient ``(-1) ** (size - shared) * weights[|K| - shared]`` for a
    coalition ``K`` sharing ``shared`` players with ``S``.
    """
    shared = jnp.arange(size + 1)[:, None]
    coalition_sizes = jnp.arange(n_players + 1)[None, :]
    outside = coalition_sizes - shared
    valid = (outside >= 0) & (outside <= n_players - size)
    signs = jnp.where((size - shared) % 2 == 0, 1.0, -1.0)
    return jnp.where(valid, signs * weights[jnp.clip(outside, 0, n_players - size)], 0.0)


def _marginal_table(n_players: int, size: int, weights: Array) -> Array:
    """Return the coefficient table of a bloc-marginal weight profile.

    A coalition containing the whole interaction contributes a gain and a
    disjoint coalition a loss; partially overlapping coalitions carry no
    coefficient.
    """
    table = jnp.zeros((size + 1, n_players + 1))
    gain_sizes = jnp.arange(size, n_players + 1)
    table = table.at[size, gain_sizes].add(weights[gain_sizes - size])
    loss_sizes = jnp.arange(n_players - size + 1)
    return table.at[0, loss_sizes].add(-weights[loss_sizes])
