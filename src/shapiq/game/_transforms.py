"""Shared basis transforms, scatter routines, and cross-basis conversion helpers."""

import math
from itertools import combinations
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
from flextype import CastUnavailable
from jax.core import Tracer
from jaxtyping import Bool, Float, Shaped
from ndlike import jax_functions as jf

from shapiq.typing import typechecked
from shapiq.value._dtype import default_float_dtype
from shapiq.value.value import JaxValue

if TYPE_CHECKING:
    from .finite import FinitePlayerGame


@typechecked
def subset_patterns(
    size: int, min_size: int = 0, max_size: int | None = None
) -> Bool[jax.Array, "n_subsets {size}"]:
    """Return the subsets of a set of size ``size`` with ``min_size <= |T| <= max_size``.

    Row ``i`` of the unfiltered powerset is the little-endian bit pattern of ``i``; filtering keeps
    that order. Built with NumPy so the size filter stays static under ``jax.jit``.
    """
    low = max(0, min_size)
    high = size if max_size is None else min(size, max_size)
    if low > high:
        return jnp.zeros((0, size), dtype=bool)
    if low == 0 and high == size:
        patterns = (np.arange(1 << size)[:, None] >> np.arange(size)) & 1  # (2**size, size)
        return jnp.asarray(patterns, dtype=bool)
    if size <= 16:
        patterns = (np.arange(1 << size)[:, None] >> np.arange(size)) & 1
        sizes = patterns.sum(axis=-1)
        return jnp.asarray(patterns[(sizes >= low) & (sizes <= high)], dtype=bool)
    combos = [combo for order in range(low, high + 1) for combo in combinations(range(size), order)]
    combos.sort(key=lambda combo: combo[::-1])
    rows = np.zeros((len(combos), size), dtype=bool)
    for row_index, combo in enumerate(combos):
        if combo:
            rows[row_index, combo] = True
    return jnp.asarray(rows, dtype=bool)


@typechecked
def powerset(
    coalition: Bool[jax.Array, "*batch n_players"],
    order: int,
    min_size: int = 0,
    max_size: int | None = None,
) -> Bool[jax.Array, "n_subsets *batch n_players"]:
    """Generate the subsets of coalitions of size ``order`` with ``min_size <= |T| <= max_size``."""
    patterns = subset_patterns(order, min_size, max_size)  # (n_subsets, order)
    rank = jnp.cumsum(coalition, axis=-1) - 1  # (*batch n_players)
    pattern_columns_of_rank = (
        rank[..., None, :] == jnp.arange(order)[:, None]
    )  # (*batch, order, n_players)
    onehot = coalition[..., None, :] & pattern_columns_of_rank  # (*batch, order, n_players)
    patterns = patterns.reshape(
        patterns.shape[0], *(1,) * (coalition.ndim - 1), order, 1
    )  # (n_subsets, *batch, order, 1)
    return jnp.any(onehot & patterns, axis=-2)  # (n_subsets, *batch, n_players)


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


def _require_dense_values(values: object, target_name: str = "DenseGame") -> jax.Array:
    """Validate that stored values are a floating ``jax.Array`` for a dense game."""
    if not isinstance(values, jax.Array) or not jnp.issubdtype(values.dtype, jnp.floating):
        msg = (
            f"{target_name} requires floating jax.Array values, got {type(values).__name__} "
            f"(dtype={getattr(values, 'dtype', None)})."
        )
        raise CastUnavailable(msg)
    return values


def _check_dense_n_players(n_players: int) -> jnp.dtype:
    """Validate that ``2**n_players`` fits in JAX's enabled signed index/dimension width."""
    index_dtype = jnp.dtype(jnp.int64 if jax.config.x64_enabled else jnp.int32)
    # Both coalition indices (up to 2**n - 1) and axis length 2**n must fit in signed index_dtype.
    max_players = jnp.iinfo(index_dtype).bits - 2
    if n_players > max_players:
        msg = "Dense game indices exceed the enabled integer width."
        raise ValueError(msg)
    return index_dtype


def _scatter_basis_coefficients(
    values: Float[jax.Array, "*batch n_interactions"],
    interactions: Bool[jax.Array, "#*batch n_interactions {n_players}"],
    n_players: int,
) -> Float[jax.Array, "*batch n_coalitions"]:
    """Scatter basis coefficients into a dense ``(*batch, 2**n_players)`` accumulation table.

    Coalition indices follow little-endian bit order where bit ``j`` corresponds to local
    player ``j``. Duplicate interactions are summed via additive scatter in ``values.dtype``.
    """
    index_dtype = _check_dense_n_players(n_players)
    batch_shape = values.shape[:-1]
    n_interactions = values.shape[-1]
    if n_players == 0:
        return jnp.sum(values, axis=-1, keepdims=True, dtype=values.dtype)
    n_coalitions = 1 << n_players
    if n_interactions == 0 or 0 in batch_shape:
        return jnp.zeros((*batch_shape, n_coalitions), dtype=values.dtype)

    weights = jnp.left_shift(
        jnp.asarray(1, dtype=index_dtype),
        jnp.arange(n_players, dtype=index_dtype),
    )
    indices = jnp.sum(interactions.astype(index_dtype) * weights, axis=-1, dtype=index_dtype)

    if math.prod(interactions.shape[:-2]) == 1:
        table = jnp.zeros((*batch_shape, n_coalitions), dtype=values.dtype)
        return table.at[..., indices.reshape(n_interactions)].add(values)

    batch_size = math.prod(batch_shape)
    batch_idx = jnp.arange(batch_size)[:, None]
    indices_flat = jnp.broadcast_to(indices, (*batch_shape, n_interactions)).reshape(
        batch_size, n_interactions
    )
    values_flat = values.reshape(batch_size, n_interactions)
    table_flat = jnp.zeros((batch_size, n_coalitions), dtype=values.dtype)
    table_flat = table_flat.at[batch_idx, indices_flat].add(values_flat)
    return table_flat.reshape(*batch_shape, n_coalitions)


def _fast_zeta_transform(
    table: Float[jax.Array, "*batch n_coalitions"],
    n_players: int,
) -> Float[jax.Array, "*batch n_coalitions"]:
    """Apply the fast zeta transform (Möbius -> Coalition) in ``O(n * 2**n)``."""
    batch_shape = table.shape[:-1]
    out = table
    for k in range(n_players):
        out = out.reshape(*batch_shape, 1 << (n_players - 1 - k), 2, 1 << k)
        low = out[..., 0, :]
        high = out[..., 1, :]
        out = jnp.stack((low, low + high), axis=-2)
    return out.reshape(*batch_shape, 1 << n_players)


def _fast_moebius_transform(
    table: Float[jax.Array, "*batch n_coalitions"],
    n_players: int,
) -> Float[jax.Array, "*batch n_coalitions"]:
    """Apply the fast Möbius transform (Coalition -> Möbius) in ``O(n * 2**n)``."""
    batch_shape = table.shape[:-1]
    out = table
    for k in range(n_players):
        out = out.reshape(*batch_shape, 1 << (n_players - 1 - k), 2, 1 << k)
        low = out[..., 0, :]
        high = out[..., 1, :]
        out = jnp.stack((low, high - low), axis=-2)
    return out.reshape(*batch_shape, 1 << n_players)


def _fast_walsh_hadamard_transform(
    table: Float[jax.Array, "*batch n_coalitions"],
    n_players: int,
    *,
    normalize: bool = False,
) -> Float[jax.Array, "*batch n_coalitions"]:
    """Apply the fast Walsh-Hadamard transform (Fourier <-> Coalition) in ``O(n * 2**n)``."""
    batch_shape = table.shape[:-1]
    out = table
    half = jnp.asarray(0.5, dtype=table.dtype)
    for k in range(n_players):
        out = out.reshape(*batch_shape, 1 << (n_players - 1 - k), 2, 1 << k)
        low = out[..., 0, :]
        high = out[..., 1, :]
        if normalize:
            out = jnp.stack((half * (low + high), half * (low - high)), axis=-2)
        else:
            out = jnp.stack((low + high, low - high), axis=-2)
    return out.reshape(*batch_shape, 1 << n_players)


def _fast_moebius_to_fourier_transform(
    table: Float[jax.Array, "*batch n_coalitions"],
    n_players: int,
) -> Float[jax.Array, "*batch n_coalitions"]:
    """Apply the direct Möbius -> Fourier Kronecker transform in ``O(n * 2**n)``."""
    batch_shape = table.shape[:-1]
    out = table
    half = jnp.asarray(0.5, dtype=table.dtype)
    for k in range(n_players):
        out = out.reshape(*batch_shape, 1 << (n_players - 1 - k), 2, 1 << k)
        low = out[..., 0, :]
        high = out[..., 1, :]
        half_high = half * high
        out = jnp.stack((low + half_high, -half_high), axis=-2)
    return out.reshape(*batch_shape, 1 << n_players)


def _fast_fourier_to_moebius_transform(
    table: Float[jax.Array, "*batch n_coalitions"],
    n_players: int,
) -> Float[jax.Array, "*batch n_coalitions"]:
    """Apply the direct Fourier -> Möbius Kronecker transform in ``O(n * 2**n)``."""
    batch_shape = table.shape[:-1]
    out = table
    neg_two = jnp.asarray(-2.0, dtype=table.dtype)
    for k in range(n_players):
        out = out.reshape(*batch_shape, 1 << (n_players - 1 - k), 2, 1 << k)
        low = out[..., 0, :]
        high = out[..., 1, :]
        out = jnp.stack((low + high, neg_two * high), axis=-2)
    return out.reshape(*batch_shape, 1 << n_players)


def _additive_to_coalition_values(
    empty: Float[jax.Array, "*batch"],
    contributions: Float[jax.Array, "*batch {n_players}"],
    n_players: int,
) -> Float[jax.Array, "*batch n_coalitions"]:
    """Build coalition values for a 1-additive game in ``O(2**n_players)`` operations."""
    table = empty[..., None]
    for player in range(n_players):
        table = jnp.concatenate((table, table + contributions[..., player : player + 1]), axis=-1)
    return table


def _scale_by_pow2[V: JaxValue](
    values: Shaped[V, "..."], exponent: int | jax.Array
) -> Shaped[V, "..."]:
    """Scale ``values`` by ``2**exponent`` using ``ldexp``, preserving float/complex dtypes."""
    if isinstance(values, jax.Array):
        if jnp.issubdtype(values.dtype, jnp.floating):
            return cast("Shaped[V, '...']", jnp.ldexp(values, exponent))
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            real_dtype = jnp.finfo(values.dtype).dtype
            scale = jnp.ldexp(jnp.ones((), dtype=real_dtype), exponent)
            return cast("Shaped[V, '...']", values * scale)
        return cast("Shaped[V, '...']", jnp.ldexp(values.astype(default_float_dtype()), exponent))
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        scale_dtype = jnp.finfo(values.dtype).dtype
    elif jnp.issubdtype(values.dtype, jnp.floating):
        scale_dtype = values.dtype
    else:
        scale_dtype = default_float_dtype()
    return cast(
        "Shaped[V, '...']",
        jf.jax_multiply(values, jnp.ldexp(jnp.ones((), dtype=scale_dtype), exponent)),
    )


def _moebius_zero_like[P, V: JaxValue](
    game: "FinitePlayerGame[P, V]",
    shape: tuple[int, ...],
) -> Shaped[V, "..."]:
    """Return zero Möbius coefficients matching ``_moebius_coefficients_by_eval`` dtypes."""
    zero = game._zeros_like_value(shape)  # noqa: SLF001 - internal lazy zero hook
    return cast(
        "Shaped[V, '...']",
        jf.jax_sum(jf.jax_multiply(jf.jax_expand_dims(zero, axis=0), jnp.int8(1)), axis=0),
    )


@typechecked
def _moebius_coefficients_by_eval[P, V: JaxValue](
    game: "FinitePlayerGame[P, V]",
    interactions: Bool[jax.Array, "#*batch {game.n_players}"],
    *,
    bound: int | None = None,
    max_order: int | None = None,
) -> Shaped[V, "*batch"]:
    """Compute Möbius coefficients by subset inversion, zeroing above ``bound``."""
    n_players = game.n_players
    effective_bound = n_players if bound is None else min(bound, n_players)
    sizes = jnp.sum(interactions, axis=-1)
    target_batch = jnp.broadcast_shapes(game.shape, interactions.shape[:-1])

    if max_order is not None:
        if max_order > effective_bound:
            return _moebius_zero_like(game, target_batch)
        cap = max_order
    elif not isinstance(sizes, Tracer):
        if sizes.size == 0:
            return _moebius_zero_like(game, target_batch)
        max_valid = int(jnp.max(jnp.where(sizes <= effective_bound, sizes, -1)))
        if max_valid < 0:
            return _moebius_zero_like(game, target_batch)
        cap = max_valid
    else:
        cap = effective_bound

    in_bound = sizes <= cap
    clamped = interactions & in_bound[..., None]
    clamped_b = jnp.broadcast_to(clamped, (*target_batch, n_players))
    sizes_b = jnp.broadcast_to(sizes, target_batch)
    in_bound_b = jnp.broadcast_to(in_bound, target_batch)

    patterns = subset_patterns(cap)
    subsets = powerset(clamped_b, cap)
    pattern_sizes = jnp.sum(patterns, axis=-1, dtype=sizes_b.dtype)
    if cap == 0:
        required_sizes = jnp.zeros((1,), dtype=sizes_b.dtype)
    else:
        slot_index = jnp.arange(1, cap + 1, dtype=sizes_b.dtype)
        required_sizes = jnp.max(jnp.where(patterns, slot_index, 0), axis=-1)

    evals = game.eval(subsets)
    lead_shape = (subsets.shape[0], *(1,) * len(target_batch))
    pattern_sizes_b = pattern_sizes.reshape(lead_shape)
    required_sizes_b = required_sizes.reshape(lead_shape)
    valid = in_bound_b[None, ...] & (required_sizes_b <= sizes_b[None, ...])
    parity = ((sizes_b[None, ...] - pattern_sizes_b) & 1) == 0

    signs = jnp.where(parity, jnp.int8(1), jnp.int8(-1))
    signed_evals = jf.jax_multiply(evals, signs)
    zero = jf.jax_zeros_like(signed_evals, shape=())
    signed = jf.jax_where(valid, signed_evals, zero)
    return cast("Shaped[V, '...']", jf.jax_sum(signed, axis=0))


def _align_leading_axis[V: JaxValue](
    values: Shaped[V, "..."],
    target_batch: tuple[int, ...],
) -> Shaped[V, "..."]:
    """Broadcast ``(n_lead, *batch)`` to ``(n_lead, *target_batch)`` while preserving axis 0."""
    n_lead = values.shape[0]
    extra = len(target_batch) - (values.ndim - 1)
    aligned = (
        jf.jax_reshape(values, (n_lead, *(1,) * extra, *values.shape[1:])) if extra > 0 else values
    )
    return cast("Shaped[V, '...']", jf.jax_broadcast_to(aligned, (n_lead, *target_batch)))


@typechecked
def _fourier_coefficients_by_eval[P, V: JaxValue](
    game: "FinitePlayerGame[P, V]",
    interactions: Bool[jax.Array, "#*batch {game.n_players}"],
) -> Shaped[V, "*batch"]:
    """Compute Fourier coefficients ``f(T) = 2**-n * sum_S (-1)**|T & S| v(S)``."""
    n_players = game.n_players
    target_batch = jnp.broadcast_shapes(game.shape, interactions.shape[:-1])
    n_coalitions = 1 << n_players
    all_coalitions = subset_patterns(n_players)  # (2**n, n)
    evals = game.eval(
        all_coalitions.reshape(n_coalitions, *(1,) * game.ndim, n_players)
    )  # (2**n, *game.shape)
    evals_b = _align_leading_axis(evals, target_batch)
    coalitions_b = all_coalitions.reshape(n_coalitions, *(1,) * (interactions.ndim - 1), n_players)
    parity = jnp.sum(coalitions_b & interactions[None, ...], axis=-1)
    signs = _align_leading_axis(
        jnp.where((parity & 1) == 0, jnp.int8(1), jnp.int8(-1)), target_batch
    )
    total = jf.jax_sum(jf.jax_multiply(evals_b, signs), axis=0)
    return cast("Shaped[V, '...']", _scale_by_pow2(total, -n_players))


@typechecked
def _coalition_by_moebius[P, V: JaxValue](
    game: "FinitePlayerGame[P, V]",
    coalitions: Bool[jax.Array, "#*batch {game.n_players}"],
    *,
    bound: int | None = None,
) -> Shaped[V, "*batch"]:
    """Evaluate coalition values ``v(S) = sum_{T <= S, |T| <= bound} m(T)``."""
    n_players = game.n_players
    effective_bound = n_players if bound is None else min(bound, n_players)
    sizes = jnp.sum(coalitions, axis=-1)
    target_batch = jnp.broadcast_shapes(game.shape, coalitions.shape[:-1])

    if not isinstance(sizes, Tracer):
        if sizes.size == 0 or 0 in target_batch:
            return cast(
                "Shaped[V, '...']",
                game.moebius_coefficients(jnp.zeros((*target_batch, n_players), dtype=bool)),
            )
        cap = int(jnp.max(sizes))
    else:
        cap = n_players
    max_order = min(effective_bound, cap)

    coalitions_b = jnp.broadcast_to(coalitions, (*target_batch, n_players))
    sizes_b = jnp.broadcast_to(sizes, target_batch)
    patterns = subset_patterns(cap, 0, max_order)
    subsets = powerset(coalitions_b, cap, 0, max_order)
    if cap == 0:
        required_sizes = jnp.zeros((1,), dtype=sizes_b.dtype)
    else:
        slot_index = jnp.arange(1, cap + 1, dtype=sizes_b.dtype)
        required_sizes = jnp.max(jnp.where(patterns, slot_index, 0), axis=-1)

    m_vals = game.moebius_coefficients(subsets)
    lead_shape = (subsets.shape[0], *(1,) * len(target_batch))
    required_sizes_b = required_sizes.reshape(lead_shape)
    valid = required_sizes_b <= sizes_b[None, ...]
    zero = jf.jax_zeros_like(m_vals, shape=())
    return cast("Shaped[V, '...']", jf.jax_sum(jf.jax_where(valid, m_vals, zero), axis=0))


@typechecked
def _fourier_by_moebius[P, V: JaxValue](
    game: "FinitePlayerGame[P, V]",
    interactions: Bool[jax.Array, "#*batch {game.n_players}"],
    *,
    bound: int | None = None,
) -> Shaped[V, "*batch"]:
    """Compute Fourier coefficients ``f(T) = (-1)**|T| sum_{U >= T, |U| <= bound} 2**- |U| m(U)``."""
    n_players = game.n_players
    effective_bound = n_players if bound is None else min(bound, n_players)
    target_batch = jnp.broadcast_shapes(game.shape, interactions.shape[:-1])
    all_u = subset_patterns(n_players, 0, effective_bound)  # (K, n)
    n_sets = all_u.shape[0]
    m_vals = game.moebius_coefficients(
        all_u.reshape(n_sets, *(1,) * game.ndim, n_players)
    )  # (K, *game.shape)
    u_sizes = jnp.sum(all_u, axis=-1).reshape(n_sets, *(1,) * game.ndim)
    scaled = _scale_by_pow2(m_vals, -u_sizes)
    scaled_b = _align_leading_axis(scaled, target_batch)
    all_u_b = all_u.reshape(n_sets, *(1,) * (interactions.ndim - 1), n_players)
    contains = ~jnp.any(interactions[None, ...] & ~all_u_b, axis=-1)
    contains_b = _align_leading_axis(contains, target_batch)
    zero = jf.jax_zeros_like(scaled_b, shape=())
    total = jf.jax_sum(jf.jax_where(contains_b, scaled_b, zero), axis=0)
    t_sizes = jnp.broadcast_to(jnp.sum(interactions, axis=-1), target_batch)
    return cast("Shaped[V, '...']", jf.jax_where((t_sizes & 1) == 0, total, jf.jax_negative(total)))


@typechecked
def _coalition_by_fourier[P, V: JaxValue](
    game: "FinitePlayerGame[P, V]",
    coalitions: Bool[jax.Array, "#*batch {game.n_players}"],
    *,
    bound: int | None = None,
) -> Shaped[V, "*batch"]:
    """Evaluate coalition values ``v(S) = sum_{|T| <= bound} (-1)**|T & S| f(T)``."""
    n_players = game.n_players
    effective_bound = n_players if bound is None else min(bound, n_players)
    target_batch = jnp.broadcast_shapes(game.shape, coalitions.shape[:-1])
    all_t = subset_patterns(n_players, 0, effective_bound)  # (K, n)
    n_sets = all_t.shape[0]
    f_vals = game.fourier_coefficients(
        all_t.reshape(n_sets, *(1,) * game.ndim, n_players)
    )  # (K, *game.shape)
    f_vals_b = _align_leading_axis(f_vals, target_batch)
    all_t_b = all_t.reshape(n_sets, *(1,) * (coalitions.ndim - 1), n_players)
    parity = jnp.sum(all_t_b & coalitions[None, ...], axis=-1)
    signs = _align_leading_axis(
        jnp.where((parity & 1) == 0, jnp.int8(1), jnp.int8(-1)), target_batch
    )
    return cast("Shaped[V, '...']", jf.jax_sum(jf.jax_multiply(f_vals_b, signs), axis=0))


@typechecked
def _moebius_by_fourier[P, V: JaxValue](
    game: "FinitePlayerGame[P, V]",
    interactions: Bool[jax.Array, "#*batch {game.n_players}"],
    *,
    bound: int | None = None,
) -> Shaped[V, "*batch"]:
    """Compute Möbius coefficients ``m(U) = (-2)**|U| sum_{T >= U, |T| <= bound} f(T)``."""
    n_players = game.n_players
    effective_bound = n_players if bound is None else min(bound, n_players)
    target_batch = jnp.broadcast_shapes(game.shape, interactions.shape[:-1])
    all_t = subset_patterns(n_players, 0, effective_bound)  # (K, n)
    n_sets = all_t.shape[0]
    f_vals = game.fourier_coefficients(
        all_t.reshape(n_sets, *(1,) * game.ndim, n_players)
    )  # (K, *game.shape)
    f_vals_b = _align_leading_axis(f_vals, target_batch)
    all_t_b = all_t.reshape(n_sets, *(1,) * (interactions.ndim - 1), n_players)
    sizes = jnp.sum(interactions, axis=-1)
    in_bound = sizes <= effective_bound
    safe_sizes = jnp.broadcast_to(jnp.where(in_bound, sizes, 0), target_batch)
    contains = (~jnp.any(interactions[None, ...] & ~all_t_b, axis=-1)) & in_bound[None, ...]
    contains_b = _align_leading_axis(contains, target_batch)
    zero = jf.jax_zeros_like(f_vals_b, shape=())
    total = jf.jax_sum(jf.jax_where(contains_b, f_vals_b, zero), axis=0)
    scaled = _scale_by_pow2(total, safe_sizes)
    return cast(
        "Shaped[V, '...']", jf.jax_where((safe_sizes & 1) == 0, scaled, jf.jax_negative(scaled))
    )


@typechecked
def _exact_match_lookup(
    values: Float[JaxValue, "#*batch n_interactions"],
    stored_interactions: Bool[jax.Array, "#*batch n_interactions n_players"],
    query_interactions: Bool[jax.Array, "#*batch n_players"],
) -> Float[JaxValue, "*batch"]:
    """Look up stored basis coefficients directly by exact set match, summing duplicates."""
    target_batch = jnp.broadcast_shapes(
        values.shape[:-1], stored_interactions.shape[:-2], query_interactions.shape[:-1]
    )
    n_interactions = values.shape[-1]
    matches = jnp.all(stored_interactions == query_interactions[..., None, :], axis=-1)
    matches_b = jnp.broadcast_to(matches, (*target_batch, n_interactions))
    values_b = jf.jax_broadcast_to(values, (*target_batch, n_interactions))
    zero = jf.jax_zeros_like(values, shape=())
    return jf.jax_sum(jf.jax_where(matches_b, values_b, zero), axis=-1)


@typechecked
def _moebius_contributions(
    values: Float[JaxValue, "*batch n_interactions"],
    stored_interactions: Bool[jax.Array, "#*batch n_interactions {n_players}"],
    n_players: int,
    *,
    dtype: jnp.dtype | None = None,
) -> Float[JaxValue, "*batch {n_players}"]:
    """Extract singleton Möbius coefficients in canonical player order directly from storage."""
    game_shape = values.shape[:-1]
    if n_players == 0:
        return jf.jax_zeros_like(values, shape=(*game_shape, 0))
    n_interactions = values.shape[-1]
    is_singleton = jnp.sum(stored_interactions, axis=-1) == 1
    mask = stored_interactions & is_singleton[..., None]
    mask_b = jnp.broadcast_to(mask, (*game_shape, n_interactions, n_players))
    values_b = jf.jax_expand_dims(jf.jax_broadcast_to(values, (*game_shape, n_interactions)), -1)
    zero = jf.jax_zeros_like(values, shape=())
    return jf.jax_sum(jf.jax_where(mask_b, values_b, zero), axis=-2, dtype=dtype)


@typechecked
def _fourier_to_moebius_lookup(
    values: Float[JaxValue, "#*batch n_interactions"],
    stored_interactions: Bool[jax.Array, "#*batch n_interactions n_players"],
    query_interactions: Bool[jax.Array, "#*batch n_players"],
    bound: int,
) -> Float[JaxValue, "*batch"]:
    """Compute Möbius coefficients directly from stored Fourier terms via ``m(U) = (-2)**|U| sum_{T >= U} f(T)``."""
    target_batch = jnp.broadcast_shapes(
        values.shape[:-1], stored_interactions.shape[:-2], query_interactions.shape[:-1]
    )
    n_interactions = values.shape[-1]
    sizes = jnp.sum(query_interactions, axis=-1)
    in_bound = sizes <= bound
    safe_sizes = jnp.where(in_bound, sizes, 0)
    contains = ~jnp.any(query_interactions[..., None, :] & ~stored_interactions, axis=-1)
    contains = contains & in_bound[..., None]
    contains_b = jnp.broadcast_to(contains, (*target_batch, n_interactions))
    values_b = jf.jax_broadcast_to(values, (*target_batch, n_interactions))
    zero = jf.jax_zeros_like(values, shape=())
    total = jf.jax_sum(jf.jax_where(contains_b, values_b, zero), axis=-1)
    safe_sizes_b = jnp.broadcast_to(safe_sizes, target_batch)
    scaled = _scale_by_pow2(total, safe_sizes_b)
    return jf.jax_where((safe_sizes_b & 1) == 0, scaled, jf.jax_negative(scaled))


@typechecked
def _moebius_to_fourier_lookup(
    values: Float[JaxValue, "#*batch n_interactions"],
    stored_interactions: Bool[jax.Array, "#*batch n_interactions n_players"],
    query_interactions: Bool[jax.Array, "#*batch n_players"],
    bound: int,
) -> Float[JaxValue, "*batch"]:
    """Compute Fourier coefficients directly from stored Möbius terms via ``f(T) = (-1)**|T| sum_{U >= T} 2**- |U| m(U)``."""
    target_batch = jnp.broadcast_shapes(
        values.shape[:-1], stored_interactions.shape[:-2], query_interactions.shape[:-1]
    )
    n_interactions = values.shape[-1]
    u_sizes = jnp.sum(stored_interactions, axis=-1)
    values_b = jf.jax_broadcast_to(values, (*target_batch, n_interactions))
    u_sizes_b = jnp.broadcast_to(u_sizes, (*target_batch, n_interactions))
    scaled_values = _scale_by_pow2(values_b, -u_sizes_b)

    t_sizes = jnp.sum(query_interactions, axis=-1)
    in_bound = t_sizes <= bound
    contains = (~jnp.any(query_interactions[..., None, :] & ~stored_interactions, axis=-1)) & (
        in_bound[..., None]
    )
    contains_b = jnp.broadcast_to(contains, (*target_batch, n_interactions))
    zero = jf.jax_zeros_like(scaled_values, shape=())
    total = jf.jax_sum(jf.jax_where(contains_b, scaled_values, zero), axis=-1)
    t_sizes_b = jnp.broadcast_to(t_sizes, target_batch)
    return jf.jax_where((t_sizes_b & 1) == 0, total, jf.jax_negative(total))
