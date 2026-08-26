"""Aggregation functions for summarizing base interaction indices into efficient indices useful for explanations."""

from __future__ import annotations

import warnings
from itertools import chain, combinations
from typing import TYPE_CHECKING, overload

import numpy as np
import scipy as sp

from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import powerset

if TYPE_CHECKING:
    from collections.abc import Mapping


def _change_index(index: str) -> str:
    """Changes the index of the interaction values to the new index.

    Args:
        index: The current index of the interaction values.

    Returns:
        The new index of the interaction values.

    """
    if index in ["SV", "BV"]:  # no change for probabilistic values like SV or BV
        return index
    return f"k-{index}"


# The aggregation is linear, so it applies equally to scalar values and to per-instance value
# arrays. The overloads keep scalar callers scalar-typed; Mapping is covariant in its value
# type, so both dict variants are accepted without casts.
@overload
def aggregate_base_attributions(
    interactions: Mapping[tuple[int, ...], float],
    index: str,
    order: int,
    min_order: int,
    baseline_value: float,
) -> tuple[dict[tuple[int, ...], float], str, int]: ...
@overload
def aggregate_base_attributions(
    interactions: Mapping[tuple[int, ...], float | np.ndarray],
    index: str,
    order: int,
    min_order: int,
    baseline_value: float,
) -> tuple[dict[tuple[int, ...], float | np.ndarray], str, int]: ...
def aggregate_base_attributions(
    interactions: Mapping[tuple[int, ...], float | np.ndarray],
    index: str,
    order: int,
    min_order: int,
    baseline_value: float,
) -> tuple[dict[tuple[int, ...], float | np.ndarray], str, int]:
    """Aggregates the interactions into an efficient interactions.

    An example aggregation would be the transformation from `SII` values to `k-SII` values.

    Args:
        interactions: The base interaction values to aggregate. Values may be scalars or arrays
            of one value per explained instance; the aggregation is linear and applies
            element-wise to arrays.
        index: The index of the interaction values.
        order: The order of the aggregation. For example, the order of the k-SII aggregation.
        min_order: The minimum order of the base interactions. If the base interactions have a minimum
            order greater than 1, a warning is raised.
        baseline_value: The baseline value of the interaction values. For example, the baseline value
            of the SII values must not be the same as the values of the empty set.

    Returns:
        A tuple containing:
            - A dictionary mapping interactions to their values.
            - The new index of the interaction values.
            - The new minimum order of the interaction values (always 0 for this aggregation).

    Raises:
        ValueError: If the `order` is smaller than 0.
    """
    if min_order > 1:
        warnings.warn(
            UserWarning(
                "The base interaction values have a minimum order greater than 1. Aggregation may "
                "not be meaningful.",
            ),
            stacklevel=2,
        )

    transformed_interactions: dict[tuple[int, ...], float | np.ndarray] = {(): baseline_value}
    base_items = [(k, v) for k, v in interactions.items() if len(k) > 0]
    if base_items and order >= 1:
        projected = _project_onto_lower_orders(base_items, order)
        if projected is None:  # inputs the vectorized path cannot represent
            projected = _project_onto_lower_orders_loop(base_items, order)
        transformed_interactions.update(projected)

    # update the index name after the aggregation (e.g., SII -> k-SII)
    new_index = _change_index(index)
    return (
        transformed_interactions,
        new_index,
        0,
    )  # always order 0 for this aggregation


# dense accumulation buffers up to this many subset codes (32 MB of float64); larger code
# spaces go through np.unique instead
_DENSE_CODE_SPACE_LIMIT = 4_194_304


def _project_onto_lower_orders(
    base_items: list[tuple[tuple[int, ...], float | np.ndarray]],
    order: int,
) -> dict[tuple[int, ...], float | np.ndarray] | None:
    """Vectorized core of :func:`aggregate_base_attributions`.

    Projects every base interaction ``S`` onto all subsets ``T`` with ``1 <= |T| <= order``,
    weighting each contribution with ``bernoulli(|S| - |T|)``, and sums the contributions per
    subset. This replaces a Python loop over ``sum_S (2^|S| - 1)`` dict updates with a handful
    of array operations.

    Returns:
        The nonzero projected interactions, or ``None`` when the inputs need the loop fallback
        (values that do not form a numeric array, or feature ids too large to encode subsets
        into int64 codes).
    """
    bernoulli_numbers = sp.special.bernoulli(order)

    values_raw = [value for _, value in base_items]
    batched = any(isinstance(value, np.ndarray) and value.ndim > 0 for value in values_raw)
    try:
        if batched:
            values_arr = np.stack([np.asarray(value, dtype=np.float64) for value in values_raw])
        else:
            values_arr = np.asarray(values_raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if values_arr.ndim != (2 if batched else 1) or not np.issubdtype(values_arr.dtype, np.number):
        return None

    sizes = np.fromiter((len(k) for k, _ in base_items), dtype=np.int64, count=len(base_items))
    max_size = int(sizes.max())
    width = min(order, max_size)  # projected tuples have at most this many entries

    keys_by_size: dict[int, list[tuple[int, ...]]] = {}
    positions_by_size: dict[int, list[int]] = {}
    for position, (key, _) in enumerate(base_items):
        keys_by_size.setdefault(len(key), []).append(key)
        positions_by_size.setdefault(len(key), []).append(position)

    keys_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    max_feature = 0
    for s, keys_list in keys_by_size.items():
        # subset patterns of a size-s interaction in powerset order (sizes ascending,
        # positions lexicographic); unused columns point at the appended pad column s
        combo_positions: list[tuple[int, ...]] = []
        combo_scaling: list[float] = []
        combo_lengths: list[int] = []
        for t in range(1, min(s, order) + 1):
            scaling = float(bernoulli_numbers[s - t])
            for combo in combinations(range(s), t):
                combo_positions.append(combo + (s,) * (width - t))
                combo_scaling.append(scaling)
                combo_lengths.append(t)
        n_keys = len(keys_list)
        keys_arr = np.fromiter(
            chain.from_iterable(keys_list), dtype=np.int64, count=n_keys * s
        ).reshape(n_keys, s)
        keys_arr.sort(axis=1)  # powerset sorts the interaction before enumerating subsets
        max_feature = max(max_feature, int(keys_arr[:, -1].max()))
        keys_parts.append(
            (
                keys_arr,
                np.asarray(combo_positions, dtype=np.int64),
                np.asarray(combo_scaling, dtype=np.float64),
                np.asarray(combo_lengths, dtype=np.int64),
                np.asarray(positions_by_size[s], dtype=np.int64),
            )
        )

    pad_value = max_feature + 1  # fills unused columns; distinct from every feature id
    encode_base = max_feature + 2
    if float(encode_base) ** width >= 2**62:  # subset codes would overflow int64
        return None

    rows_parts: list[np.ndarray] = []
    contrib_parts: list[np.ndarray] = []
    len_parts: list[np.ndarray] = []
    for keys_arr, combo_pos_arr, scaling_arr, combo_len_arr, base_pos_arr in keys_parts:
        n_keys = len(keys_arr)
        keys_ext = np.concatenate(
            [keys_arr, np.full((n_keys, 1), pad_value, dtype=np.int64)], axis=1
        )
        rows_parts.append(keys_ext[:, combo_pos_arr].reshape(-1, width))
        if batched:
            contrib_parts.append(
                (values_arr[base_pos_arr, None, :] * scaling_arr[None, :, None]).reshape(
                    -1, values_arr.shape[1]
                )
            )
        else:
            contrib_parts.append(
                (values_arr[base_pos_arr, None] * scaling_arr[None, :]).reshape(-1)
            )
        len_parts.append(np.tile(combo_len_arr, n_keys))

    rows = np.concatenate(rows_parts)
    contribs = np.concatenate(contrib_parts)
    lengths = np.concatenate(len_parts)

    codes = np.zeros(len(rows), dtype=np.int64)
    for column in range(width):
        codes = codes * encode_base + rows[:, column]

    code_space = encode_base**width
    if not batched and code_space <= _DENSE_CODE_SPACE_LIMIT:
        # dense path: accumulate straight into the full code space, skipping the sort
        # inside np.unique
        dense_sums = np.bincount(codes, weights=contribs, minlength=code_space)
        kept_codes = np.flatnonzero(dense_sums)
        kept_values = dense_sums[kept_codes].tolist()
        # decode codes back into (padded) subset rows
        decoded = np.empty((len(kept_codes), width), dtype=np.int64)
        remaining = kept_codes
        for column in range(width - 1, -1, -1):
            decoded[:, column] = remaining % encode_base
            remaining = remaining // encode_base
        kept_lengths = (width - (decoded == pad_value).sum(axis=1)).tolist()
        kept_rows = decoded.tolist()
        return {
            tuple(row[:length]): value
            for row, length, value in zip(kept_rows, kept_lengths, kept_values, strict=True)
        }

    unique_codes, first_idx, inverse = np.unique(codes, return_index=True, return_inverse=True)
    if batched:
        sums = np.zeros((len(unique_codes), contribs.shape[1]), dtype=np.float64)
        np.add.at(sums, inverse, contribs)
        keep = sums.any(axis=1)
    else:
        sums = np.bincount(inverse, weights=contribs, minlength=len(unique_codes))
        keep = sums != 0
    # unique_codes is sorted, so iterating it directly matches the dense path's key order;
    # bulk .tolist() conversions keep the remaining Python loop to one tuple-slice per key
    kept_idx = np.flatnonzero(keep)
    kept_rows = rows[first_idx[kept_idx]].tolist()
    kept_lengths = lengths[first_idx[kept_idx]].tolist()
    if batched:
        kept_sums = sums[kept_idx]
        return {
            tuple(row[:length]): kept_sums[i]
            for i, (row, length) in enumerate(zip(kept_rows, kept_lengths, strict=True))
        }
    kept_values = sums[kept_idx].tolist()
    return {
        tuple(row[:length]): value
        for row, length, value in zip(kept_rows, kept_lengths, kept_values, strict=True)
    }


def _is_zero(value: float | np.ndarray) -> bool:
    """Whether an aggregation term is (entirely) zero.

    ``np.all(value == 0)`` would be the one-liner, but it costs a full numpy reduction dispatch
    per scalar comparison; only array values need numpy at all.
    """
    return not value.any() if isinstance(value, np.ndarray) else value == 0


def _project_onto_lower_orders_loop(
    base_items: list[tuple[tuple[int, ...], float | np.ndarray]],
    order: int,
) -> dict[tuple[int, ...], float | np.ndarray]:
    """Loop fallback of :func:`_project_onto_lower_orders` for inputs it cannot vectorize."""
    # converted to python floats once: the scaling is read once per projected subset
    bernoulli_numbers = [float(number) for number in sp.special.bernoulli(order)]
    projected: dict[tuple[int, ...], float | np.ndarray] = {}
    for base_interaction, base_interaction_value in base_items:
        base_size = len(base_interaction)
        for interaction in powerset(base_interaction, min_size=1, max_size=order):
            scaling = bernoulli_numbers[base_size - len(interaction)]
            update_interaction = scaling * base_interaction_value
            # the aggregation is linear, so values may also be arrays (one entry per explained
            # instance); the zero skip applies only when every entry is zero
            if _is_zero(update_interaction):
                continue
            projected[interaction] = projected.get(interaction, 0) + update_interaction
            # if the interactions sum to 0, we pop them from the dict
            if _is_zero(projected[interaction]):
                projected.pop(interaction)
    return projected


def aggregate_base_interaction(
    base_interactions: InteractionValues,
    order: int | None = None,
) -> InteractionValues:
    """Aggregates the basis interaction values into an efficient interaction index.

    An example aggregation would be the transformation from `SII` values to `k-SII` values.

    Args:
        base_interactions: The basis interaction values to aggregate.
        order: The order of the aggregation. For example, the order of the k-SII aggregation. If
            `None`, the maximum order of the base interactions is used. Defaults to `None`.

    Returns:
        The aggregated interaction values.

    Raises:
        ValueError: If the `order` is smaller than 0.

    Examples:
        >>> import numpy as np
        >>> from shapiq.interaction_values import InteractionValues
        >>> sii_values = InteractionValues(
        ...     n_players=3,
        ...     values=np.array([-0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        ...     index="SII",
        ...     interaction_lookup={(): 0, (1,): 1, (2,): 2, (3,): 3, (1, 2): 4, (2, 3): 5, (1, 3): 6},
        ...     baseline_value=0,  # for SII, the baseline value must not be the same as the values of emptyset
        ...     min_order=0,
        ...     max_order=2,
        ... )
        >>> k_sii_values = aggregate_base_interaction(sii_values)
        >>> k_sii_values.index
        'k-SII'
        >>> k_sii_values.baseline_value
        0
        >>> k_sii_values.interaction_lookup
        {(): 0, (1,): 1, (2,): 2, (3,): 3, (1, 2): 4, (2, 3): 5, (1, 3): 6}
        >>> k_sii_values.max_order
        2

    """
    order = order or base_interactions.max_order
    transformed_interactions, new_index, new_min_order = aggregate_base_attributions(
        interactions=base_interactions.interactions,
        index=base_interactions.index,
        order=order,
        min_order=base_interactions.min_order,
        baseline_value=float(base_interactions.baseline_value),
    )

    return InteractionValues(
        values=transformed_interactions,
        n_players=base_interactions.n_players,
        index=new_index,
        baseline_value=base_interactions.baseline_value,
        min_order=new_min_order,
        max_order=order,
        estimated=base_interactions.estimated,
        estimation_budget=base_interactions.estimation_budget,
    )


def aggregate_to_one_dimension(
    interactions: InteractionValues,
) -> tuple[np.ndarray, np.ndarray]:
    """Flattens the higher-order interaction values to positive and negative one-dimensional values.

    The aggregation summarizes all higher-order interaction in the positive and negative
    one-dimensional values for each player. The aggregation is done by distributing the interaction
    scores uniformly to all players in the interaction. For example, the interaction value 5 of
    the interaction `(1, 2)` is distributed to player 1 and player 2 as 2.5 each.

    Args:
        interactions: The interaction values to convert.

    Returns:
        The positive and negative interaction values as a 1-dimensional array for each player.

    """
    n = interactions.n_players
    pos_values = np.zeros(shape=(n,), dtype=float)
    neg_values = np.zeros(shape=(n,), dtype=float)

    for interaction in interactions.interaction_lookup:
        if len(interaction) == 0:
            continue  # skip the empty set
        interaction_value = interactions[interaction] / len(interaction)  # distribute uniformly
        for player in interaction:
            if interaction_value >= 0:
                pos_values[player] += interaction_value
            else:
                neg_values[player] += interaction_value

    return pos_values, neg_values
