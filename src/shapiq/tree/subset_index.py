"""Hash index over sparse subset tables, shared by the tree kernels.

A kernel that computes interactions only over the feature subsets that can be non-zero (those
co-occurring on some root-to-leaf path) keeps, per order, a sorted table of those subsets and
needs to map an enumerated subset to its row. :func:`build_subset_index` turns the tables into
flat numpy arrays the C++ side probes with one multiply-shift hash and linear probing.
"""

from __future__ import annotations

import numpy as np


def subset_table_keys(
    tables: dict[int, np.ndarray], max_order: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten per-order subset tables into the kernel layout ``(keys, starts, counts)``.

    Args:
        tables: Per order ``>= 2``, a ``(count, order)`` ``int32`` array of sorted subsets, the
            rows in lexicographic order.
        max_order: Highest order; ``starts`` and ``counts`` have ``max_order + 1`` entries.

    Returns:
        The tables back to back as one ``int32`` array, and per order where its table begins
        in that array and how many rows it has.
    """
    starts = np.zeros(max_order + 1, dtype=np.int64)
    counts = np.zeros(max_order + 1, dtype=np.int64)
    parts: list[np.ndarray] = []
    position = 0
    for order in range(2, max_order + 1):
        table = tables.get(order)
        if table is None:
            continue
        starts[order] = position
        counts[order] = table.shape[0]
        parts.append(table.reshape(-1))
        position += table.size
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
    return np.ascontiguousarray(flat, dtype=np.int32), starts, counts


# Multiplier of the subset index hash; must equal kIndexHashMultiplier in every C++ kernel
# that probes the index (2^64 / golden ratio).
INDEX_HASH_MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)


def build_subset_index(
    n_features: int,
    max_order: int,
    keys: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Builds the lookup from each subset to the output position its value is stored at.

    The C++ extension enumerates subsets along decision paths to have a fast lookup of where subset contribution belong in the output.
    This function builds a array flat hash table for every order >= 2, where each slot holds a subset's key and row id (with -1 marking an empty slot).
    It is built once per explainer.
    Its hashing must match ``merged_position`` in ``cext/quadrature_tree_shap.cc``, which reads it.

    Args:
        n_features: Number of features the ensemble splits on. Subsets use the remapped ids
            ``0 .. n_features - 1``, and each subset's key is read as a number in this base.

        max_order: Highest interaction order. Orders ``2 .. max_order`` each get their own
            block of the index; order 1 needs none, as a feature's position is its id.

        keys: The flat ``int32`` subset tables from ``_subset_table_args``: per order, its
            sorted subsets back to back, ``order`` feature ids each.

        starts: Per order, the position in ``keys`` where its table begins.

        counts: Per order, the number of subsets in its table.

    Returns:
        ``(slot_keys, slot_rows, block_starts, block_shifts)``, the arrays the kernel reads:
        each slot holds a subset's key and row id (``-1`` marks an empty slot), and per order
        ``block_starts`` and ``block_shifts`` give where its slots begin and how many there
        are (``2 ** (64 - shift)``). ``None`` when there is no order ``>= 2`` or the keys would
        overflow an ``int64``; the kernel then searches the subset tables directly.
    """
    if max_order < 2 or n_features**max_order > np.iinfo(np.int64).max:
        return None
    if int(np.max(counts)) > np.iinfo(np.int32).max:  # row ids are stored as int32
        return None
    # Build hash table flat arrays for every order >= 2.
    block_starts = np.zeros(max_order + 1, dtype=np.int64)
    block_shifts = np.full(max_order + 1, 63, dtype=np.int64)
    slot_keys: list[np.ndarray] = []
    slot_rows: list[np.ndarray] = []
    position = 0
    for order in range(2, max_order + 1):
        count = int(counts[order])
        # 2d array of the order's subsets, each row is a sorted tuple of feature ids
        table = keys[starts[order] : starts[order] + count * order].reshape(count, order)
        row_keys = np.zeros(count, dtype=np.int64)
        # Compute a unique integer key for each row by treating the row as a base-n_features
        for column in range(order):
            row_keys = row_keys * n_features + table[:, column]
        # Compute the smallest power-of-two block size that can hold all rows with at least one
        bits = max(1, (2 * count - 1).bit_length())  # 2**bits >= 2 * count
        size = 1 << bits
        # The hash of a row key is the top bits of key * INDEX_HASH_MULTIPLIER, which is uniformly distributed over the 64-bit space.
        # Standard linear probing is used to resolve collisions, wrapping within the block.
        home = (row_keys.astype(np.uint64) * INDEX_HASH_MULTIPLIER) >> np.uint64(64 - bits)
        block_keys = np.full(size, -1, dtype=np.int64)
        block_rows = np.full(size, -1, dtype=np.int32)
        slot = home.astype(np.int64)
        pending = np.arange(count)
        # linear probing for all rows at once: every pending row tries its current slot, one
        # row wins each free slot, and the others move on by one
        while pending.size:
            target = slot[pending]
            free = block_keys[target] == -1
            # Find the first free slot for each pending row
            claimed, first = np.unique(target[free], return_index=True)
            winners = pending[free][first]
            block_keys[claimed] = row_keys[winners]
            block_rows[claimed] = winners
            # The remaining rows that didn't find a free slot will try the next slot in the next iteration
            pending = np.setdiff1d(pending, winners, assume_unique=True)
            slot[pending] = (slot[pending] + 1) & (size - 1)
        block_starts[order] = position
        block_shifts[order] = 64 - bits
        slot_keys.append(block_keys)
        slot_rows.append(block_rows)
        position += size
    return np.concatenate(slot_keys), np.concatenate(slot_rows), block_starts, block_shifts
