"""Pure-numpy reference builder of the subset hash index.

The tree kernels receive, per interaction order, a sorted table of the feature subsets that
co-occur on some decision path and a flat hash index mapping each subset to its row. Both are
built in C++ (``preprocess_subset_tables`` in ``src/shapiq/tree/cext/subset_tables.hpp``, exposed
by the interventional and the quadrature extension). This module is the independent oracle the
tests compare that index against: the same key encoding, hash and linear probing, written once
in numpy.
"""

from __future__ import annotations

import numpy as np

# Multiplier of the subset index hash; must equal kIndexHashMultiplier in
# src/shapiq/tree/cext/subset_tables.hpp (2^64 / golden ratio). The block size rule below
# (smallest power of two holding 2 * count slots) must equal block_bits there and in
# shapiq.tree.subset_layout.
INDEX_HASH_MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)


def build_subset_index(
    n_features: int,
    max_order: int,
    keys: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Builds the lookup from each subset to the row its value is stored at.

    For every order >= 2 a flat hash table is built where each slot holds a subset's key and
    row id (``-1`` marks an empty slot). A subset's key reads its sorted features as digits in
    base ``n_features``; its home slot is the top bits of ``key * INDEX_HASH_MULTIPLIER`` and
    collisions are resolved by linear probing within the order's block.

    Args:
        n_features: Number of features the ensemble splits on; subsets use ids
            ``0 .. n_features - 1``.
        max_order: Highest interaction order. Orders ``2 .. max_order`` each get their own
            block of the index; order 1 needs none, as a feature's position is its id.
        keys: The flat ``int32`` subset tables: per order, its sorted subsets back to back,
            ``order`` feature ids each.
        starts: Per order, the position in ``keys`` where its table begins.
        counts: Per order, the number of subsets in its table.

    Returns:
        ``(slot_keys, slot_rows, block_starts, block_shifts)``: each slot holds a subset's key
        and row id, and per order ``block_starts`` and ``block_shifts`` give where its slots
        begin and how many there are (``2 ** (64 - shift)``). ``None`` when there is no order
        ``>= 2`` or the keys would overflow an ``int64``.
    """
    if max_order < 2 or n_features**max_order > np.iinfo(np.int64).max:
        return None
    if int(np.max(counts)) > np.iinfo(np.int32).max:  # row ids are stored as int32
        return None
    block_starts = np.zeros(max_order + 1, dtype=np.int64)
    block_shifts = np.full(max_order + 1, 63, dtype=np.int64)
    slot_keys: list[np.ndarray] = []
    slot_rows: list[np.ndarray] = []
    position = 0
    for order in range(2, max_order + 1):
        count = int(counts[order])
        table = keys[starts[order] : starts[order] + count * order].reshape(count, order)
        row_keys = np.zeros(count, dtype=np.int64)
        for column in range(order):
            row_keys = row_keys * n_features + table[:, column]
        bits = max(1, (2 * count - 1).bit_length())  # 2**bits >= 2 * count
        size = 1 << bits
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
            claimed, first = np.unique(target[free], return_index=True)
            winners = pending[free][first]
            block_keys[claimed] = row_keys[winners]
            block_rows[claimed] = winners
            pending = np.setdiff1d(pending, winners, assume_unique=True)
            slot[pending] = (slot[pending] + 1) & (size - 1)
        block_starts[order] = position
        block_shifts[order] = 64 - bits
        slot_keys.append(block_keys)
        slot_rows.append(block_rows)
        position += size
    return np.concatenate(slot_keys), np.concatenate(slot_rows), block_starts, block_shifts
