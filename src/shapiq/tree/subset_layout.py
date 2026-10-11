"""Where each interaction lives in the output array of the tree kernels.

The C++ ``preprocess_subset_tables`` (``src/shapiq/tree/cext/subset_tables.hpp``, exposed by
the quadrature and the interventional extension) returns ``(keys, counts)``: for every order
``s >= 2`` the sorted table of the interactions whose features all occur on one root-to-leaf
path, ``counts[s]`` entries each listing its ``s`` feature ids in ascending order, all tables
back to back in the ``int32`` array ``keys``.

A kernel's output array is the order-1 block (one entry per feature, when ``min_order == 1``)
followed by one block per order with one entry per table entry. The *position* of an
interaction is its index within its table, which is also its offset in the order's block and
what the hash index returns for it. The helpers below mirror ``derive_geometry`` in the header.
"""

from __future__ import annotations

from math import comb

import numpy as np


def table_starts(counts: np.ndarray) -> np.ndarray:
    """Per order, the offset in ``keys`` where its table of interactions begins."""
    starts = np.zeros(len(counts), dtype=np.int64)
    position = 0
    for order in range(2, len(counts)):
        starts[order] = position
        position += int(counts[order]) * order
    return starts


def table_rows(keys: np.ndarray, counts: np.ndarray, order: int) -> np.ndarray:
    """The table of interactions of one order as a ``(counts[order], order)`` array of feature ids."""
    start = int(table_starts(counts)[order])
    return keys[start : start + int(counts[order]) * order].reshape(-1, order)


def table_is_complete(count: int, order: int, n_features: int) -> bool:
    """Whether a table holds every interaction of its order (orders 2 and 3 only).

    Positions in a complete table are lexicographic ranks, so it keeps an empty two-slot
    index block.
    """
    return order <= 3 and count > 0 and count == comb(n_features, order)


def block_bits(count: int, complete: bool = False) -> int:  # noqa: FBT001, FBT002
    """log2 of the number of hash index slots for a table of ``count`` interactions (at most half full)."""
    if complete:
        return 1
    return max(1, (2 * count - 1).bit_length())


def block_geometry(counts: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Per order, the first slot of its hash index block and ``64 - log2(block size)``."""
    starts = np.zeros(len(counts), dtype=np.int64)
    shifts = np.full(len(counts), 63, dtype=np.int64)
    position = 0
    for order in range(2, len(counts)):
        count = int(counts[order])
        bits = block_bits(count, table_is_complete(count, order, n_features))
        starts[order] = position
        shifts[order] = 64 - bits
        position += 1 << bits
    return starts, shifts


def output_offsets(counts: np.ndarray, n_features: int, min_order: int) -> dict[int, int]:
    """Per computed order, the offset in the output array where its block begins."""
    offsets: dict[int, int] = {}
    offset = 0
    if min_order <= 1:
        offsets[1] = 0
        offset = n_features
    for order in range(max(min_order, 2), len(counts)):
        offsets[order] = offset
        offset += int(counts[order])
    return offsets


def output_size(counts: np.ndarray, n_features: int, min_order: int) -> int:
    """Length of the output array: the order-1 block plus one entry per table entry."""
    size = n_features if min_order <= 1 else 0
    return size + int(sum(int(counts[order]) for order in range(max(min_order, 2), len(counts))))
