"""Output layout of the sparse tree kernels, derived from the subset tables.

Both :class:`~shapiq.tree.quadrature.computer.QuadratureTreeSHAP` and
:class:`~shapiq.tree.interventional.computer.InterventionalTreeSHAPIQ` compute interactions only
over the feature subsets that co-occur on some root-to-leaf path. The C++ routine
``preprocess_subset_tables`` (``src/shapiq/tree/cext/subset_tables.hpp``, exposed by both
extensions) collects those subsets once per explainer into ``(keys, counts)``: per order
``>= 2`` a lexicographically sorted table of ``counts[order]`` rows of ``order`` feature ids
each, all tables back to back in the ``int32`` array ``keys``. The kernels write their results
into one flat array: the order-1 block (one entry per feature, when ``min_order == 1``) followed
by one entry per table row, order by order. Where each block starts follows from ``counts``
alone and which subset a position holds from ``keys``; the functions here derive it with the
same rules as ``derive_geometry`` in the header, and the C++ ``layout_to_dict`` (exposed by both
extensions) reads a kernel's output array back into ``{feature tuple: value}``.
"""

from __future__ import annotations

import numpy as np


def table_starts(counts: np.ndarray) -> np.ndarray:
    """Per order, the position in ``keys`` where its table begins."""
    starts = np.zeros(len(counts), dtype=np.int64)
    position = 0
    for order in range(2, len(counts)):
        starts[order] = position
        position += int(counts[order]) * order
    return starts


def table_rows(keys: np.ndarray, counts: np.ndarray, order: int) -> np.ndarray:
    """The sorted subsets of one order as a ``(counts[order], order)`` array."""
    start = int(table_starts(counts)[order])
    return keys[start : start + int(counts[order]) * order].reshape(-1, order)


def block_bits(count: int) -> int:
    """log2 of the hash index block size for a table of ``count`` rows (at most half full)."""
    return max(1, (2 * count - 1).bit_length())


def block_geometry(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per order, the first slot of its hash index block and ``64 - log2(block size)``."""
    starts = np.zeros(len(counts), dtype=np.int64)
    shifts = np.full(len(counts), 63, dtype=np.int64)
    position = 0
    for order in range(2, len(counts)):
        bits = block_bits(int(counts[order]))
        starts[order] = position
        shifts[order] = 64 - bits
        position += 1 << bits
    return starts, shifts


def output_offsets(counts: np.ndarray, n_features: int, min_order: int) -> dict[int, int]:
    """Per computed order, the position in the output array where its block begins."""
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
    """Length of the output array: the order-1 block plus one entry per table row."""
    size = n_features if min_order <= 1 else 0
    return size + int(sum(int(counts[order]) for order in range(max(min_order, 2), len(counts))))
