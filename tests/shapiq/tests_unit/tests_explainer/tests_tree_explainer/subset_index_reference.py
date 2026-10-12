"""Pure-numpy reference builder of the interaction hash index.

The tree kernels receive, per interaction order, a table of the interactions that can be
non-zero (their features occur together on some decision path), each listing its feature ids,
and a flat hash index that maps an interaction to its position in that table. Both are built in
C++ (``preprocess_subset_tables`` in ``src/shapiq/tree/cext/subset_tables.hpp``, exposed by the
interventional and the quadrature extension). This module is the independent oracle the tests
compare that index against: the same key encoding, hash and linear probing, written once in
numpy.
"""

from __future__ import annotations

from math import comb

import numpy as np

# Multiplier of the subset index hash; must equal kIndexHashMultiplier in
# src/shapiq/tree/cext/subset_tables.hpp (2^64 / golden ratio). The block size rule below
# (smallest power of two holding 2 * count slots) must equal block_bits there and in
# shapiq.tree.subset_layout.
INDEX_HASH_MULTIPLIER = 0x9E3779B97F4A7C15
_WORD = 1 << 64
_EMPTY_HI = _WORD - 1  # hi == -1 as int64 marks an empty slot


def subset_key(features: tuple[int, ...], n_features: int) -> int:
    """The key of an interaction: its ascending feature ids as digits in base ``n_features``."""
    key = 0
    for feature in features:
        key = key * n_features + int(feature)
    return key


def key_words(key: int) -> tuple[int, int]:
    """The two 64-bit words ``(hi, lo)`` of a key (``kernel: subset_tables::Key128``)."""
    return key >> 64, key & (_WORD - 1)


def key_hash(key: int) -> int:
    """The slot hash of a key (``subset_tables::key_hash``): a probe starts at its top bits."""
    hi, lo = key_words(key)
    return ((lo ^ ((hi * INDEX_HASH_MULTIPLIER) % _WORD)) * INDEX_HASH_MULTIPLIER) % _WORD


def _as_int64(word: int) -> int:
    return word - _WORD if word >= 1 << 63 else word


def build_subset_index(
    n_features: int,
    max_order: int,
    keys: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Builds the lookup from each interaction to its position in its order's table.

    For every order >= 2 a flat hash table is built where each slot holds an interaction's key
    (two int64 words, ``hi`` then ``lo``) and position (``-1`` marks an empty slot). The key
    reads the interaction's ascending feature ids as digits in base ``n_features``; its home
    slot is the top bits of :func:`key_hash` and collisions are resolved by linear probing
    within the order's block. A complete table (every possible interaction of order 2 or 3)
    is addressed by lexicographic rank by the kernels and keeps an empty block of two slots.

    Args:
        n_features: Number of features the ensemble splits on; feature ids are
            ``0 .. n_features - 1``.
        max_order: Highest interaction order. Orders ``2 .. max_order`` each get their own
            block of the index; order 1 needs none, as a feature's position is its id.
        keys: The flat ``int32`` interaction tables: per order, its interactions back to back,
            ``order`` feature ids each.
        starts: Per order, the offset in ``keys`` where its table begins.
        counts: Per order, the number of interactions in its table.

    Returns:
        ``(slot_keys, slot_rows, block_starts, block_shifts)``: ``slot_keys`` holds two words
        per slot, ``slot_rows`` the position per slot, and per order ``block_starts`` and
        ``block_shifts`` give where its slots begin and how many there are
        (``2 ** (64 - shift)``). ``None`` when there is no order ``>= 2`` or the keys would
        reach 2^127.
    """
    if max_order < 2 or n_features**max_order >= 1 << 127:
        return None
    if int(np.max(counts)) > np.iinfo(np.int32).max:  # positions are stored as int32
        return None
    block_starts = np.zeros(max_order + 1, dtype=np.int64)
    block_shifts = np.full(max_order + 1, 63, dtype=np.int64)
    slot_keys: list[np.ndarray] = []
    slot_rows: list[np.ndarray] = []
    position = 0
    for order in range(2, max_order + 1):
        count = int(counts[order])
        table = keys[starts[order] : starts[order] + count * order].reshape(count, order)
        complete = order <= 3 and count > 0 and count == comb(n_features, order)
        row_keys = (
            []
            if complete
            else [subset_key(tuple(int(v) for v in row), n_features) for row in table]
        )
        bits = 1 if complete else max(1, (2 * count - 1).bit_length())  # 2**bits >= 2 * count
        size = 1 << bits
        block_keys = np.full(2 * size, -1, dtype=np.int64)
        block_rows = np.full(size, -1, dtype=np.int32)
        for table_position, key in enumerate(
            row_keys
        ):  # linear probing, the way the kernel inserts
            slot = key_hash(key) >> (64 - bits)
            while block_rows[slot] != -1:
                slot = (slot + 1) & (size - 1)
            hi, lo = key_words(key)
            block_keys[2 * slot] = _as_int64(hi)
            block_keys[2 * slot + 1] = _as_int64(lo)
            block_rows[slot] = table_position
        block_starts[order] = position
        block_shifts[order] = 64 - bits
        slot_keys.append(block_keys)
        slot_rows.append(block_rows)
        position += size
    return np.concatenate(slot_keys), np.concatenate(slot_rows), block_starts, block_shifts
