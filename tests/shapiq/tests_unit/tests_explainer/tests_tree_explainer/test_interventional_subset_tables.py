"""Tests for the interventional kernel's structural subset tables and their hash index."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from shapiq.tree.interventional import InterventionalTreeSHAPIQ
from shapiq.tree.interventional.cext import preprocess_subset_tables  # ty: ignore[unresolved-import]
from shapiq.tree.subset_index import INDEX_HASH_MULTIPLIER, build_subset_index


def _cooccurring_subsets_reference(trees, max_order: int) -> dict[int, set[tuple[int, ...]]]:
    """Every subset of path features (orders 2..max_order), by a plain recursive walk."""
    sets: dict[int, set[tuple[int, ...]]] = {k: set() for k in range(2, max_order + 1)}

    def walk(tree, node: int, path: frozenset[int]) -> None:
        if tree.children_left[node] == tree.children_right[node]:
            return
        feature = int(tree.features[node])
        if feature not in path:
            for order, subsets in sets.items():
                for chosen in combinations(sorted(path), order - 1):
                    subsets.add(tuple(sorted((*chosen, feature))))
        walk(tree, int(tree.children_left[node]), path | {feature})
        walk(tree, int(tree.children_right[node]), path | {feature})

    for tree in trees:
        walk(tree, 0, frozenset())
    return sets


@pytest.fixture(scope="module")
def forest_explainer():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 25))
    y = X[:, 0] * X[:, 1] + X[:, 2] + rng.normal(size=500)
    model = RandomForestRegressor(n_estimators=10, max_depth=7, random_state=0).fit(X, y)
    return InterventionalTreeSHAPIQ(model, X[:20], max_order=4, index="SII")


def test_subset_tables_hold_exactly_the_path_cooccurring_subsets(forest_explainer):
    """The C++ collector emits each co-occurring subset once, sorted, in lexicographic order."""
    ex = forest_explainer
    keys, starts, counts = ex._subset_tables
    reference = _cooccurring_subsets_reference(ex.tree, ex.max_order)
    for order in range(2, ex.max_order + 1):
        rows = keys[starts[order] : starts[order] + counts[order] * order].reshape(-1, order)
        as_tuples = [tuple(int(v) for v in row) for row in rows]
        assert set(as_tuples) == reference[order]
        assert as_tuples == sorted(reference[order])  # lexicographic, no duplicates
    assert counts[2] > 50  # precondition: a real forest, not a trivial table


def test_subset_tables_overflow_returns_none():
    """Keys that would not fit in an int64 leave the tables unbuilt (the kernel falls back)."""
    features, left, right = np.array([0, -2, -2]), np.array([1, -1, -1]), np.array([2, -1, -1])
    offsets = np.array([0, 3])
    assert preprocess_subset_tables(features, left, right, offsets, 1000, 7) is None  # 1000**7 > 2**63


def test_cpp_index_matches_the_python_builder(forest_explainer):
    """The C++ index has the blocks and (key, row) entries ``build_subset_index`` produces.

    Slot positions may differ (colliding keys are placed in a different order), so the
    comparison is per block on the set of entries, not on the arrays.
    """
    ex = forest_explainer
    keys, starts, counts = ex._subset_tables
    expected = build_subset_index(ex.n_features, ex.max_order, keys, starts, counts)
    assert expected is not None
    got_keys, got_rows, got_starts, got_shifts = ex._subset_index
    exp_keys, exp_rows, exp_starts, exp_shifts = expected
    np.testing.assert_array_equal(got_starts, exp_starts)
    np.testing.assert_array_equal(got_shifts, exp_shifts)
    for order in range(2, ex.max_order + 1):
        block = slice(int(got_starts[order]), int(got_starts[order]) + (1 << (64 - int(got_shifts[order]))))
        got = {(int(k), int(r)) for k, r in zip(got_keys[block], got_rows[block], strict=True) if k >= 0}
        exp = {(int(k), int(r)) for k, r in zip(exp_keys[block], exp_rows[block], strict=True) if k >= 0}
        assert got == exp


def test_subset_index_finds_every_row_the_way_the_kernel_probes(forest_explainer):
    """Probing the flat index as the C++ kernels do yields every table row's id."""
    ex = forest_explainer
    keys, starts, counts = ex._subset_tables
    slot_keys, slot_rows, block_starts, block_shifts = ex._subset_index
    for order in range(2, ex.max_order + 1):
        bits = 64 - int(block_shifts[order])
        size = 1 << bits
        block = slice(int(block_starts[order]), int(block_starts[order]) + size)
        block_keys, block_rows = slot_keys[block], slot_rows[block]
        rows = keys[starts[order] : starts[order] + counts[order] * order].reshape(-1, order)
        for row_id, row in enumerate(rows):
            key = 0
            for feature in row:
                key = key * ex.n_features + int(feature)
            slot = ((key * int(INDEX_HASH_MULTIPLIER)) % 2**64) >> (64 - bits)
            while block_keys[slot] != key:
                assert block_keys[slot] != -1
                slot = (slot + 1) % size
            assert block_rows[slot] == row_id
