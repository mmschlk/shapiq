"""Tests for the interventional kernel's structural subset tables and their hash index."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from shapiq.tree.interventional import InterventionalTreeSHAPIQ
from shapiq.tree.interventional.cext import (
    preprocess_subset_tables,  # ty: ignore[unresolved-import]
)
from shapiq.tree.subset_layout import block_geometry, table_rows, table_starts
from tests.shapiq.tests_unit.tests_explainer.tests_tree_explainer.subset_index_reference import (
    INDEX_HASH_MULTIPLIER,
    build_subset_index,
)


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
    keys, counts = ex._subset_tables
    reference = _cooccurring_subsets_reference(ex.tree, ex.max_order)
    for order in range(2, ex.max_order + 1):
        rows = table_rows(keys, counts, order)
        as_tuples = [tuple(int(v) for v in row) for row in rows]
        assert set(as_tuples) == reference[order]
        assert as_tuples == sorted(reference[order])  # lexicographic, no duplicates
    assert counts[2] > 50  # precondition: a real forest, not a trivial table


def test_subset_tables_without_index_when_keys_overflow():
    """Keys that would not fit in an int64 leave the index unbuilt; the tables are still collected.

    The interventional explainer takes its sparse route then, the quadrature kernel searches
    the tables by tuple comparison.
    """
    # a perfect depth-2 tree splitting on features 0, 1 and 2: the pairs (0, 1) and (0, 2)
    features = np.array([0, 1, 2, -2, -2, -2, -2])
    left = np.array([1, 3, 5, -1, -1, -1, -1])
    right = np.array([2, 4, 6, -1, -1, -1, -1])
    offsets = np.array([0, 7])
    (keys, counts), index = preprocess_subset_tables(
        features, left, right, offsets, 1000, 7
    )  # 1000**7 > 2**63
    assert index is None
    np.testing.assert_array_equal(counts, [0, 0, 2, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(table_rows(keys, counts, 2), [[0, 1], [0, 2]])


def test_cpp_index_matches_the_python_builder(forest_explainer):
    """The C++ index has the blocks and (key, row) entries ``build_subset_index`` produces.

    Slot positions may differ (colliding keys are placed in a different order), so the
    comparison is per block on the set of entries, not on the arrays.
    """
    ex = forest_explainer
    keys, counts = ex._subset_tables
    starts = table_starts(counts)
    expected = build_subset_index(ex.n_features, ex.max_order, keys, starts, counts)
    assert expected is not None
    got_keys, got_rows = ex._subset_index
    got_starts, got_shifts = block_geometry(counts)  # derived, as the kernels derive it
    exp_keys, exp_rows, exp_starts, exp_shifts = expected
    np.testing.assert_array_equal(got_starts, exp_starts)
    np.testing.assert_array_equal(got_shifts, exp_shifts)
    for order in range(2, ex.max_order + 1):
        block = slice(
            int(got_starts[order]), int(got_starts[order]) + (1 << (64 - int(got_shifts[order])))
        )
        got = {
            (int(k), int(r))
            for k, r in zip(got_keys[block], got_rows[block], strict=True)
            if k >= 0
        }
        exp = {
            (int(k), int(r))
            for k, r in zip(exp_keys[block], exp_rows[block], strict=True)
            if k >= 0
        }
        assert got == exp


def test_subset_index_finds_every_row_the_way_the_kernel_probes(forest_explainer):
    """Probing the flat index as the C++ kernels do yields every table row's id."""
    ex = forest_explainer
    keys, counts = ex._subset_tables
    slot_keys, slot_rows = ex._subset_index
    block_starts, block_shifts = block_geometry(counts)
    for order in range(2, ex.max_order + 1):
        bits = 64 - int(block_shifts[order])
        size = 1 << bits
        block = slice(int(block_starts[order]), int(block_starts[order]) + size)
        block_keys, block_rows = slot_keys[block], slot_rows[block]
        rows = table_rows(keys, counts, order)
        for row_id, row in enumerate(rows):
            key = 0
            for feature in row:
                key = key * ex.n_features + int(feature)
            slot = ((key * int(INDEX_HASH_MULTIPLIER)) % 2**64) >> (64 - bits)
            while block_keys[slot] != key:
                assert block_keys[slot] != -1
                slot = (slot + 1) % size
            assert block_rows[slot] == row_id


def test_both_explainers_share_the_output_layout():
    """Quadrature and interventional build the same tables and read their arrays out alike.

    Both explainers run `preprocess_subset_tables` on the same trees, derive the layout with
    `shapiq.tree.subset_layout` and read the kernel's array back with `layout_to_dict`, each
    through its own extension module. For a model that splits on every feature (no feature-id
    remapping on the quadrature side) the tables and the readout order must coincide.
    """
    from shapiq.tree.interventional.cext import layout_to_dict as interventional_readout
    from shapiq.tree.quadrature import QuadratureTreeSHAP
    from shapiq.tree.quadrature.cext import layout_to_dict as quadrature_readout

    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 6))
    y = X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] * X[:, 4] + X[:, 5] + rng.normal(size=400)
    model = RandomForestRegressor(n_estimators=5, max_depth=6, random_state=0).fit(X, y)
    interventional = InterventionalTreeSHAPIQ(model, X[:10], max_order=3, index="SII")
    quadrature = QuadratureTreeSHAP(model, max_order=3, index="SII")
    assert quadrature._n_features_in_tree == 6  # precondition: no remapping
    for got, expected in zip(interventional._subset_tables, quadrature._subset_tables, strict=True):
        np.testing.assert_array_equal(got, expected)
    keys, counts = interventional._subset_tables
    out = np.arange(1, interventional._n_structural_interactions + 1, dtype=np.float64)
    readouts = [quadrature_readout, interventional_readout]
    first, second = (read(out, keys, counts, 6, 1, None, False) for read in readouts)  # noqa: FBT003
    assert first == second
    # the explained results agree on every interaction they both report
    iv_i, iv_q = interventional.explain(X[0]), quadrature.explain(X[0])
    # interventional skips exact zeros and carries the baseline under (); quadrature reports
    # every interaction of the layout
    assert {k for k in iv_i.interaction_lookup if k} <= set(iv_q.interaction_lookup)
