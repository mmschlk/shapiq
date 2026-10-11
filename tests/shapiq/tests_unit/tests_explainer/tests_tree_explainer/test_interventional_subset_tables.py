"""The interaction tables, the hash index and the scale limits of the sparse tree kernels.

Both the interventional explainer and the path-dependent quadrature explainer compute only the
interactions whose features co-occur on a root-to-leaf path. ``preprocess_subset_tables``
collects them per order into a sorted table of feature ids plus a flat hash index from an
interaction to its position in its table; the kernels write one output entry per table row.
The first half of this module checks that layout, the second half that the kernels keep
working past the sizes the code has limits or special routes for (64 and 128 features on one
path, keys beyond 64 bits, 2**21 features, unlimited depth, high orders, keys beyond 127 bits).
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from shapiq.tree.interventional import InterventionalTreeSHAPIQ
from shapiq.tree.quadrature import QuadratureTreeSHAP
from shapiq.tree.subset_layout import block_geometry, table_rows, table_starts
from tests.shapiq.tests_unit.tests_explainer.tests_tree_explainer.subset_index_reference import (
    build_subset_index,
)
from tests.shapiq.tests_unit.tests_explainer.tests_tree_explainer.test_correct_calculation import (
    _average_single_reference,
)


def _chain_tree(features: list[int], seed: int = 0) -> dict:
    """A chain of splits on the given feature ids: the left child continues, the right child
    is a leaf, so the deepest path holds every feature."""
    d = len(features)
    n_nodes = 2 * d + 1
    feats = np.full(n_nodes, -2)
    left = np.full(n_nodes, -1)
    right = np.full(n_nodes, -1)
    for i, f in enumerate(features):
        feats[i] = f
        left[i] = i + 1
        right[i] = d + 1 + i
    is_leaf = feats < 0
    rng = np.random.default_rng(seed)
    return {
        "children_left": left,
        "children_right": right,
        "children_missing": left.copy(),
        "features": feats,
        "thresholds": np.where(is_leaf, -2.0, 0.0),
        "node_sample_weight": np.ones(n_nodes),
        "values": np.where(is_leaf, rng.normal(size=n_nodes), 0.0),
    }


def _perfect_tree_with_distinct_features(depth: int, seed: int = 0) -> tuple[dict, np.ndarray]:
    """A perfect binary tree whose every decision node splits on its own feature, and a point."""
    n_decision_nodes = 2**depth - 1
    node = np.arange(2 ** (depth + 1) - 1)
    is_leaf = node >= n_decision_nodes
    node_depth = np.floor(np.log2(node + 1)).astype(int)
    rng = np.random.default_rng(seed)
    tree = {
        "children_left": np.where(is_leaf, -1, 2 * node + 1),
        "children_right": np.where(is_leaf, -1, 2 * node + 2),
        "children_missing": np.where(is_leaf, -1, 2 * node + 1),
        "features": np.where(is_leaf, -2, node),
        "thresholds": np.where(is_leaf, -2.0, 0.0),
        "node_sample_weight": 2.0 ** (depth - node_depth),
        "values": np.where(is_leaf, rng.normal(size=node.size), 0.0),
    }
    return tree, rng.normal(size=n_decision_nodes)


def _assert_cohort_equals_averaged_singles(explainer, points, *, index="SII") -> None:
    """The cohort fill of ``explainer`` equals the averaged single-reference fill on ``points``."""
    for x in points:
        got = explainer.explain_function(x).dict_values
        expected = _average_single_reference(
            explainer.tree, explainer.reference_data, x, index, explainer.max_order
        )
        assert set(got) == set(expected)
        for key, value in expected.items():
            assert got[key] == pytest.approx(value, rel=1e-9, abs=1e-9)


def _assert_order_one_is_efficient(explainer, x) -> None:
    """The order-1 SII values (the Shapley values) sum to the prediction minus the baseline."""
    result = explainer.explain_function(x)
    prediction = float(sum(tree.predict_one(x) for tree in explainer.tree))
    total = sum(v for k, v in result.dict_values.items() if len(k) == 1)
    assert total == pytest.approx(prediction - result.baseline_value, rel=1e-9, abs=1e-9)


@pytest.fixture(scope="module")
def no_completion():
    """Keep the tables exactly the co-occurring interactions (no near-dense completion)."""
    from shapiq.tree.interventional import computer as interventional_module

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(interventional_module, "_STRUCTURAL_COMPLETE_ABOVE", 1.0)
        yield


@pytest.fixture(scope="module")
def forest_explainer(no_completion):
    """An order-4 explainer of a 10-tree forest on 25 features (thousands of table rows)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 25))
    y = X[:, 0] * X[:, 1] + X[:, 2] + rng.normal(size=500)
    model = RandomForestRegressor(n_estimators=10, max_depth=7, random_state=0).fit(X, y)
    return InterventionalTreeSHAPIQ(model, X[:20], max_order=4, index="SII")


# ------------------------------- layout invariants -------------------------------


def test_tables_hold_exactly_the_co_occurring_interactions(forest_explainer):
    """Each table lists every interaction whose features share a path, once, in lexicographic order."""
    ex = forest_explainer
    keys, counts = ex._subset_tables
    reference: dict[int, set[tuple[int, ...]]] = {s: set() for s in range(2, ex.max_order + 1)}

    def walk(tree, node: int, path: frozenset[int]) -> None:
        if tree.children_left[node] == tree.children_right[node]:
            return
        feature = int(tree.features[node])
        if feature not in path:
            for order, subsets in reference.items():
                for chosen in combinations(sorted(path), order - 1):
                    subsets.add(tuple(sorted((*chosen, feature))))
        walk(tree, int(tree.children_left[node]), path | {feature})
        walk(tree, int(tree.children_right[node]), path | {feature})

    for tree in ex.tree:
        walk(tree, 0, frozenset())
    assert counts[2] > 50  # precondition: a real forest, not a trivial table
    for order in range(2, ex.max_order + 1):
        rows = [tuple(int(v) for v in row) for row in table_rows(keys, counts, order)]
        assert rows == sorted(reference[order])


@pytest.mark.parametrize("layout", ["forest", "wide_chain", "quadrature"])
def test_cpp_index_equals_the_python_oracle(request, layout):
    """The hash index built in C++ equals the numpy oracle slot by slot, block geometry included.

    Three layouts: the forest (collisions in every block), a chain on 2**17 features whose
    order-4 keys all need the high 64-bit word, and a tree indexed by the quadrature extension.
    """
    if layout == "forest":
        ex = request.getfixturevalue("forest_explainer")
        n_features, max_order = ex.n_features, ex.max_order
    elif layout == "wide_chain":
        n_features, max_order = 1 << 17, 4
        tree = _chain_tree([10_000, 70_000, 100_000, 131_000, 99_999])  # every id >= 2**13
        ex = InterventionalTreeSHAPIQ(tree, np.zeros((2, n_features)), max_order=4)
    else:
        tree, _ = _perfect_tree_with_distinct_features(depth=5)
        ex = QuadratureTreeSHAP(tree, max_order=3, index="SII")
        n_features, max_order = ex._n_features_in_tree, 3
    keys, counts = ex._subset_tables
    assert ex._subset_index is not None
    got_keys, got_rows = ex._subset_index
    expected = build_subset_index(n_features, max_order, keys, table_starts(counts), counts)
    assert expected is not None
    exp_keys, exp_rows, exp_starts, exp_shifts = expected
    np.testing.assert_array_equal(got_keys, exp_keys)
    np.testing.assert_array_equal(got_rows, exp_rows)
    got_starts, got_shifts = block_geometry(counts, n_features)  # derived as the kernels derive it
    np.testing.assert_array_equal(got_starts, exp_starts)
    np.testing.assert_array_equal(got_shifts, exp_shifts)
    if layout == "wide_chain":  # only the order-4 keys exceed 64 bits: 5 slots with hi > 0
        assert counts[4] == 5
        assert int((got_keys[0::2] > 0).sum()) == 5


@pytest.mark.parametrize(
    ("max_order", "complete_above"), [(2, 0.5), (3, 0.5), (3, 0.7), (4, 0.5), (4, 0.7)]
)
def test_near_dense_tables_are_completed_and_ranked(monkeypatch, max_order, complete_above):
    """A table holding more than ``complete_above`` of all C(n, s) interactions is completed
    to all of them and the kernel positions by lexicographic rank; the values do not change.

    A forest on 8 features reaches 27 of 28 pairs and 35 of 56 triples: at 0.5 pairs and
    triples are completed, at 0.7 only the pairs, so every ranked/probed combination of the
    unrolled orders 2-3 and the recursive order 4 runs. Each must equal the probe path
    (completion disabled) and the averaged single-reference fill.
    """
    from shapiq.tree.interventional import computer as interventional_module

    rng = np.random.default_rng(5)
    X = rng.normal(size=(600, 8))
    y = X[:, 0] * X[:, 1] * X[:, 2] + np.sin(3 * X[:, 3]) * X[:, 4] + rng.normal(size=600)
    model = RandomForestRegressor(n_estimators=3, max_depth=5, random_state=0).fit(X, y)
    reference, points = X[:30], X[30:36]

    monkeypatch.setattr(interventional_module, "_STRUCTURAL_COMPLETE_ABOVE", complete_above)
    completed = InterventionalTreeSHAPIQ(model, reference, max_order=max_order, index="SII")
    keys, counts = completed._subset_tables
    expected_complete = {2: True, 3: complete_above < 35 / 56}
    for order in range(2, min(max_order, 3) + 1):
        if expected_complete[order]:
            assert counts[order] == comb(8, order)
            rows = [tuple(int(v) for v in row) for row in table_rows(keys, counts, order)]
            assert rows == list(combinations(range(8), order))
        else:
            assert counts[order] < comb(8, order)

    monkeypatch.setattr(interventional_module, "_STRUCTURAL_COMPLETE_ABOVE", 1.0)
    probed = InterventionalTreeSHAPIQ(model, reference, max_order=max_order, index="SII")
    assert probed._subset_tables[1][2] < counts[2]  # precondition: the pairs were completed
    for x in points:
        got = completed.explain_function(x).dict_values
        expected = probed.explain_function(x).dict_values
        assert set(got) == set(expected)
        for key, value in expected.items():
            assert got[key] == pytest.approx(value, rel=1e-9, abs=1e-12)
    _assert_cohort_equals_averaged_singles(completed, points[:2])


def test_both_explainers_share_the_output_layout(no_completion):
    """Interventional and quadrature build the same tables and read an output array out alike.

    For a model that splits on every feature (no feature-id remapping on the quadrature side)
    the tables coincide and ``layout_to_dict`` of either extension maps the same output entry
    to the same interaction.
    """
    from shapiq.tree.interventional.cext import layout_to_dict as interventional_readout
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
    out = np.arange(1, interventional._n_outputs + 1, dtype=np.float64)
    first = quadrature_readout(out, keys, counts, 6, 1, None, False)  # noqa: FBT003
    second = interventional_readout(out, keys, counts, 6, 1, None, False)  # noqa: FBT003
    assert first == second
    # interventional skips exact zeros and carries the baseline under (); quadrature reports
    # every interaction of the layout
    iv_i, iv_q = interventional.explain(X[0]), quadrature.explain(X[0])
    assert {k for k in iv_i.interaction_lookup if k} <= set(iv_q.interaction_lookup)


def _explainer_with_index(kind: str, index):
    """A small order-2 explainer of either kind whose kernel reads ``index`` on its next call.

    The tree is fitted on 0/1 data and explained the way ProxySHAP explains its proxy trees: the
    reference row is all zeros and the explained point all ones, so every split diverges and the
    leaf update reaches every interaction of the tables.
    """
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(300, 12)).astype(float)
    y = X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] + rng.normal(scale=0.1, size=300)
    model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)
    x = np.ones(12)
    if kind == "interventional":
        explainer = InterventionalTreeSHAPIQ(model, np.zeros((1, 12)), max_order=2)
    else:
        explainer = QuadratureTreeSHAP(model, max_order=2, index="SII")
    if index is not None:
        explainer._subset_index = index(*explainer._subset_index)
    return explainer, x


@pytest.mark.parametrize("kind", ["interventional", "quadrature"])
def test_kernels_raise_on_an_interaction_missing_from_the_index(kind):
    """An interaction the walk reaches but the index lacks raises instead of being misplaced."""

    def drop_first_entry(slot_keys, slot_rows):
        slot_keys = slot_keys.copy()
        first_used = np.flatnonzero(slot_keys[0::2] >= 0)[0]  # hi words of used slots are >= 0
        slot_keys[2 * first_used] = -1
        return slot_keys, slot_rows

    explainer, x = _explainer_with_index(kind, drop_first_entry)
    with pytest.raises(RuntimeError, match="missing from the interaction tables"):
        explainer.explain(x)


@pytest.mark.parametrize("kind", ["interventional", "quadrature"])
def test_kernels_reject_an_index_whose_blocks_exceed_its_arrays(kind):
    """Index arrays shorter than the blocks derived from ``counts`` are rejected before any probe."""
    explainer, x = _explainer_with_index(
        kind, lambda slot_keys, slot_rows: (slot_keys[:-2], slot_rows[:-1])
    )
    with pytest.raises(ValueError, match="do not match the block sizes"):
        explainer.explain(x)


# ------------------------------- scale -------------------------------


@pytest.mark.parametrize(("path_length", "max_order"), [(70, 1), (70, 3), (130, 2)])
def test_both_fills_handle_paths_with_more_than_64_features(path_length, max_order):
    """A root-to-leaf path with more than 64 or more than 128 distinct features is explained
    like any other: the cohort fill equals the averaged single-reference fill and the order-1
    values are efficient (per-leaf scratch must scale with the path, not with a constant)."""
    tree = _chain_tree(list(range(path_length)))
    rng = np.random.default_rng(1)
    reference = rng.choice([-1.0, 1.0], size=(5, path_length))
    explainer = InterventionalTreeSHAPIQ(tree, reference, max_order=max_order, index="SII")
    assert explainer._subset_tables[1][max_order] == comb(path_length, max_order) * (max_order > 1)
    points = [-np.ones(path_length), np.ones(path_length), reference[0] * -1]
    _assert_cohort_equals_averaged_singles(explainer, points)
    _assert_order_one_is_efficient(explainer, points[0])


@pytest.mark.parametrize(
    ("n_features", "features", "max_order"),
    [
        (1 << 17, [3, 10_000, 70_000, 100_000, 131_000], 4),  # order-4 keys need 68 bits
        ((1 << 21) + 1, [5, 1 << 20, (1 << 21) - 1, 1 << 21], 3),  # past the unrolled int64 route
    ],
)
def test_wide_feature_spaces_are_explained(n_features, features, max_order):
    """Feature ids up to 2**21 are indexed, probed and explained like small ones.

    Keys beyond 64 bits use the high word of the 128-bit key; beyond 2**21 features the leaf
    update leaves the unrolled orders 1-3 (int64 key arithmetic) for the general enumeration.
    """
    tree = _chain_tree(features)
    rng = np.random.default_rng(7)
    reference = np.vstack([np.ones(n_features), rng.choice([-1.0, 1.0], size=(2, n_features))])
    explainer = InterventionalTreeSHAPIQ(tree, reference, max_order=max_order, index="SII")
    assert explainer._subset_index is not None
    assert explainer._subset_tables[1][max_order] == comb(len(features), max_order)
    x = -np.ones(n_features)  # walks the chain to its end; the first row diverges at every split
    got = explainer.explain_function(x).dict_values
    assert tuple(sorted(features[:max_order])) in got  # the deepest leaf emits a top-order term
    _assert_cohort_equals_averaged_singles(explainer, [x])
    _assert_order_one_is_efficient(explainer, x)


def test_unlimited_depth_forest_on_200_features():
    """A forest grown to full depth on 200 features is explained: sparse pair table, cohort
    fill equal to the averaged single-reference fill, efficient order-1 values."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(300, 200))
    y = X[:, :10] @ rng.normal(size=10) + 3 * np.sin(X[:, 0] * X[:, 1])
    model = RandomForestRegressor(n_estimators=3, max_depth=None, random_state=0).fit(X, y)
    assert max(tree.get_depth() for tree in model.estimators_) > 10
    explainer = InterventionalTreeSHAPIQ(model, X[:5], max_order=2, index="SII")
    assert 0 < explainer._subset_tables[1][2] < comb(200, 2)
    _assert_cohort_equals_averaged_singles(explainer, X[10:13])
    _assert_order_one_is_efficient(explainer, X[10])


def test_high_order_on_shallow_trees_is_not_refused():
    """``max_order`` far above the path width stays explainable: a depth-4 forest on 30
    features at ``max_order=30`` has empty tables above order 4 and keeps its index although
    ``30 ** 30`` is far beyond the 127-bit key limit (the limit applies to the widest path)."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 30))
    y = X[:, 0] * X[:, 1] + X[:, 2]
    model = RandomForestRegressor(n_estimators=4, max_depth=4, random_state=0).fit(X, y)
    ex = InterventionalTreeSHAPIQ(model, X[:10], max_order=30, index="STII")
    assert ex._subset_index is not None
    assert ex._subset_tables[1][5:].sum() == 0
    values = ex.explain_function(X[20]).dict_values
    assert max(len(k) for k in values) <= 4


@pytest.mark.parametrize("kind", ["cohort", "single_reference", "quadrature"])
def test_fill_without_index_matches_the_indexed_fill(request, kind):
    """Called without an index, every kernel finds each interaction by searching the sorted
    tables and reproduces the indexed result exactly."""
    if kind == "quadrature":
        tree, x = _perfect_tree_with_distinct_features(depth=5)
        explainer = QuadratureTreeSHAP(tree, max_order=3, index="SII")
        indexed = explainer.explain(x)
        assert np.count_nonzero(indexed.values) > 100  # precondition: real interactions
        explainer._kernel_args = None  # the kernel arguments are cached on the first call
        explainer._subset_index = None
        np.testing.assert_allclose(explainer.explain(x).values, indexed.values, rtol=0, atol=1e-12)
        return
    forest = request.getfixturevalue("forest_explainer")
    rows = forest.reference_data[:1] if kind == "single_reference" else forest.reference_data
    explainer = InterventionalTreeSHAPIQ(forest.tree, rows, max_order=4, index="SII")
    assert (explainer.reference_data.shape[0] == 1) == (kind == "single_reference")
    x = rows[0] + 0.5
    indexed = explainer.explain_function(x).dict_values
    explainer._subset_index = None
    searched = explainer.explain_function(x).dict_values
    assert set(searched) == set(indexed)
    for key, value in indexed.items():
        assert searched[key] == pytest.approx(value, rel=1e-12, abs=1e-14)


def test_key_overflow_falls_back_to_the_table_search():
    """Orders whose keys reach 2^127 on the widest path are still explained, by searching.

    A chain of 8 splits on 2**17 features fits at order 7 (keys below 2**119) and overflows at
    order 8 (2**136): no index is built then and both fills search the sorted tables. SII
    values of one order do not depend on ``max_order``, so the searched order-8 explanation
    must agree with the indexed order-7 one up to order 7, and the cohort fill must still
    equal the averaged single-reference fill.
    """
    n_features = 1 << 17
    tree = _chain_tree([3, 70_000, 80_000, 90_000, 100_000, 110_000, 120_000, 130_000], seed=1)
    rng = np.random.default_rng(1)
    # x walks the chain to its end; the first reference row diverges at every split, so the
    # deepest leaf is reached with all eight features in E and the order-8 interaction is non-zero
    x = -np.ones(n_features)
    data = np.vstack([np.ones(n_features), rng.choice([-1.0, 1.0], size=(2, n_features))])
    indexed = InterventionalTreeSHAPIQ(tree, data, max_order=7, index="SII")
    searched = InterventionalTreeSHAPIQ(tree, data, max_order=8, index="SII")
    assert indexed._subset_index is not None
    assert searched._subset_index is None
    assert searched._subset_tables[1][8] == 1  # the chain's single 8-subset is in the tables
    by_index = indexed.explain_function(x).dict_values
    by_search = searched.explain_function(x).dict_values
    assert max(len(k) for k in by_search) == 8
    assert {k for k in by_search if len(k) <= 7} == set(by_index)
    for key, value in by_index.items():
        assert by_search[key] == pytest.approx(value, rel=1e-9, abs=1e-12)
    _assert_cohort_equals_averaged_singles(searched, [x])
