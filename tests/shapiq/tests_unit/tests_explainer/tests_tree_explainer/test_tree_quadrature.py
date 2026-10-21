"""Does the path-dependent quadrature kernel compute the right values?

The oracles are TreeSHAP-IQ (the polynomial path-dependent explainer), linear TreeSHAP at
order 1, a brute-force enumeration of the path-dependent game on a small tree, and the numpy
twin of the C kernel (``quadrature_numpy_reference``). Layout and index tests of the kernel
live in ``test_interventional_subset_tables.py``; the two explainers share them.
"""

from __future__ import annotations

import pickle
import warnings
from itertools import combinations
from math import comb, factorial

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from shapiq.tree import QuadratureTreeSHAP, TreeExplainer, TreeModel, TreeSHAPIQ
from shapiq.tree.linear import LinearTreeSHAP
from shapiq.utils.sets import powerset
from tests.shapiq.tests_unit.tests_explainer.tests_tree_explainer.quadrature_numpy_reference import (
    numpy_explain,
)

IMPLEMENTATIONS = ["numpy", "cpp"]


def _explain(explainer, x, implementation):
    """Route through the shipped C kernel or the numpy test oracle."""
    if implementation == "numpy":
        return numpy_explain(explainer, x)
    return explainer.explain(x)


def _all_interactions(n_features: int, min_order: int, max_order: int):
    return powerset(range(n_features), min_size=min_order, max_size=max_order)


# ------------------------------- against TreeSHAP-IQ and linear TreeSHAP -------------------------------


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("index", "max_order"), [("SV", 1), ("SII", 1), ("SII", 2), ("SII", 3), ("k-SII", 2)]
)
def test_matches_treeshapiq_on_a_shallow_regressor(
    dt_reg_model, background_reg_data, index, max_order, implementation
):
    """The C kernel and its numpy twin equal TreeSHAP-IQ on every interaction up to order 3."""
    x = background_reg_data[0]
    reference = TreeSHAPIQ(dt_reg_model, max_order=max_order, index=index).explain(x)
    result = _explain(
        QuadratureTreeSHAP(dt_reg_model, max_order=max_order, index=index), x, implementation
    )
    assert result.baseline_value == pytest.approx(reference.baseline_value)
    for interaction in _all_interactions(background_reg_data.shape[1], 1, max_order):
        assert result[interaction] == pytest.approx(reference[interaction], abs=1e-10)


@pytest.mark.parametrize(
    "model_fixture",
    [
        "dt_clf_model",
        "xgb_reg_model",
        "xgb_cat_reg_model",
        "lightgbm_reg_model",
        "hist_gb_reg_model",
        "gb_reg_model",
        "catboost_clf_model",  # oblivious trees with unreachable zero-cover leaves
    ],
)
def test_matches_treeshapiq_across_model_families(model_fixture, request):
    """Every converter family (split conventions, categorical splits, zero-cover leaves) equals TreeSHAP-IQ."""
    model = request.getfixturevalue(model_fixture)
    if model_fixture == "catboost_clf_model":
        from shapiq.tree.validation import validate_tree_model

        # precondition: the model holds unreachable leaves (0/0 cover ratios in the kernel)
        assert any(
            np.any(np.asarray(tree.node_sample_weight) == 0) for tree in validate_tree_model(model)
        )
    if model_fixture == "xgb_cat_reg_model":
        data = np.asarray(request.getfixturevalue("background_cat_dataset")[0], dtype=float)
    elif "clf" in model_fixture:
        data = request.getfixturevalue("background_clf_data")
    else:
        data = request.getfixturevalue("background_reg_data")
    x = np.asarray(data[0], dtype=float)
    reference = TreeSHAPIQ(model, max_order=2, index="SII").explain(x)
    result = QuadratureTreeSHAP(model, max_order=2, index="SII").explain(x)
    assert result.baseline_value == pytest.approx(reference.baseline_value)
    for interaction in _all_interactions(x.shape[0], 1, 2):
        assert result[interaction] == pytest.approx(reference[interaction], abs=1e-8)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_order_one_matches_lineartreeshap(dt_reg_model, background_reg_data, implementation):
    """The order-1 values equal linear TreeSHAP's Shapley values."""
    x = background_reg_data[0]
    reference = LinearTreeSHAP(dt_reg_model).explain_function(x)
    result = _explain(QuadratureTreeSHAP(dt_reg_model, index="SV"), x, implementation)
    for feature in range(background_reg_data.shape[1]):
        assert result[(feature,)] == pytest.approx(reference[(feature,)], abs=1e-10)


def test_values_stay_paired_with_their_feature_ids():
    """A tree splitting on feature ids 6 and 14 only attributes to those ids.

    The explainer remaps the split features to a dense range; the set ``{6, 14}`` iterates as
    ``[14, 6]`` under CPython hashing, which once permuted the packed values against the
    sorted interaction lookup.
    """
    rng = np.random.default_rng(5)
    X = rng.random((2000, 16))
    y = 3.0 * (X[:, 6] > 0.5) + 2.0 * (X[:, 14] > 0.5)
    model = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)
    used = sorted({int(f) for f in model.tree_.feature if f >= 0})
    assert used == [6, 14]
    x = X[0]
    reference = LinearTreeSHAP(model).explain_function(x)
    for explainer in (
        QuadratureTreeSHAP(model, index="SV"),
        TreeSHAPIQ(model, max_order=1, index="SV"),
    ):
        result = explainer.explain(x)
        for feature in used:
            assert result[(feature,)] == pytest.approx(reference[(feature,)], abs=1e-10)


def test_matches_treeshapiq_with_missing_values(background_reg_dataset):
    """NaN inputs route through the missing-value children exactly like TreeSHAP-IQ."""
    xgboost = pytest.importorskip("xgboost")
    X, y = background_reg_dataset
    X = X.copy()
    rng = np.random.default_rng(0)
    X[rng.random(X.shape) < 0.2] = np.nan
    model = xgboost.XGBRegressor(n_estimators=3, max_depth=4, random_state=0).fit(X, y)
    trees = model.get_booster()
    x = X[0].copy()
    x[0] = np.nan
    reference = TreeSHAPIQ(trees, max_order=2, index="SII").explain(x)
    result = QuadratureTreeSHAP(trees, max_order=2, index="SII").explain(x)
    for interaction in _all_interactions(X.shape[1], 1, 2):
        assert result[interaction] == pytest.approx(reference[interaction], abs=1e-8)


# ------------------------------- against a brute-force enumeration -------------------------------


def _paths_of_tree(model, x):
    """Per-leaf (value, {feature: (hot, cold)}) pairs of the path-dependent game."""
    tree = model.tree_
    paths = []

    def rec(node, feats):
        if tree.children_left[node] == -1:
            paths.append((float(tree.value[node][0][0]), dict(feats)))
            return
        feature = int(tree.feature[node])
        goes_left = x[feature] <= tree.threshold[node]
        children = (
            (int(tree.children_left[node]), goes_left),
            (int(tree.children_right[node]), not goes_left),
        )
        for child, hot in children:
            cover = tree.weighted_n_node_samples[child] / tree.weighted_n_node_samples[node]
            old = feats.get(feature)
            hot_acc, cold_acc = old if old is not None else (1.0, 1.0)
            feats[feature] = (hot_acc * (1.0 if hot else 0.0), cold_acc * cover)
            rec(child, feats)
            if old is None:
                del feats[feature]
            else:
                feats[feature] = old

    rec(0, {})
    return paths


def _game_value(paths, coalition):
    total = 0.0
    for value, feats in paths:
        product = 1.0
        for feature, (hot, cold) in feats.items():
            product *= hot if feature in coalition else cold
        total += value * product
    return total


def _coalition_weight(index, n_features, order, size):
    """Weight of a coalition of ``size`` players outside an ``order``-sized subset."""
    if index == "BII":
        return 1.0 / 2 ** (n_features - order)
    if index == "Fourier":  # (-1/2)^|S| times the Banzhaf weight
        return (-0.5) ** order / 2 ** (n_features - order)
    if index == "Moebius":  # the Moebius coefficient is the discrete derivative at {}
        return 1.0 if size == 0 else 0.0
    return (
        factorial(size) * factorial(n_features - order - size) / factorial(n_features - order + 1)
    )


def _brute_interaction(paths, n_features, subset, *, index):
    subset = set(subset)
    order = len(subset)
    others = [f for f in range(n_features) if f not in subset]
    total = 0.0
    for size in range(len(others) + 1):
        weight = _coalition_weight(index, n_features, order, size)
        if weight == 0.0:
            continue
        for coalition in combinations(others, size):
            derivative = 0.0
            for included in range(order + 1):
                for part in combinations(sorted(subset), included):
                    sign = (-1) ** (order - included)
                    derivative += sign * _game_value(paths, set(coalition) | set(part))
            total += weight * derivative
    return total


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("index", ["SII", "BII", "Moebius", "Fourier"])
def test_matches_brute_force_interactions(index, implementation):
    """SII, BII, Moebius and Fourier up to order 3 equal the discrete derivatives of the enumerated game."""
    rng = np.random.default_rng(0)
    X = rng.random((400, 5))
    y = X @ rng.normal(size=5) + np.sin(9 * X[:, 0]) * X[:, 1]
    model = DecisionTreeRegressor(max_depth=5, random_state=0).fit(X, y)
    x = X[0]
    paths = _paths_of_tree(model, x)
    result = _explain(QuadratureTreeSHAP(model, max_order=3, index=index), x, implementation)
    for subset in _all_interactions(5, 1, 3):
        expected = _brute_interaction(paths, 5, subset, index=index)
        assert result[subset] == pytest.approx(expected, abs=1e-12)


# ------------------------------- deep and degenerate trees -------------------------------


@pytest.fixture(scope="module")
def deep_sparse_tree():
    """A depth-40 tree whose paths hold ~40 distinct features: unusable for TreeSHAP-IQ."""
    rng = np.random.default_rng(7)
    X = (rng.random((8000, 80)) < 0.02).astype(float)
    y = X @ rng.normal(scale=10.0, size=80) + rng.normal(size=8000)
    model = DecisionTreeRegressor(max_depth=40, random_state=0).fit(X, y)
    return model, X


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_deep_tree_efficiency(deep_sparse_tree, implementation):
    """On a depth-40 tree the Shapley values plus the baseline equal the prediction."""
    model, X = deep_sparse_tree
    explainer = QuadratureTreeSHAP(model, index="SV")
    for row in range(3):
        x = X[row]
        prediction = model.predict(x.reshape(1, -1))[0]
        result = _explain(explainer, x, implementation)
        total = sum(result[(feature,)] for feature in range(80)) + result.baseline_value
        assert total == pytest.approx(prediction, abs=1e-8)


def test_deep_tree_quadrature_rule_is_exact(deep_sparse_tree):
    """Adding quadrature points beyond the default rule does not change the values."""
    model, X = deep_sparse_tree
    x = X[0]
    default_explainer = QuadratureTreeSHAP(model, max_order=2, index="SII")
    default = default_explainer.explain(x)
    more_points = QuadratureTreeSHAP(
        model,
        max_order=2,
        index="SII",
        n_quadrature_points=default_explainer._t.shape[0] + 10,
    ).explain(x)
    assert np.allclose(default.values, more_points.values, atol=1e-10)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_zero_cover_subtree_contributes_nothing(implementation):
    """A subtree no training row reached (zero cover) that re-splits the same feature adds exactly zero.

    Inside it every cover ratio is 0/0; the kernel must not read that as full cover.
    TreeSHAP-IQ agrees and linear TreeSHAP refuses such trees instead of returning NaN.
    """
    dead_tree = {
        "children_left": np.asarray([1, -1, 3, -1, -1]),
        "children_right": np.asarray([2, -1, 4, -1, -1]),
        "children_missing": np.asarray([1, -1, 3, -1, -1]),
        "features": np.asarray([0, -2, 0, -2, -2]),
        "thresholds": np.asarray([0.0, -2, 1.0, -2, -2]),
        "node_sample_weight": np.asarray([100.0, 100.0, 0.0, 0.0, 0.0]),
        "values": np.asarray([0.0, 1.0, 0.0, 5.0, 9.0]),
    }
    explainer = QuadratureTreeSHAP(dead_tree, index="SV")
    assert explainer.empty_prediction == pytest.approx(1.0)
    # cold routing (reachable leaf): the dead region must not leak into the attribution
    cold = _explain(explainer, np.asarray([-1.0]), implementation)
    assert cold[(0,)] == pytest.approx(0.0, abs=1e-12)
    # hot routing into the dead region: prediction = 9.0, so SV = prediction - baseline
    hot = _explain(explainer, np.asarray([2.0]), implementation)
    assert hot[(0,)] == pytest.approx(8.0, abs=1e-12)

    reference = TreeSHAPIQ(dead_tree, max_order=1, index="SV")
    assert reference.explain(np.asarray([-1.0]))[(0,)] == pytest.approx(0.0, abs=1e-12)
    assert reference.explain(np.asarray([2.0]))[(0,)] == pytest.approx(8.0, abs=1e-12)
    with pytest.raises(ValueError, match="zero-cover"):
        LinearTreeSHAP(dead_tree)


# ------------------------ zero-cover nodes on the explained path ------------------------


def _path_dependent_value(trees, x: np.ndarray, coalition: set[int]) -> float:
    """Brute-force path-dependent game ``v(S)`` of an ensemble (sum over its trees).

    Follows ``x`` at splits on features in ``S`` and otherwise averages the children by cover
    (weights summing to one). A zero-cover node splits equally, the convention of the edge
    tree (:func:`~shapiq.tree.conversion.edges.create_edge_tree`).
    """

    def visit(tree, x_tree: np.ndarray, node: int) -> float:
        if tree.leaf_mask[node]:
            return float(tree.values[node])
        feature = int(tree.feature_map_internal_original[int(tree.features[node])])
        left, right = int(tree.children_left[node]), int(tree.children_right[node])
        if feature in coalition:
            child = left if tree.goes_left(node, x_tree[feature]) else right
            return visit(tree, x_tree, child)
        w_left = float(tree.node_sample_weight[left])
        w_right = float(tree.node_sample_weight[right])
        share = w_left / (w_left + w_right) if w_left + w_right > 0 else 0.5
        return share * visit(tree, x_tree, left) + (1 - share) * visit(tree, x_tree, right)

    return sum(
        visit(tree, tree.cast_input(np.asarray(x, dtype=np.float64)), int(tree.root_node_id))
        for tree in trees
    )


def _brute_force(trees, x: np.ndarray, index: str, order: int):
    """Exact interaction values of the brute-force path-dependent game."""
    from shapiq.game_theory.exact import ExactComputer

    n_players = x.shape[0]
    table = {
        subset: _path_dependent_value(trees, x, set(subset))
        for subset in powerset(range(n_players))
    }

    def game(coalitions: np.ndarray) -> np.ndarray:
        return np.array([table[tuple(np.flatnonzero(c))] for c in coalitions])

    return ExactComputer(game, n_players=n_players)(index, order)


def _assert_matches_brute_force(trees, x: np.ndarray, implementation: str = "cpp") -> None:
    """QuadratureTreeSHAP equals the brute-force game for SV and order-2 k-SII."""
    for index, order in (("SV", 1), ("k-SII", 2)):
        reference = _brute_force(trees, x, index, order)
        result = _explain(
            QuadratureTreeSHAP(trees, index=index, max_order=order), x, implementation
        )
        for interaction in _all_interactions(x.shape[0], 1, order):
            assert result[interaction] == pytest.approx(reference[interaction], abs=1e-10), (
                index,
                interaction,
            )


def _zero_cover_entered_by_x(tree, x: np.ndarray) -> bool:
    """Whether ``x``'s routing at some split selects a zero-cover internal child.

    Coalitions containing that split's feature then carry positive mass into the zero-cover
    node (cover-weighting further up only scales it), so the split ratios below it matter.
    """
    x_tree = tree.cast_input(np.asarray(x, dtype=np.float64))
    for node in np.flatnonzero(~tree.leaf_mask):
        feature = int(tree.feature_map_internal_original[int(tree.features[node])])
        go_left = tree.goes_left(int(node), x_tree[feature])
        child = int(tree.children_left[node] if go_left else tree.children_right[node])
        if not tree.leaf_mask[child] and tree.node_sample_weight[child] == 0:
            return True
    return False


def _zero_cover_on_path_tree(leaf_values=(1.0, 4.0, 10.0)) -> dict:
    """Root splits feature 0; its right child has zero cover and splits feature 1.

    ``x = [1, 5]`` (or ``x_0 = NaN``, routed right) reaches the zero-cover node, below
    which feature 1 decides between two leaves with different values.
    """
    left, dead_left, dead_right = leaf_values
    return {
        "children_left": np.asarray([1, -1, 3, -1, -1]),
        "children_right": np.asarray([2, -1, 4, -1, -1]),
        "children_missing": np.asarray([2, -1, 3, -1, -1]),
        "features": np.asarray([0, -2, 1, -2, -2]),
        "thresholds": np.asarray([0.0, -2, 0.0, -2, -2]),
        "node_sample_weight": np.asarray([100.0, 100.0, 0.0, 0.0, 0.0]),
        "values": np.asarray([0.0, left, 0.0, dead_left, dead_right]),
    }


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("x", [[1.0, 5.0], [np.nan, 5.0]], ids=["out-of-range", "nan-routing"])
def test_zero_cover_node_on_explained_path(x, implementation):
    """A zero-cover node entered by ``x``'s routing splits absent features equally.

    Regression test: the edge tree's cover ratio below a zero-cover node was ``0/0``, and
    the subtree's mass was dropped as if it were unreachable. With the equal-split
    convention ``v({0}) = (4 + 10) / 2 = 7``, ``v({}) = v({1}) = 1`` and ``v({0, 1}) = 10``,
    so ``SV = (7.5, 1.5)``; dropping the subtree gave ``(4, 5)``.
    """
    from shapiq.tree.validation import validate_tree_model

    x = np.asarray(x)
    trees = validate_tree_model(_zero_cover_on_path_tree())
    assert _zero_cover_entered_by_x(trees[0], x)
    result = _explain(QuadratureTreeSHAP(trees, index="SV"), x, implementation)
    assert result.baseline_value == pytest.approx(1.0)
    assert result[(0,)] == pytest.approx(7.5, abs=1e-12)
    assert result[(1,)] == pytest.approx(1.5, abs=1e-12)
    _assert_matches_brute_force(trees, x, implementation)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("index", ["SV", "k-SII", "BII"])
@pytest.mark.parametrize("x", [[1.0, 5.0], [np.nan, -5.0], [-1.0, 5.0]])
def test_zero_cover_constant_tree_is_null_player(x, index, implementation):
    """A constant tree attributes nothing, even when ``x``'s routing enters a zero-cover node."""
    explainer = QuadratureTreeSHAP(
        _zero_cover_on_path_tree(leaf_values=(3.0, 3.0, 3.0)), index=index, max_order=2
    )
    result = _explain(explainer, np.asarray(x), implementation)
    assert result.baseline_value == pytest.approx(3.0)
    for interaction in _all_interactions(2, 1, 2):
        assert result[interaction] == pytest.approx(0.0, abs=1e-12)


def _catboost_zero_cover_model():
    """The CatBoost regressor of the bug report: NaNs in feature 1, oblivious depth-3 trees."""
    catboost = pytest.importorskip("catboost")
    rng = np.random.default_rng(0)
    n = 400
    X = np.column_stack(
        [rng.integers(0, 6, n), rng.normal(size=n), rng.normal(size=n), rng.integers(0, 4, n)]
    ).astype(float)
    X[rng.random(n) < 0.15, 1] = np.nan
    y = (X[:, 0] % 3) + np.nan_to_num(X[:, 1]) * X[:, 2] + (X[:, 3] == 2)
    model = catboost.CatBoostRegressor(
        iterations=5, depth=3, random_seed=0, verbose=0, allow_writing_files=False
    )
    return model.fit(X, y)


_ZERO_COVER_INSTANCES = [[2.0, np.nan, 0.5, 2.0], [2.0, -1.0, 3.0, 2.0]]


@pytest.mark.parametrize("x", _ZERO_COVER_INSTANCES)
def test_catboost_zero_cover_on_path_matches_brute_force(x):
    """Every tree of the bug report's CatBoost model, and the ensemble, match brute force."""
    from shapiq.tree.validation import validate_tree_model

    x = np.asarray(x)
    trees = validate_tree_model(_catboost_zero_cover_model())
    # precondition: x's routing enters a zero-cover node (otherwise this tests nothing new)
    assert any(_zero_cover_entered_by_x(tree, x) for tree in trees)
    for tree in trees:
        _assert_matches_brute_force([tree], x)
    _assert_matches_brute_force(trees, x)
    # the default TreeExplainer path agrees
    explainer = TreeExplainer(trees, index="SV", max_order=1)
    reference = _brute_force(trees, x, "SV", 1)
    assert np.allclose(
        explainer.explain(x).get_n_order_values(1),
        reference.get_n_order_values(1),
        atol=1e-10,
    )


@pytest.mark.parametrize("x", _ZERO_COVER_INSTANCES)
def test_catboost_zero_cover_constant_tree_is_null_player(x):
    """Constant-valued copies of the CatBoost trees get all-zero Shapley values."""
    from shapiq.tree.validation import validate_tree_model

    x = np.asarray(x)
    for tree in validate_tree_model(_catboost_zero_cover_model()):
        tree.values[:] = 1.0
        values = TreeExplainer([tree], index="SV", max_order=1).explain(x)
        assert np.allclose(values.get_n_order_values(1), 0.0, atol=1e-12)


def _nan_dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    n = 300
    X = rng.normal(size=(n, 4))
    y = X[:, 0] + X[:, 1] * X[:, 2] + (X[:, 3] > 0)
    X[rng.random(n) < 0.2, 0] = np.nan
    X[rng.random(n) < 0.2, 1] = np.nan
    return X, y


@pytest.mark.parametrize("library", ["xgboost", "lightgbm", "catboost"])
def test_nan_models_match_brute_force(library):
    """No regression on NaN-routing XGBoost/LightGBM regressors and a CatBoost classifier."""
    from shapiq.tree.validation import validate_tree_model

    X, y = _nan_dataset()
    if library == "xgboost":
        xgboost = pytest.importorskip("xgboost")
        model = xgboost.XGBRegressor(n_estimators=5, max_depth=3, random_state=0).fit(X, y)
    elif library == "lightgbm":
        lightgbm = pytest.importorskip("lightgbm")
        model = lightgbm.LGBMRegressor(
            n_estimators=5, num_leaves=8, random_state=0, verbose=-1
        ).fit(X, y)
    else:
        catboost = pytest.importorskip("catboost")
        model = catboost.CatBoostClassifier(
            iterations=5, depth=3, random_seed=0, verbose=0, allow_writing_files=False
        ).fit(X, y > np.median(y))
    trees = validate_tree_model(model)
    for x in (X[0], np.array([np.nan, np.nan, 0.3, -0.2]), np.array([9.0, -9.0, 9.0, -9.0])):
        _assert_matches_brute_force(trees, x)


def test_output_holds_only_co_occurring_interactions():
    """A depth-3 tree on 200 features reports at most one triple per path and reads absent pairs as zero."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 200))
    y = X[:, :5] @ rng.normal(size=5)
    model = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)
    explainer = QuadratureTreeSHAP(model, max_order=3, index="SII")
    result = explainer.explain(X[0])
    used = sorted(int(f) for f in explainer._trees[0].feature_ids)

    n_pairs = sum(1 for k in result.interaction_lookup if len(k) == 2)
    n_triples = sum(1 for k in result.interaction_lookup if len(k) == 3)
    assert n_triples <= 8  # at most one triple per root-to-leaf path
    assert n_pairs < comb(len(used), 2)  # strictly below the dense enumeration

    missing = [pair for pair in combinations(used, 2) if pair not in result.interaction_lookup]
    assert missing  # precondition: the tree has non-co-occurring feature pairs
    assert result[missing[0]] == 0.0


def test_trivial_trees(background_reg_data):
    """Constant, single-feature and single-split trees are explained without a kernel call."""
    X = np.zeros((50, 3))
    X[:25, 1] = 1.0
    y = 2.0 * X[:, 1] + 1.0
    single_feature = DecisionTreeRegressor(max_depth=2).fit(X, y)
    x = X[0]
    result = QuadratureTreeSHAP(single_feature, index="SV").explain(x)
    prediction = single_feature.predict(x.reshape(1, -1))[0]
    assert result[(1,)] == pytest.approx(prediction - result.baseline_value)

    constant = DecisionTreeRegressor(max_depth=2).fit(X, np.ones(50))
    result = QuadratureTreeSHAP(constant, index="SV").explain(x)
    assert result.baseline_value == pytest.approx(1.0)

    # a one-split tree over a 4-feature input through the TreeExplainer front end
    stump = TreeModel(
        children_left=np.array([1, -1, -1]),
        children_right=np.array([2, -1, -1]),
        children_missing=np.array([1, -1, -1]),
        features=np.array([0, -2, -2]),
        thresholds=np.array([2.5, -2, -2]),
        values=np.array([0.5, 0.0, 1]),
        node_sample_weight=np.array([14, 5, 9]),
    )
    result = TreeExplainer(model=stump, index="SV", max_order=1).explain(np.array([2.0, 1, 1, 1]))
    assert result[(0,)] == pytest.approx(0.0 - stump.empty_prediction)


# ------------------------------- API -------------------------------


def test_explainer_pickles_after_explaining(rf_reg_model, background_reg_data):
    """A used explainer (cached kernel arguments, index included) pickles and the copy agrees."""
    x = background_reg_data[0]
    explainer = QuadratureTreeSHAP(rf_reg_model, max_order=3, index="SII")
    before = explainer.explain(x)
    assert explainer._kernel_args is not None

    restored = pickle.loads(pickle.dumps(explainer))
    after = restored.explain(x)

    np.testing.assert_array_equal(after.values, before.values)
    np.testing.assert_array_equal(explainer.explain(x).values, before.values)  # original intact


def test_dict_tree_model_input():
    """A dictionary tree is accepted and explained like TreeSHAP-IQ explains it."""
    tree_model = {
        "children_left": np.asarray([1, 2, 3, -1, -1, -1, 7, -1, -1]),
        "children_right": np.asarray([6, 5, 4, -1, -1, -1, 8, -1, -1]),
        "children_missing": np.asarray([1, 2, 3, -1, -1, -1, 7, -1, -1]),
        "features": np.asarray([0, 1, 0, -2, -2, -2, 2, -2, -2]),
        "thresholds": np.asarray([0, 0, -0.5, -2, -2, -2, 0, -2, -2]),
        "node_sample_weight": np.asarray([100, 50, 38, 15, 23, 12, 50, 20, 30]),
        "values": np.asarray([110, 105, 95, 20, 50, 100, 75, 10, 40]),
    }
    x = np.asarray([-1.0, -0.5, 1.0, 0.0])
    reference = TreeSHAPIQ(tree_model, max_order=2, index="SII").explain(x)
    result = QuadratureTreeSHAP(tree_model, max_order=2, index="SII").explain(x)
    for interaction in _all_interactions(3, 1, 2):
        assert result[interaction] == pytest.approx(reference[interaction], abs=1e-10)


def test_invalid_arguments(dt_reg_model):
    """Unsupported indices and ``max_order=0`` are rejected at construction."""
    with pytest.raises(ValueError, match="not supported"):
        QuadratureTreeSHAP(dt_reg_model, index="STII")
    with pytest.raises(ValueError, match="order"):
        QuadratureTreeSHAP(dt_reg_model, max_order=0)


def test_missing_cext_raises(dt_reg_model, monkeypatch):
    """A build without the C extension fails at construction (packaging error, no fallback)."""
    import sys

    monkeypatch.setitem(sys.modules, "shapiq.tree.quadrature.cext", None)
    with pytest.raises(ImportError, match="C extension"):
        QuadratureTreeSHAP(dt_reg_model)


# ------------------------------- TreeExplainer front end -------------------------------


def test_tree_explainer_defaults_to_quadrature(background_reg_dataset):
    """``TreeExplainer(backend="shapiq")`` explains path-dependently with the quadrature kernel
    and its k-SII values equal TreeSHAP-IQ's on a random forest."""
    X, y = background_reg_dataset
    model = RandomForestRegressor(n_estimators=5, max_depth=4, random_state=0).fit(X, y)
    x = X[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = TreeExplainer(model=model, index="k-SII", max_order=2, backend="shapiq")
        explanation = explainer.explain(x)
        assert isinstance(explainer._pathdependent_explainer, QuadratureTreeSHAP)
        reference = TreeSHAPIQ(model=model, index="k-SII", max_order=2).explain(x)
    for interaction in _all_interactions(X.shape[1], 1, 2):
        assert explanation[interaction] == pytest.approx(reference[interaction], abs=1e-8)


@pytest.mark.parametrize("index", ["BV", "BII"])
def test_tree_explainer_banzhaf_without_woodelf(dt_reg_model, background_reg_data, index):
    """Path-dependent Banzhaf indices come from the quadrature kernel, not from Woodelf."""
    max_order = 1 if index == "BV" else 2
    explainer = TreeExplainer(
        model=dt_reg_model, index=index, max_order=max_order, backend="shapiq"
    )
    x = background_reg_data[0]
    result = explainer.explain(x)
    assert result.index == index
    assert np.any(result.values != 0)
    reference = QuadratureTreeSHAP(dt_reg_model, index=index, max_order=max_order).explain(x)
    for interaction in _all_interactions(background_reg_data.shape[1], 1, max_order):
        assert result[interaction] == pytest.approx(reference[interaction], abs=1e-12)
