"""Tests for the quadrature-based path-dependent TreeSHAP explainer."""

from __future__ import annotations

import warnings
from itertools import combinations
from math import factorial

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from shapiq.tree import QuadratureTreeSHAP, TreeExplainer, TreeSHAPIQ
from shapiq.tree.linear import LinearTreeSHAP
from shapiq.utils.sets import powerset

IMPLEMENTATIONS = ["numpy", "cpp"]


def _all_interactions(n_features: int, min_order: int, max_order: int):
    return powerset(range(n_features), min_size=min_order, max_size=max_order)


# ------------------------------- equivalence with TreeSHAP-IQ -------------------------------


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("index", "max_order"), [("SV", 1), ("SII", 1), ("SII", 2), ("SII", 3), ("k-SII", 2)]
)
def test_matches_treeshapiq_on_sklearn_regressor(
    dt_reg_model, background_reg_data, index, max_order, implementation
):
    """Quadrature values match TreeSHAP-IQ on a shallow scikit-learn regressor."""
    x = background_reg_data[0]
    reference = TreeSHAPIQ(dt_reg_model, max_order=max_order, index=index).explain(x)
    result = QuadratureTreeSHAP(
        dt_reg_model, max_order=max_order, index=index, implementation=implementation
    ).explain(x)
    assert result.baseline_value == pytest.approx(reference.baseline_value)
    n_features = background_reg_data.shape[1]
    for interaction in _all_interactions(n_features, 1, max_order):
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
    ],
)
def test_matches_treeshapiq_across_model_families(model_fixture, request):
    """Quadrature values match TreeSHAP-IQ across converter families and split conventions."""
    model = request.getfixturevalue(model_fixture)
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
def test_matches_lineartreeshap_shapley_values(dt_reg_model, background_reg_data, implementation):
    """Order-1 quadrature values match LinearTreeSHAP."""
    x = background_reg_data[0]
    reference = LinearTreeSHAP(dt_reg_model).explain_function(x)
    result = QuadratureTreeSHAP(dt_reg_model, index="SV", implementation=implementation).explain(x)
    for feature in range(background_reg_data.shape[1]):
        assert result[(feature,)] == pytest.approx(reference[(feature,)], abs=1e-10)


def test_numpy_and_cpp_agree(dt_reg_model, background_reg_data):
    """The two implementations agree to machine precision."""
    x = background_reg_data[0]
    for index, order in (("SV", 1), ("SII", 3), ("BII", 2)):
        a = QuadratureTreeSHAP(
            dt_reg_model, max_order=order, index=index, implementation="numpy"
        ).explain(x)
        b = QuadratureTreeSHAP(
            dt_reg_model, max_order=order, index=index, implementation="cpp"
        ).explain(x)
        assert np.allclose(a.values, b.values, atol=1e-12)


def test_sparse_high_feature_ids_match_lineartreeshap():
    """Attributions stay paired with the right features when set iteration is non-ascending.

    Regression test: ``{6, 14}`` iterates ``[14, 6]`` under CPython hash-slot probing, which
    previously permuted the packed values against the sorted interaction lookup (issue found
    in review; LinearTreeSHAP, which packs in the original feature space, was unaffected).
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
    """NaN instances route through the missing-value defaults identically to TreeSHAP-IQ."""
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


# ------------------------------- exact brute-force oracles -------------------------------


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


def _brute_interaction(paths, n_features, subset, *, banzhaf):
    subset = set(subset)
    order = len(subset)
    others = [f for f in range(n_features) if f not in subset]
    total = 0.0
    for size in range(len(others) + 1):
        if banzhaf:
            weight = 1.0 / 2 ** (n_features - order)
        else:
            weight = (
                factorial(size)
                * factorial(n_features - order - size)
                / factorial(n_features - order + 1)
            )
        for coalition in combinations(others, size):
            derivative = 0.0
            for included in range(order + 1):
                for part in combinations(sorted(subset), included):
                    sign = (-1) ** (order - included)
                    derivative += sign * _game_value(paths, set(coalition) | set(part))
            total += weight * derivative
    return total


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("index", ["SII", "BII"])
def test_matches_brute_force_interactions(index, implementation):
    """Quadrature SII and BII match exact brute-force enumeration on a small tree."""
    rng = np.random.default_rng(0)
    X = rng.random((400, 5))
    y = X @ rng.normal(size=5) + np.sin(9 * X[:, 0]) * X[:, 1]
    model = DecisionTreeRegressor(max_depth=5, random_state=0).fit(X, y)
    x = X[0]
    paths = _paths_of_tree(model, x)
    result = QuadratureTreeSHAP(
        model, max_order=3, index=index, implementation=implementation
    ).explain(x)
    for subset in _all_interactions(5, 1, 3):
        expected = _brute_interaction(paths, 5, subset, banzhaf=index == "BII")
        assert result[subset] == pytest.approx(expected, abs=1e-12)


# ------------------------------- deep trees (issue #545) -------------------------------


@pytest.fixture(scope="module")
def deep_sparse_tree():
    """A tree whose decision paths use ~40 distinct features: unusable for TreeSHAP-IQ."""
    rng = np.random.default_rng(7)
    X = (rng.random((8000, 80)) < 0.02).astype(float)
    y = X @ rng.normal(scale=10.0, size=80) + rng.normal(size=8000)
    model = DecisionTreeRegressor(max_depth=40, random_state=0).fit(X, y)
    return model, X


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_deep_tree_efficiency(deep_sparse_tree, implementation):
    """Shapley values satisfy efficiency on trees far beyond the polynomial explainers' limit."""
    model, X = deep_sparse_tree
    explainer = QuadratureTreeSHAP(model, index="SV", implementation=implementation)
    for row in range(3):
        x = X[row]
        prediction = model.predict(x.reshape(1, -1))[0]
        result = explainer.explain(x)
        total = sum(result[(feature,)] for feature in range(80)) + result.baseline_value
        assert total == pytest.approx(prediction, abs=1e-8)


def test_deep_tree_quadrature_point_robustness(deep_sparse_tree):
    """Results are stable under adding quadrature points (the default rule is exact)."""
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


# ------------------------------- edge cases and API -------------------------------


def test_trivial_trees():
    """Constant and single-feature trees take the trivial path."""
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


def test_dict_tree_model_input():
    """A dictionary tree representation is accepted like in TreeSHAP-IQ."""
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
    """Invalid indices, orders, and implementations are rejected."""
    with pytest.raises(ValueError, match="not supported"):
        QuadratureTreeSHAP(dt_reg_model, index="STII")
    with pytest.raises(ValueError, match="order"):
        QuadratureTreeSHAP(dt_reg_model, max_order=0)
    with pytest.raises(ValueError, match="implementation"):
        QuadratureTreeSHAP(dt_reg_model, implementation="fortran")


# ------------------------------- TreeExplainer integration -------------------------------


def test_tree_explainer_quadrature_backend(background_reg_dataset):
    """The quadrature backend matches the shapiq backend on a random forest."""
    X, y = background_reg_dataset
    model = RandomForestRegressor(n_estimators=5, max_depth=4, random_state=0).fit(X, y)
    x = X[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = TreeExplainer(model=model, index="k-SII", max_order=2, backend="shapiq")
        result = TreeExplainer(model=model, index="k-SII", max_order=2, backend="quadrature")
        explanation_ref = reference.explain(x)
        explanation_quad = result.explain(x)
    for interaction in _all_interactions(X.shape[1], 1, 2):
        assert explanation_quad[interaction] == pytest.approx(
            explanation_ref[interaction], abs=1e-8
        )


def test_tree_explainer_quadrature_backend_banzhaf(dt_reg_model, background_reg_data):
    """Path-dependent Banzhaf values work through the quadrature backend without woodelf."""
    explainer = TreeExplainer(model=dt_reg_model, index="BV", max_order=1, backend="quadrature")
    result = explainer.explain(background_reg_data[0])
    assert result.index == "BV"
    assert np.any(result.values != 0)


def test_tree_explainer_quadrature_backend_interventional_rejected(dt_reg_model):
    """The quadrature backend is path-dependent only."""
    with pytest.raises(ValueError, match="pathdependent"):
        TreeExplainer(
            model=dt_reg_model,
            mode="interventional",
            reference_dataset=np.zeros((5, 3)),
            backend="quadrature",
        )
