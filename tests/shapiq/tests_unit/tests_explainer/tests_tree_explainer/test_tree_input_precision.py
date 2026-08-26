"""Tests for the prediction-input precision contract of converted tree models.

XGBoost and CatBoost cast prediction inputs to float32 before comparing them against their
(float32) thresholds; LightGBM compares in float64. The converted ``TreeModel`` records this
via ``input_precision`` and rounds inputs accordingly (``cast_input``), so the explainers route
every instance to the same leaf as the source library's own prediction.
"""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.tree.quadrature import QuadratureTreeSHAP
from shapiq.tree.validation import validate_tree_model


def _uniform_regression_data(n_rows: int = 3000, n_features: int = 8) -> tuple:
    rng = np.random.default_rng(0)
    X = rng.random((n_rows, n_features))  # generic floats are not float32-exact
    y = X @ rng.normal(size=n_features) + rng.normal(scale=0.1, size=n_rows)
    return X, y


def test_xgboost_margin_agreement():
    """Converted-tree predictions match booster margins on non-float32-exact inputs."""
    xgboost = pytest.importorskip("xgboost")
    X, y = _uniform_regression_data()
    model = xgboost.XGBRegressor(
        n_estimators=30, max_depth=6, tree_method="hist", n_jobs=1, random_state=0, base_score=0.5
    ).fit(X, y)
    trees = validate_tree_model(model)
    assert all(tree.input_precision == "float32" for tree in trees)
    booster = model.get_booster()
    margins = booster.predict(xgboost.DMatrix(X[:50]), output_margin=True)
    ours = np.array([sum(tree.predict_one(x) for x in [row] for tree in trees) for row in X[:50]])
    assert np.allclose(ours, margins, atol=1e-4)


def test_xgboost_explainer_efficiency_vs_booster():
    """Quadrature explanations satisfy efficiency against the booster's own margin."""
    xgboost = pytest.importorskip("xgboost")
    X, y = _uniform_regression_data()
    model = xgboost.XGBRegressor(
        n_estimators=20, max_depth=6, tree_method="hist", n_jobs=1, random_state=0, base_score=0.5
    ).fit(X, y)
    booster = model.get_booster()
    trees = validate_tree_model(model)
    for row in range(5):
        x = X[row]
        margin = float(booster.predict(xgboost.DMatrix(x.reshape(1, -1)), output_margin=True)[0])
        total = 0.0
        for tree in trees:
            explainer = QuadratureTreeSHAP(tree, index="SV")
            iv = explainer.explain(x)
            total += sum(iv[(i,)] for i in range(X.shape[1])) + iv.baseline_value
        assert total == pytest.approx(margin, abs=1e-4)


def test_lightgbm_keeps_float64_routing():
    """LightGBM compares in float64; the cast must NOT be applied to its trees."""
    lightgbm = pytest.importorskip("lightgbm")
    X, y = _uniform_regression_data()
    model = lightgbm.LGBMRegressor(
        n_estimators=10, max_depth=4, n_jobs=1, random_state=0, verbose=-1
    ).fit(X, y)
    trees = validate_tree_model(model)
    assert all(tree.input_precision == "float64" for tree in trees)
    # instances placed one float64 ulp on either side of a real threshold must route exactly
    # like the booster — a float32 round trip would collapse the two sides
    thresholds = [t for tree in trees for t in tree.thresholds[~np.isnan(tree.thresholds)]]
    threshold = thresholds[0]
    feature = next(
        int(tree.features[i])
        for tree in trees
        for i in range(tree.n_nodes)
        if not np.isnan(tree.thresholds[i]) and tree.thresholds[i] == threshold
    )
    for side in (-np.inf, np.inf):
        x = np.full(X.shape[1], 0.5)
        x[feature] = np.nextafter(threshold, side)
        ours = sum(tree.predict_one(x) for tree in trees)
        booster_pred = float(model.predict(x.reshape(1, -1))[0])
        assert ours == pytest.approx(booster_pred, abs=1e-9)


def test_catboost_casts_input():
    """CatBoost casts inputs to float32; the converted trees must route identically."""
    catboost = pytest.importorskip("catboost")
    X, y = _uniform_regression_data(n_rows=1500, n_features=4)
    model = catboost.CatBoostRegressor(iterations=15, depth=4, verbose=0, random_seed=0).fit(X, y)
    trees = validate_tree_model(model)
    assert all(tree.input_precision == "float32" for tree in trees)
    for row in range(10):
        x = X[row]
        ours = sum(tree.predict_one(x) for tree in trees)
        assert ours == pytest.approx(float(model.predict(x.reshape(1, -1))[0]), abs=1e-6)


def test_sklearn_unaffected():
    """sklearn thresholds are float32-midpoints; float64 routing stays correct without a cast."""
    from sklearn.tree import DecisionTreeRegressor

    X, y = _uniform_regression_data()
    model = DecisionTreeRegressor(max_depth=6, random_state=0).fit(X, y)
    trees = validate_tree_model(model)
    assert all(tree.input_precision == "float64" for tree in trees)
    for row in range(20):
        assert trees[0].predict_one(X[row]) == pytest.approx(
            float(model.predict(X[row].reshape(1, -1))[0]), abs=1e-12
        )
