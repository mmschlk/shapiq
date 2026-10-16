"""Tests for the numerical precision guards of the polynomial tree explainers (issue #545)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from shapiq.tree import (
    LinearTreeSHAP,
    QuadratureTreeSHAP,
    TreeNumericalPrecisionError,
    TreeNumericalPrecisionWarning,
    TreeSHAPIQ,
)
from shapiq.tree.explainer import TreeExplainer
from shapiq.tree.precision import (
    ERROR_FEATURES_PER_PATH,
    WARN_FEATURES_PER_PATH,
    check_features_per_path,
)


def _sparse_indicator_tree(max_depth: int) -> tuple[DecisionTreeRegressor, np.ndarray]:
    """A tree whose decision paths use one new indicator feature per level."""
    rng = np.random.default_rng(7)
    X = (rng.random((8000, 80)) < 0.02).astype(float)
    y = X @ rng.normal(scale=10.0, size=80) + rng.normal(size=8000)
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=0).fit(X, y)
    # the guard tests below rely on the fitted tree actually reaching the intended band;
    # make that precondition explicit so a future sklearn change fails loudly here
    assert QuadratureTreeSHAP(model).max_features_per_path >= max_depth
    return model, X


def test_check_features_per_path_bands():
    """The guard is silent, warns, and raises in the documented bands."""
    check_features_per_path(WARN_FEATURES_PER_PATH - 1, algorithm="TreeSHAP-IQ")  # no warning

    with pytest.warns(TreeNumericalPrecisionWarning, match="distinct features"):
        check_features_per_path(WARN_FEATURES_PER_PATH, algorithm="TreeSHAP-IQ")

    with pytest.raises(TreeNumericalPrecisionError, match="Quadrature"):
        check_features_per_path(ERROR_FEATURES_PER_PATH, algorithm="TreeSHAP-IQ")


@pytest.mark.parametrize("explainer_cls", [LinearTreeSHAP, TreeSHAPIQ])
def test_explainers_guard_unreliable_trees(explainer_cls):
    """Both polynomial explainers refuse trees beyond the reliability limit."""
    model, _ = _sparse_indicator_tree(max_depth=ERROR_FEATURES_PER_PATH + 5)
    with pytest.raises(TreeNumericalPrecisionError, match="distinct features"):
        explainer_cls(model)


@pytest.mark.parametrize("explainer_cls", [LinearTreeSHAP, TreeSHAPIQ])
def test_explainers_warn_in_degrading_band(explainer_cls):
    """Both polynomial explainers warn inside the degrading precision band."""
    model, _ = _sparse_indicator_tree(max_depth=WARN_FEATURES_PER_PATH + 1)
    with pytest.warns(TreeNumericalPrecisionWarning):
        explainer_cls(model)


def test_tree_explainer_computes_trees_the_guard_refuses():
    """``TreeExplainer`` computes trees beyond the polynomial explainers' reliability limit.

    Its default quadrature algorithm is exact regardless of the number of distinct features
    per path, so the guard never fires through the front end and efficiency holds.
    """
    model, X = _sparse_indicator_tree(max_depth=ERROR_FEATURES_PER_PATH + 5)
    explainer = TreeExplainer(model=model, index="SV", min_order=1, backend="shapiq")
    x = X[0]
    explanation = explainer.explain(x)
    prediction = model.predict(x.reshape(1, -1))[0]
    total = explanation.values.sum() + explanation.baseline_value
    assert total == pytest.approx(prediction, abs=1e-8)


def test_deep_tree_over_few_features_is_fine():
    """Depth alone does not trip the guard: only distinct features per path count.

    Such trees previously risked ``LinAlgError`` from unused interpolation rows; they must
    now construct and satisfy the efficiency property.
    """
    rng = np.random.default_rng(3)
    X = rng.random((10000, 6))
    y = np.sin(5 * X).sum(axis=1) + rng.normal(scale=0.05, size=10000)
    model = DecisionTreeRegressor(max_depth=45, random_state=0).fit(X, y)
    assert model.get_depth() >= ERROR_FEATURES_PER_PATH

    x = X[0]
    prediction = model.predict(x.reshape(1, -1))[0]

    sv = LinearTreeSHAP(model).explain_function(x)
    assert sv.values.sum() == pytest.approx(prediction, abs=1e-5)

    iv = TreeSHAPIQ(model, max_order=1, index="SV").explain(x)
    total = sum(iv[(feature,)] for feature in range(6)) + iv.baseline_value
    assert total == pytest.approx(prediction, abs=1e-8)
