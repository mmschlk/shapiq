"""Efficiency-property regression tests for the tree explainers across every model family.

The Shapley *efficiency* property requires that the attributions returned by an explainer
sum to the prediction of the model it explains. For the converted internal representation
this means::

    sum(per-feature SV) + baseline  ==  sum(tree.predict_one(x) for tree in trees)

i.e. the explainer must route ``x`` through the tree exactly like ``predict_one``
does. This module guards two distinct routing bugs that silently broke that
invariant (both produced plausible-looking but wrong Shapley values):

* ``LinearTreeSHAP`` down-cast the explain point to ``float32`` before handing it
  to the C kernel, flipping the ``x <= threshold`` decision for instances within
  ``float32`` epsilon of a split.
* Both ``LinearTreeSHAP`` and ``TreeSHAPIQ`` hard-coded ``x <= threshold`` routing
  and ignored :attr:`TreeModel.decision_type`, so XGBoost trees (which split with
  the strict ``x < threshold`` convention) were mis-routed for any instance lying
  exactly on a split threshold.

Both edge cases are exercised explicitly by also explaining points placed exactly
on split thresholds, and all three path-dependent code paths are covered: the
``QuadratureTreeSHAP`` default of ``TreeExplainer`` plus the standalone
``LinearTreeSHAP`` and ``TreeSHAPIQ`` explainers.
"""

from __future__ import annotations

import numpy as np
import pytest

from shapiq import TreeExplainer
from shapiq.tree.linear import LinearTreeSHAP
from shapiq.tree.quadrature import QuadratureTreeSHAP
from shapiq.tree.treeshapiq import TreeSHAPIQ
from shapiq.tree.validation import validate_tree_model

# Model families covered. Each entry maps to (model fixture, dataset fixture).
_REG_CASES = [
    ("dt_reg_model", "background_reg_dataset"),
    ("rf_reg_model", "background_reg_dataset"),
    ("gb_reg_model", "background_reg_dataset"),
    ("hist_gb_reg_model", "background_reg_dataset"),
    ("hist_gb_cat_reg_model", "background_cat_dataset"),
    ("xgb_reg_model", "background_reg_dataset"),
    ("xgb_cat_reg_model", "background_cat_dataset"),
    ("lightgbm_reg_model", "background_reg_dataset"),
    ("lightgbm_cat_reg_model", "background_cat_dataset"),
    ("catboost_reg_model", "background_reg_dataset"),
]
_CLF_CASES = [
    ("dt_clf_model", "background_clf_dataset"),
    ("rf_clf_model", "background_clf_dataset"),
    ("gb_clf_model", "background_clf_dataset"),
    ("hist_gb_clf_model", "background_clf_dataset"),
    ("hist_gb_cat_clf_model", "background_cat_dataset"),
    ("xgb_clf_model", "background_clf_dataset"),
    ("lightgbm_clf_model", "background_clf_dataset"),
    ("catboost_clf_model", "background_clf_dataset"),
]


def _on_threshold_points(trees, x_base: np.ndarray) -> list[np.ndarray]:
    """Build explain points that land exactly on split thresholds.

    For every finite split threshold in the converted ensemble, returns a copy of
    ``x_base`` whose splitting feature is set exactly equal to that threshold (in
    both ``float64`` and ``float32`` precision). These are precisely the inputs for
    which a ``<`` vs ``<=`` routing mismatch changes the predicted leaf.
    """
    points: list[np.ndarray] = []
    for tree in trees:
        thresholds = np.asarray(tree.thresholds, dtype=np.float64)
        features = np.asarray(tree.features)
        for node_id in range(len(thresholds)):
            feat = int(features[node_id])
            thr = thresholds[node_id]
            if feat < 0 or not np.isfinite(thr):
                continue
            for dtype in (np.float64, np.float32):
                x = x_base.astype(np.float64).copy()
                x[feat] = np.asarray(thr, dtype=dtype).astype(np.float64)
                points.append(x)
    return points


def _assert_efficiency(model, X: np.ndarray, *, path: str) -> None:
    """Assert sum(SV) + baseline == converted-ensemble prediction on the requested path."""
    if path == "quadrature":
        explainer = TreeExplainer(model=model, max_order=1, min_order=1, index="SV")
        explainer._init_explainers()
        # Sanity-check that TreeExplainer actually defaults to the quadrature algorithm.
        assert isinstance(explainer._pathdependent_explainer, QuadratureTreeSHAP)
    elif path == "linear":
        explainer = LinearTreeSHAP(model=model)
    else:
        explainer = TreeSHAPIQ(model=model, max_order=1, index="SV")

    trees = validate_tree_model(model)
    n_features = X.shape[1]
    explain_points = [X[i] for i in range(min(20, len(X)))]
    explain_points += _on_threshold_points(trees, X[0])

    for x in explain_points:
        ensemble_prediction = float(sum(tree.predict_one(x) for tree in trees))
        explanation = explainer.explain(x)
        shapley_sum = float(
            sum(explanation[(feature,)] for feature in range(n_features))
            + explanation.baseline_value
        )
        assert shapley_sum == pytest.approx(ensemble_prediction, rel=1e-4, abs=1e-4), (
            f"Efficiency violated ({path} path): sum(SV)+baseline={shapley_sum} != "
            f"ensemble prediction={ensemble_prediction}"
        )


_PATHS = ["quadrature", "linear", "treeshapiq"]


@pytest.mark.parametrize("path", _PATHS, ids=_PATHS)
@pytest.mark.parametrize(("model_fixture", "data_fixture"), _REG_CASES)
def test_tree_explainer_efficiency_regression(model_fixture, data_fixture, path, request):
    """Efficiency holds for every regression model family on all explainer paths."""
    model = request.getfixturevalue(model_fixture)
    X, _ = request.getfixturevalue(data_fixture)
    _assert_efficiency(model, np.asarray(X), path=path)


@pytest.mark.parametrize("path", _PATHS, ids=_PATHS)
@pytest.mark.parametrize(("model_fixture", "data_fixture"), _CLF_CASES)
def test_tree_explainer_efficiency_classification(model_fixture, data_fixture, path, request):
    """Efficiency holds for every classification model family on all explainer paths."""
    model = request.getfixturevalue(model_fixture)
    X, _ = request.getfixturevalue(data_fixture)
    _assert_efficiency(model, np.asarray(X), path=path)


def test_interventional_sparse_matches_dense_categorical(
    hist_gb_cat_reg_model, background_cat_dataset
):
    """The sparse interventional C kernel routes categorical splits like the dense Python path.

    The dense path routes instances in pure Python (``TreeModel.goes_left``); the sparse path
    routes inside the C kernel. Forcing both onto the same order must produce identical
    interaction values, including on rows with NaN values.
    """
    import shapiq.tree.interventional.explainer as interventional_module
    from shapiq.tree.interventional.explainer import InterventionalTreeSHAPIQ

    X, _ = background_cat_dataset
    dense = InterventionalTreeSHAPIQ(model=hist_gb_cat_reg_model, data=X[:20], max_order=2)
    budget = interventional_module._DENSE_FLATTEN_MAX_RESULT_SIZE
    interventional_module._DENSE_FLATTEN_MAX_RESULT_SIZE = 0
    try:
        sparse = InterventionalTreeSHAPIQ(model=hist_gb_cat_reg_model, data=X[:20], max_order=2)
    finally:
        interventional_module._DENSE_FLATTEN_MAX_RESULT_SIZE = budget
    assert not dense._use_sparse_path
    assert sparse._use_sparse_path
    for i in range(5):
        dense_values = dense.explain_function(X[i]).dict_values
        sparse_values = sparse.explain_function(X[i]).dict_values
        for interaction, value in dense_values.items():
            assert value == pytest.approx(sparse_values.get(interaction, 0.0), abs=1e-5)
