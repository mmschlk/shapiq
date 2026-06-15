"""Efficiency-property regression tests for ``TreeExplainer`` across every supported model family.

The Shapley *efficiency* property requires that the attributions returned by
``TreeExplainer`` sum to the prediction of the model it explains. For the converted
internal representation this means::

    sum(sv.values)  ==  sum(tree.predict_one(x) for tree in explainer._trees)

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
on split thresholds, and both explainer code paths (the ``LinearTreeSHAP`` fast
path and the ``TreeSHAPIQ`` fallback) are covered.
"""

from __future__ import annotations

import numpy as np
import pytest

from shapiq import TreeExplainer

# Model families covered. Each entry maps to (model fixture, dataset fixture).
_REG_CASES = [
    ("dt_reg_model", "background_reg_dataset"),
    ("rf_reg_model", "background_reg_dataset"),
    ("xgb_reg_model", "background_reg_dataset"),
    ("lightgbm_reg_model", "background_reg_dataset"),
    ("catboost_reg_model", "background_reg_dataset"),
]
_CLF_CASES = [
    ("dt_clf_model", "background_clf_dataset"),
    ("rf_clf_model", "background_clf_dataset"),
    ("xgb_clf_model", "background_clf_dataset"),
    ("lightgbm_clf_model", "background_clf_dataset"),
    ("catboost_clf_model", "background_clf_dataset"),
]


def _on_threshold_points(explainer: TreeExplainer, x_base: np.ndarray) -> list[np.ndarray]:
    """Build explain points that land exactly on split thresholds.

    For every finite split threshold in the converted ensemble, returns a copy of
    ``x_base`` whose splitting feature is set exactly equal to that threshold (in
    both ``float64`` and ``float32`` precision). These are precisely the inputs for
    which a ``<`` vs ``<=`` routing mismatch changes the predicted leaf.
    """
    points: list[np.ndarray] = []
    for tree in explainer._trees:
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


def _assert_efficiency(model, X: np.ndarray, *, force_treeshapiq: bool) -> None:
    """Assert sum(SV) == converted-ensemble prediction for many instances and both paths."""
    if force_treeshapiq:
        original = TreeExplainer._can_use_lineartreeshap
        TreeExplainer._can_use_lineartreeshap = lambda self: False
    try:
        explainer = TreeExplainer(model=model, max_order=1, min_order=0, index="SV")
    finally:
        if force_treeshapiq:
            TreeExplainer._can_use_lineartreeshap = original

    # Sanity-check that we are actually exercising the intended code path.
    if force_treeshapiq:
        assert explainer._treeshapiq_explainers and not explainer._lineartreeshap_explainers
    else:
        assert explainer._lineartreeshap_explainers

    explain_points = [X[i] for i in range(min(20, len(X)))]
    explain_points += _on_threshold_points(explainer, X[0])

    for x in explain_points:
        ensemble_prediction = float(sum(tree.predict_one(x) for tree in explainer._trees))
        shapley_sum = float(explainer.explain(x).values.sum())
        assert shapley_sum == pytest.approx(ensemble_prediction, rel=1e-4, abs=1e-4), (
            f"Efficiency violated ({'TreeSHAPIQ' if force_treeshapiq else 'LinearTreeSHAP'} "
            f"path): sum(SV)={shapley_sum} != ensemble prediction={ensemble_prediction}"
        )


@pytest.mark.parametrize("force_treeshapiq", [False, True], ids=["linear", "treeshapiq"])
@pytest.mark.parametrize(("model_fixture", "data_fixture"), _REG_CASES)
def test_tree_explainer_efficiency_regression(
    model_fixture, data_fixture, force_treeshapiq, request
):
    """Efficiency holds for every regression model family on both explainer paths."""
    model = request.getfixturevalue(model_fixture)
    X, _ = request.getfixturevalue(data_fixture)
    _assert_efficiency(model, np.asarray(X), force_treeshapiq=force_treeshapiq)


@pytest.mark.parametrize("force_treeshapiq", [False, True], ids=["linear", "treeshapiq"])
@pytest.mark.parametrize(("model_fixture", "data_fixture"), _CLF_CASES)
def test_tree_explainer_efficiency_classification(
    model_fixture, data_fixture, force_treeshapiq, request
):
    """Efficiency holds for every classification model family on both explainer paths."""
    model = request.getfixturevalue(model_fixture)
    X, _ = request.getfixturevalue(data_fixture)
    _assert_efficiency(model, np.asarray(X), force_treeshapiq=force_treeshapiq)
