"""Do all tree explainers route ``x`` through every model family exactly like ``predict_one``?

Efficiency (the Shapley values plus the baseline equal the prediction) fails as soon as an
explainer sends ``x`` down a different branch than the converted tree does. Besides 20
background rows, every explainer is asked about points placed exactly on a split threshold,
in float64 and in float32: that is where ``x < threshold`` and ``x <= threshold`` routing
disagree and where a float32 cast of the input flips the decision.
"""

from __future__ import annotations

import numpy as np
import pytest

from shapiq import TreeExplainer
from shapiq.tree.linear import LinearTreeSHAP
from shapiq.tree.quadrature import QuadratureTreeSHAP
from shapiq.tree.treeshapiq import TreeSHAPIQ
from shapiq.tree.validation import validate_tree_model

# (model fixture, dataset fixture)
_CASES = [
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
    ("dt_clf_model", "background_clf_dataset"),
    ("rf_clf_model", "background_clf_dataset"),
    ("gb_clf_model", "background_clf_dataset"),
    ("hist_gb_clf_model", "background_clf_dataset"),
    ("hist_gb_cat_clf_model", "background_cat_dataset"),
    ("xgb_clf_model", "background_clf_dataset"),
    ("lightgbm_clf_model", "background_clf_dataset"),
    ("catboost_clf_model", "background_clf_dataset"),
]
_PATHS = ["quadrature", "linear", "treeshapiq", "interventional"]


def _on_threshold_points(trees, x_base: np.ndarray) -> list[np.ndarray]:
    """Copies of ``x_base`` with one feature set exactly to a split threshold of the ensemble,
    in float64 and in float32 precision."""
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


def _explainer(model, X: np.ndarray, path: str):
    if path == "quadrature":
        explainer = TreeExplainer(model=model, max_order=1, min_order=1, index="SV")
        explainer._init_explainers()
        assert isinstance(explainer._pathdependent_explainer, QuadratureTreeSHAP)
        return explainer
    if path == "linear":
        try:
            return LinearTreeSHAP(model=model)
        except ValueError as error:  # trees with unreachable subtrees are refused by design
            assert "zero-cover" in str(error)
            pytest.skip("LinearTreeSHAP refuses trees with unreachable zero-cover subtrees")
    if path == "treeshapiq":
        return TreeSHAPIQ(model=model, max_order=1, index="SV")
    return TreeExplainer(
        model=model,
        mode="interventional",
        reference_dataset=X[:10],
        max_order=1,
        min_order=1,
        index="SV",
        backend="shapiq",
    )


@pytest.mark.parametrize("path", _PATHS, ids=_PATHS)
@pytest.mark.parametrize(("model_fixture", "data_fixture"), _CASES)
def test_efficiency_holds_on_every_model_family(model_fixture, data_fixture, path, request):
    """The Shapley values plus the baseline equal the converted ensemble's prediction, also on
    points lying exactly on a split threshold."""
    model = request.getfixturevalue(model_fixture)
    X = np.asarray(request.getfixturevalue(data_fixture)[0])
    explainer = _explainer(model, X, path)
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
