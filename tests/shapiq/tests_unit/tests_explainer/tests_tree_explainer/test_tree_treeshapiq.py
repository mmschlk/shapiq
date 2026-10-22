"""Do TreeSHAP-IQ and linear TreeSHAP compute the right path-dependent values?

TreeSHAP-IQ is checked against the values of the original TreeSHAP-IQ implementation on a
hand-built tree; linear TreeSHAP is checked against TreeSHAP-IQ's Shapley values on every
model family it accepts.
"""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.tree import TreeExplainer, TreeModel, TreeSHAPIQ
from shapiq.tree.linear import LinearTreeSHAP


def test_against_old_treeshapiq_implementation():
    """SII values on a hand-built 3-feature tree equal the original TreeSHAP-IQ implementation's."""
    tree_model = TreeModel(
        children_left=np.asarray([1, 2, 3, -1, -1, -1, 7, -1, -1]),
        children_right=np.asarray([6, 5, 4, -1, -1, -1, 8, -1, -1]),
        children_missing=np.asarray([1, 2, 3, -1, -1, -1, 7, -1, -1]),
        features=np.asarray([0, 1, 0, -2, -2, -2, 2, -2, -2]),
        thresholds=np.asarray([0, 0, -0.5, -2, -2, -2, 0, -2, -2]),
        node_sample_weight=np.asarray([100, 50, 38, 15, 23, 12, 50, 20, 30]),
        values=np.asarray([110, 105, 95, 20, 50, 100, 75, 10, 40]),
    )
    expected = {
        (0,): -10.18947368,
        (1,): -13.31052632,
        (2,): 3.0,
        (0, 1): -11.77894737,
        (0, 2): -6.0,
        (1, 2): 0,
    }
    explanation = TreeSHAPIQ(model=tree_model, max_order=2, index="SII").explain(
        np.asarray([-1, -0.5, 1, 0])
    )
    for key, value in expected.items():
        assert explanation[key] == pytest.approx(value, abs=1e-5)


@pytest.mark.parametrize(
    ("model_fixture", "data_fixture"),
    [
        ("dt_clf_model", "background_clf_data"),
        ("dt_reg_model", "background_reg_data"),
        ("lightgbm_clf_model", "background_clf_data"),
        ("lightgbm_reg_model", "background_reg_data"),
        ("xgb_clf_model", "background_clf_data"),
        ("xgb_reg_model", "background_reg_data"),
    ],
)
def test_linear_tree_shap_matches_treeshapiq(request, model_fixture, data_fixture):
    """Linear TreeSHAP's Shapley values equal TreeSHAP-IQ's order-1 values on every feature."""
    model = request.getfixturevalue(model_fixture)
    x = request.getfixturevalue(data_fixture)[0]
    linear = LinearTreeSHAP(model=model).explain_function(x)
    reference = TreeSHAPIQ(model=model, max_order=1, index="SV").explain(x)
    for interaction, value in reference.interactions.items():
        assert interaction in linear.interactions
        assert linear.interactions[interaction] == pytest.approx(value, abs=1e-6)


@pytest.mark.parametrize("index", ["STII", "FSII", "FBII", "BII", "BV"])
def test_invalid_arguments(dt_clf_model, index):
    """Indices TreeSHAP-IQ cannot compute and ``max_order=0`` are rejected at construction.

    ``TreeExplainer`` rejects the same indices in path-dependent mode, except the Banzhaf
    ones, which its quadrature default computes.
    """
    with pytest.raises(ValueError, match="not supported by TreeSHAP-IQ"):
        TreeSHAPIQ(model=dt_clf_model, max_order=2, index=index)
    with pytest.raises(ValueError):
        TreeSHAPIQ(model=dt_clf_model, max_order=0)
    if index in ("BII", "BV"):
        return
    with pytest.raises(ValueError, match="'pathdependent' mode"):
        TreeExplainer(model=dt_clf_model, max_order=2, index=index)
