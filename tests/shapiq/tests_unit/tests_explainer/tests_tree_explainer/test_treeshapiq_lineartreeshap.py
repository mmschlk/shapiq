"""This module contains all tests for the TreeSHAPIQ class of the shapiq package."""

from __future__ import annotations

import numpy as np

from shapiq.tree.linear import LinearTreeSHAP
from shapiq.tree.treeshapiq import TreeSHAPIQ


def test_linear_tree_shap_dt_clf(dt_clf_model, background_clf_data):
    """Test the LinearTreeSHAP explainer on a simple tree model."""

    lineartreeshap = LinearTreeSHAP(model=dt_clf_model)
    treeshapiq = TreeSHAPIQ(model=dt_clf_model, max_order=1, index="SV")
    x_explain = background_clf_data[0]
    shap_values = lineartreeshap.explain_function(x_explain)
    shap_values_treeshapiq = treeshapiq.explain(x_explain)

    for interaction, value in shap_values_treeshapiq.interactions.items():
        assert interaction in shap_values.interactions, (
            f"Interaction {interaction} is missing in LinearTreeSHAP results"
        )
        assert np.isclose(shap_values.interactions[interaction], value, atol=1e-6), (
            f"Interaction {interaction} has different values: {shap_values.interactions[interaction]} vs {value}"
        )


def test_linear_tree_shap_dt_reg(dt_reg_model, background_reg_data):
    """Test the LinearTreeSHAP explainer on a simple tree model."""

    lineartreeshap = LinearTreeSHAP(model=dt_reg_model)
    treeshapiq = TreeSHAPIQ(model=dt_reg_model, max_order=1, index="SV")
    x_explain = background_reg_data[0]
    shap_values = lineartreeshap.explain_function(x_explain)
    shap_values_treeshapiq = treeshapiq.explain(x_explain)

    for interaction, value in shap_values_treeshapiq.interactions.items():
        assert interaction in shap_values.interactions, (
            f"Interaction {interaction} is missing in LinearTreeSHAP results"
        )
        assert np.isclose(shap_values.interactions[interaction], value, atol=1e-6), (
            f"Interaction {interaction} has different values: {shap_values.interactions[interaction]} vs {value}"
        )


def test_linear_tree_shap_lgbm_clf(lightgbm_clf_model, background_clf_data):
    """Test the LinearTreeSHAP explainer on a simple tree model."""

    lineartreeshap = LinearTreeSHAP(model=lightgbm_clf_model)
    treeshapiq = TreeSHAPIQ(model=lightgbm_clf_model, max_order=1, index="SV")
    x_explain = background_clf_data[0]
    shap_values = lineartreeshap.explain_function(x_explain)
    shap_values_treeshapiq = treeshapiq.explain(x_explain)

    for interaction, value in shap_values_treeshapiq.interactions.items():
        assert interaction in shap_values.interactions, (
            f"Interaction {interaction} is missing in LinearTreeSHAP results"
        )
        assert np.isclose(shap_values.interactions[interaction], value, atol=1e-6), (
            f"Interaction {interaction} has different values: {shap_values.interactions[interaction]} vs {value}"
        )


def test_linear_tree_shap_lgbm_reg(lightgbm_reg_model, background_reg_data):
    """Test the LinearTreeSHAP explainer on a simple tree model."""

    lineartreeshap = LinearTreeSHAP(model=lightgbm_reg_model)
    treeshapiq = TreeSHAPIQ(model=lightgbm_reg_model, max_order=1, index="SV")
    x_explain = background_reg_data[0]
    shap_values = lineartreeshap.explain_function(x_explain)
    shap_values_treeshapiq = treeshapiq.explain(x_explain)

    for interaction, value in shap_values_treeshapiq.interactions.items():
        assert interaction in shap_values.interactions, (
            f"Interaction {interaction} is missing in LinearTreeSHAP results"
        )
        assert np.isclose(shap_values.interactions[interaction], value, atol=1e-6), (
            f"Interaction {interaction} has different values: {shap_values.interactions[interaction]} vs {value}"
        )


def test_linear_tree_shap_xgb_clf(xgb_clf_model, background_clf_data):
    """Test the LinearTreeSHAP explainer on a simple tree model."""

    lineartreeshap = LinearTreeSHAP(model=xgb_clf_model)
    treeshapiq = TreeSHAPIQ(model=xgb_clf_model, max_order=1, index="SV")
    x_explain = background_clf_data[0]
    shap_values = lineartreeshap.explain_function(x_explain)
    shap_values_treeshapiq = treeshapiq.explain(x_explain)

    for interaction, value in shap_values_treeshapiq.interactions.items():
        assert interaction in shap_values.interactions, (
            f"Interaction {interaction} is missing in LinearTreeSHAP results"
        )
        assert np.isclose(shap_values.interactions[interaction], value, atol=1e-6), (
            f"Interaction {interaction} has different values: {shap_values.interactions[interaction]} vs {value}"
        )


def test_linear_tree_shap_xgb_reg(xgb_reg_model, background_reg_data):
    """Test the LinearTreeSHAP explainer on a simple tree model."""

    lineartreeshap = LinearTreeSHAP(model=xgb_reg_model)
    treeshapiq = TreeSHAPIQ(model=xgb_reg_model, max_order=1, index="SV")
    x_explain = background_reg_data[0]
    shap_values = lineartreeshap.explain_function(x_explain)
    shap_values_treeshapiq = treeshapiq.explain(x_explain)

    for interaction, value in shap_values_treeshapiq.interactions.items():
        assert interaction in shap_values.interactions, (
            f"Interaction {interaction} is missing in LinearTreeSHAP results"
        )
        assert np.isclose(shap_values.interactions[interaction], value, atol=1e-6), (
            f"Interaction {interaction} has different values: {shap_values.interactions[interaction]} vs {value}"
        )
