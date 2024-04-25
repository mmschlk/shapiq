"""This test module contains all tests for the tree explainer module of the shapiq package."""

import numpy as np
import pytest

from shapiq.explainer.tree import TreeExplainer, TreeModel


def test_decision_tree_classifier(dt_clf_model, background_clf_data):
    """Test TreeExplainer with a simple decision tree classifier."""
    explainer = TreeExplainer(model=dt_clf_model, max_order=2, min_order=1)

    x_explain = background_clf_data[0]
    explanation = explainer.explain(x_explain)

    assert type(explanation).__name__ == "InteractionValues"  # check correct return type

    # check init with class label
    _ = TreeExplainer(model=dt_clf_model, max_order=2, min_order=1, class_label=0)

    assert True

    explainer = _ = TreeExplainer(model=dt_clf_model, max_order=1, min_order=1, class_label=1)
    explanation = explainer.explain(x_explain)

    # compare baseline_value with empty_predictions
    assert explainer.baseline_value == sum(
        [treeshapiq.empty_prediction for treeshapiq in explainer._treeshapiq_explainers]
    )
    assert explanation.baseline_value == explainer.baseline_value


def test_decision_tree_regression(dt_reg_model, background_reg_data):
    """Test TreeExplainer with a simple decision tree regressor."""
    explainer = TreeExplainer(model=dt_reg_model, max_order=2, min_order=1)

    x_explain = background_reg_data[0]
    explanation = explainer.explain(x_explain)

    assert type(explanation).__name__ == "InteractionValues"  # check correct return type

    # compare baseline_value with empty_predictions
    assert explainer.baseline_value == sum(
        [treeshapiq.empty_prediction for treeshapiq in explainer._treeshapiq_explainers]
    )
    assert explanation.baseline_value == explainer.baseline_value


def test_random_forrest_regression(rf_reg_model, background_reg_data):
    """Test TreeExplainer with a simple decision tree regressor."""
    explainer = TreeExplainer(model=rf_reg_model, max_order=2, min_order=1)

    x_explain = background_reg_data[0]
    explanation = explainer.explain(x_explain)

    assert type(explanation).__name__ == "InteractionValues"  # check correct return type

    # compare baseline_value with empty_predictions
    assert explainer.baseline_value == sum(
        [treeshapiq.empty_prediction for treeshapiq in explainer._treeshapiq_explainers]
    )
    assert explanation.baseline_value == explainer.baseline_value


def test_random_forrest_classification(rf_clf_model, background_clf_data):
    """Test TreeExplainer with a simple decision tree regressor."""
    explainer = TreeExplainer(model=rf_clf_model, max_order=2, min_order=1)

    x_explain = background_clf_data[0]
    explanation = explainer.explain(x_explain)

    assert type(explanation).__name__ == "InteractionValues"  # check correct return type

    # compare baseline_value with empty_predictions
    assert explainer.baseline_value == sum(
        [treeshapiq.empty_prediction for treeshapiq in explainer._treeshapiq_explainers]
    )
    assert explanation.baseline_value == explainer.baseline_value


def test_against_shap_implementation():
    """Test the tree explainer against the shap implementation's tree explainer results."""
    # manual values for a tree to test against the shap implementation
    children_left = np.asarray([1, 2, 3, -1, -1, -1, 7, -1, -1])
    children_right = np.asarray([6, 5, 4, -1, -1, -1, 8, -1, -1])
    features = np.asarray([0, 1, 0, -2, -2, -2, 2, -2, -2])
    thresholds = np.asarray([0, 0, -0.5, -2, -2, -2, 0, -2, -2])
    node_sample_weight = np.asarray([100, 50, 38, 15, 23, 12, 50, 20, 30])

    # create a classification tree model
    values = [110, 105, 95, 20, 50, 100, 75, 10, 40]
    values = [values[i] / max(values) for i in range(len(values))]
    values = np.asarray(values)

    x_explain = np.asarray([-1, -0.5, 1, 0])

    tree_model = TreeModel(
        children_left=children_left,
        children_right=children_right,
        features=features,
        thresholds=thresholds,
        node_sample_weight=node_sample_weight,
        values=values,
    )

    explainer = TreeExplainer(model=tree_model, max_order=1, min_order=1, interaction_type="SII")
    explanation = explainer.explain(x_explain)

    assert explanation[(0,)] == pytest.approx(-0.09263158, abs=1e-4)
    assert explanation[(1,)] == pytest.approx(-0.12100478, abs=1e-4)
    assert explanation[(2,)] == pytest.approx(0.02727273, abs=1e-4)
    assert explanation[(3,)] == pytest.approx(0.0, abs=1e-4)

    explainer = TreeExplainer(model=tree_model, max_order=1, min_order=1, interaction_type="SII")
    explanation = explainer.explain(x_explain)

    explainer = TreeExplainer(model=tree_model, max_order=1, min_order=1, interaction_type="SII")
    explanation = explainer.explain(x_explain)
