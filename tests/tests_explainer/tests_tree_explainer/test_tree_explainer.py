"""This test module contains all tests for the tree explainer module of the shapiq package."""
import numpy as np
import pytest

from shapiq.explainer.tree import TreeExplainer


def test_decision_tree_classifier(dt_clf_model, background_clf_data):
    """Test TreeExplainer with a simple decision tree classifier."""
    explainer = TreeExplainer(model=dt_clf_model, max_order=2, min_order=1)

    x_explain = background_clf_data[0]
    explanation = explainer.explain(x_explain)

    assert type(explanation).__name__ == "InteractionValues"  # check correct return type

    # check init with class label
    _ = TreeExplainer(model=dt_clf_model, max_order=2, min_order=1, class_label=0)

    assert True

    # check with invalid output type
    with pytest.raises(NotImplementedError):
        _ = TreeExplainer(
            model=dt_clf_model, max_order=2, min_order=1, output_type="invalid_output_type"
        )


def test_decision_tree_regression(dt_reg_model, background_reg_data):
    """Test TreeExplainer with a simple decision tree regressor."""
    explainer = TreeExplainer(model=dt_reg_model, max_order=2, min_order=1)

    x_explain = background_reg_data[0]
    explanation = explainer.explain(x_explain)

    assert type(explanation).__name__ == "InteractionValues"  # check correct return type


def test_random_forrest_regression(rf_reg_model, background_reg_data):
    """Test TreeExplainer with a simple decision tree regressor."""
    explainer = TreeExplainer(model=rf_reg_model, max_order=2, min_order=1)

    x_explain = background_reg_data[0]
    explanation = explainer.explain(x_explain)

    assert type(explanation).__name__ == "InteractionValues"  # check correct return type


def test_random_forrest_classification(rf_clf_model, background_clf_data):
    """Test TreeExplainer with a simple decision tree regressor."""
    explainer = TreeExplainer(model=rf_clf_model, max_order=2, min_order=1)

    x_explain = background_clf_data[0]
    explanation = explainer.explain(x_explain)

    assert type(explanation).__name__ == "InteractionValues"  # check correct return type
