"""This module contains all tests for the TreeExplainer class of the shapiq package."""
from shapiq.explainer.tree import TreeSHAPIQ


def test_init(dt_clf_model, background_clf_data):
    """Test the initialization of the TreeExplainer class."""
    explainer = TreeSHAPIQ(model=dt_clf_model, max_order=1, interaction_type="SII")
    x_explain = background_clf_data[0]
    explanation = explainer.explain(x_explain)
    print(explanation)

    explainer = TreeSHAPIQ(model=dt_clf_model, max_order=1, interaction_type="k-SII")
    x_explain = background_clf_data[0]
    explanation = explainer.explain(x_explain)
    print(explanation)
    assert True
