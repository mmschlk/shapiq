"""This module contains all tests for the TreeExplainer class of the shapiq package."""
import pytest
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

from shapiq.explainer import TreeExplainer


def test_init(dt_clf_model, background_clf_data):
    """Test the initialization of the TreeExplainer class."""
    explainer = TreeExplainer(
        model=dt_clf_model,
        background_data=background_clf_data,
    )
    # TODO add more tests
    assert True
