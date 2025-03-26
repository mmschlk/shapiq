"""This test module tests the TabPFNExplainer object."""

import pytest

from shapiq import Explainer, InteractionValues, TabPFNExplainer, TabularExplainer
from tests.markers import skip_if_no_tabpfn


@skip_if_no_tabpfn
@pytest.mark.external_libraries
def test_tabpfn_explainer_clf(tabpfn_classification_problem):
    """Test the TabPFNExplainer class for classification problems."""
    import tabpfn

    # setup
    model, data, labels, x_test = tabpfn_classification_problem
    x_explain = x_test[0]
    assert isinstance(model, tabpfn.TabPFNClassifier)
    if model.n_features_in_ == data.shape[1]:
        model.fit(data, labels)
    assert model.n_features_in_ == data.shape[1]

    explainer = TabPFNExplainer(model=model, data=data, labels=labels, x_test=x_test)
    explanation = explainer.explain(x=x_explain)
    assert isinstance(explanation, InteractionValues)

    # test that bare explainer gets turned into TabPFNExplainer
    explainer = Explainer(model=model, data=data, labels=labels, x_test=x_test)
    assert isinstance(explainer, TabPFNExplainer)

    # test that TabularExplainer works as well
    with pytest.warns(UserWarning):
        explainer = TabularExplainer(model=model, data=data, class_index=1, imputer="baseline")
        assert isinstance(explainer, TabularExplainer)


@skip_if_no_tabpfn
@pytest.mark.external_libraries
def test_tabpfn_explainer_reg(tabpfn_regression_problem):
    """Test the TabPFNExplainer class for regression problems."""
    import tabpfn

    # setup
    model, data, labels, x_test = tabpfn_regression_problem
    x_explain = x_test[0]
    assert isinstance(model, tabpfn.TabPFNRegressor)
    if model.n_features_in_ == data.shape[1]:
        model.fit(data, labels)
    assert model.n_features_in_ == data.shape[1]

    explainer = TabPFNExplainer(model=model, data=data, labels=labels, x_test=x_test)
    explanation = explainer.explain(x=x_explain)
    assert isinstance(explanation, InteractionValues)

    # test that bare explainer gets turned into TabPFNExplainer
    explainer = Explainer(model=model, data=data, labels=labels, x_test=x_test)
    assert isinstance(explainer, TabPFNExplainer)

    # test that TabularExplainer works as well
    with pytest.warns(UserWarning):
        explainer = TabularExplainer(model=model, data=data, class_index=1, imputer="baseline")
        assert isinstance(explainer, TabularExplainer)
