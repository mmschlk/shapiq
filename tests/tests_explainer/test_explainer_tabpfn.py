"""This test module tests the TabPFNExplainer object."""

from __future__ import annotations

import pytest

from shapiq import Explainer, InteractionValues, TabPFNExplainer, TabularExplainer
from tests.fixtures.data import BUDGET_NR_FEATURES_SMALL
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
    explanation = explainer.explain(x=x_explain, budget=BUDGET_NR_FEATURES_SMALL)
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
    explanation = explainer.explain(x=x_explain, budget=BUDGET_NR_FEATURES_SMALL)
    assert isinstance(explanation, InteractionValues)

    # test that bare explainer gets turned into TabPFNExplainer
    explainer = Explainer(model=model, data=data, labels=labels, x_test=x_test)
    assert isinstance(explainer, TabPFNExplainer)

    # test that TabularExplainer works as well
    with pytest.warns(UserWarning):
        explainer = TabularExplainer(model=model, data=data, class_index=1, imputer="baseline")
        assert isinstance(explainer, TabularExplainer)


class TestTabPFNExplainer:
    def test_copy_model(self, tabpfn_regression_problem):
        """Tests that the model is copied when copy_model is True."""
        from tabpfn import TabPFNRegressor

        model, data, labels, x_test = tabpfn_regression_problem
        x_explain = x_test[0].reshape(1, -1)
        explainer = TabPFNExplainer(
            model=model, data=data, labels=labels, x_test=x_test, copy_model=True
        )
        assert isinstance(explainer.model, TabPFNRegressor)

        # check that the model is a copy
        explainer.model.fit(data[:, :1], labels)  # fit on a different column
        with pytest.raises(ValueError):
            explainer.model.predict(x_explain)
        model.predict(x_explain)  # original model should still work


class TestTabPFNExplainerBugFixes:
    """Tests for bug fixes conducted in the TabPFNExplainer."""

    @skip_if_no_tabpfn
    @pytest.mark.external_libraries
    def test_after_explanation_prediction(self, tabpfn_regression_problem):
        """Tests that the model can be used for prediction after explanation.

        This bug was raised in issue [#396](https://github.com/mmschlk/shapiq/issues/396)
        """
        model, data, labels, x_test = tabpfn_regression_problem
        x_explain = x_test[0]

        prediction = model.predict(x_explain.reshape(1, -1))
        assert prediction is not None

        explainer = TabPFNExplainer(
            model=model, data=data, labels=labels, x_test=x_test, copy_model=False
        )
        _ = explainer.explain(x=x_explain, budget=10)

        prediction = model.predict(x_explain.reshape(1, -1))
        assert prediction is not None  # this would fail if the bug was present
