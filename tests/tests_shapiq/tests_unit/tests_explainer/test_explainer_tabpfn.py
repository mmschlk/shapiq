"""This test module tests the TabPFNExplainer object."""

from __future__ import annotations

import pytest
from tests.fixtures.data import BUDGET_NR_FEATURES_SMALL
from tests.markers import skip_if_no_tabpfn

from shapiq import Explainer, InteractionValues, TabPFNExplainer, TabularExplainer


@skip_if_no_tabpfn
@pytest.mark.external_libraries
class TestTabPFNExplainer:
    """Tests for the TabPFNExplainer class."""

    def test_tabpfn_explainer_clf(self, tabpfn_classification_problem):
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

    def test_tabpfn_explainer_reg(self, tabpfn_regression_problem):
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

    def test_tabpfn_bare_explainer(self, tabpfn_classification_problem, tabpfn_regression_problem):
        """Test that the TabPFNExplainer can be initialized without data and labels."""

        def _run_test(problem):
            """Helper function to run the test."""
            model, data, labels, x_test = problem
            explainer = Explainer(model=model, data=data, labels=labels, x_test=x_test)
            assert isinstance(explainer, TabPFNExplainer)

            # test that TabularExplainer works as well
            with pytest.warns(UserWarning):
                explainer = TabularExplainer(
                    model=model, data=data, class_index=1, imputer="baseline"
                )
                assert isinstance(explainer, TabularExplainer)

        _run_test(tabpfn_regression_problem)
        _run_test(tabpfn_classification_problem)

    def test_tabpfn_user_warning(self, tabpfn_regression_problem, tabpfn_classification_problem):
        """Test that the TabularExplainer can be used with TabPFN models but raises a UserWarning."""

        def _run_test(problem):
            """Helper function to run the test."""
            model, data, _, _ = problem
            with pytest.warns(UserWarning):
                explainer = TabularExplainer(
                    model=model, data=data, class_index=1, imputer="baseline"
                )
                assert isinstance(explainer, TabularExplainer)

        _run_test(tabpfn_regression_problem)
        _run_test(tabpfn_regression_problem)


@skip_if_no_tabpfn
@pytest.mark.external_libraries
class TestTabPFNExplainerBugFixes:
    """Tests for bug fixes conducted in the TabPFNExplainer."""

    def test_after_explanation_prediction(self, tabpfn_regression_problem):
        """Tests that the model can be used for prediction after explanation.

        This bug was raised in issue [#396](https://github.com/mmschlk/shapiq/issues/396)
        """
        model, data, labels, x_test = tabpfn_regression_problem
        x_explain = x_test[0]

        _ = model.predict(x_explain.reshape(1, -1))

        explainer = TabPFNExplainer(model=model, data=data, labels=labels, x_test=x_test)
        explainer.explain(x=x_explain, budget=3)
        assert model.n_features_in_ == data.shape[1]

        model.predict(x_explain.reshape(1, -1))  # should not raise an error
