"""This test module contains tests for the Tabular explainer module given differnt model types."""

import pytest

from shapiq import InteractionValues
from shapiq.explainer import Explainer, TabularExplainer


def test_torch_reg(torch_reg_model, background_reg_data):
    """Test the explainer with basic torch regression model."""
    import torch

    x_explain = background_reg_data[0]
    x_explain_tensor = torch.tensor(x_explain, dtype=torch.float32).reshape(1, -1)
    prediction = torch_reg_model(x_explain_tensor).detach().numpy()[0]

    explainer = Explainer(model=torch_reg_model, data=background_reg_data)
    values = explainer.explain(x_explain)
    assert isinstance(values, InteractionValues)
    sum_of_values = sum(values.values) + values.baseline_value
    assert prediction == pytest.approx(sum_of_values, rel=0.01)


def test_torch_clf(torch_clf_model, background_clf_data):
    """Test the explainer with basic torch classification model."""
    import torch

    x_explain = background_clf_data[0]
    x_explain_tensor = torch.tensor(x_explain, dtype=torch.float32).reshape(1, -1)
    prediction = torch_clf_model(x_explain_tensor).detach().numpy()[0]

    explainer = Explainer(model=torch_clf_model, data=background_clf_data, class_index=2)
    values = explainer.explain(x_explain)
    assert isinstance(values, InteractionValues)
    sum_of_values = sum(values.values) + values.baseline_value
    assert prediction[2] == pytest.approx(sum_of_values, rel=0.001)

    explainer = Explainer(model=torch_clf_model, data=background_clf_data, class_index=0)
    values = explainer.explain(x_explain)
    assert isinstance(values, InteractionValues)
    sum_of_values = sum(values.values) + values.baseline_value
    assert prediction[0] == pytest.approx(sum_of_values, rel=0.001)


def test_sklearn_clf(rf_clf_model, background_clf_data):
    """Test the explainer with a basic sklearn classification model."""

    x_explain = background_clf_data[0]
    prediction = rf_clf_model.predict_proba(x_explain.reshape(1, -1))[0]

    explainer = TabularExplainer(model=rf_clf_model, data=background_clf_data, class_index=2)
    values = explainer.explain(x_explain)
    assert isinstance(values, InteractionValues)
    sum_of_values = sum(values.values) + values.baseline_value
    assert prediction[2] == pytest.approx(sum_of_values, rel=0.001)

    explainer = TabularExplainer(model=rf_clf_model, data=background_clf_data, class_index=0)
    values = explainer.explain(x_explain)
    assert isinstance(values, InteractionValues)
    sum_of_values = sum(values.values) + values.baseline_value
    assert prediction[0] == pytest.approx(sum_of_values, rel=0.001)

    # do the same with the bare explainer (only for class_label=2)
    explainer = Explainer(model=rf_clf_model, data=background_clf_data, class_index=2)
    values = explainer.explain(x_explain)
    assert isinstance(values, InteractionValues)
    sum_of_values = sum(values.values)
    assert prediction[2] == pytest.approx(sum_of_values, rel=0.001)


def test_sklearn_reg(rf_reg_model, background_reg_data):
    """Test the explainer with a basic sklearn regression model."""

    x_explain = background_reg_data[0]
    prediction = rf_reg_model.predict(x_explain.reshape(1, -1))[0]

    explainer = TabularExplainer(model=rf_reg_model, data=background_reg_data)
    values = explainer.explain(x_explain)
    assert isinstance(values, InteractionValues)
    sum_of_values = sum(values.values) + values.baseline_value
    assert prediction == pytest.approx(sum_of_values)

    # do the same with the bare explainer
    explainer = Explainer(model=rf_reg_model, data=background_reg_data)
    values = explainer.explain(x_explain)
    assert isinstance(values, InteractionValues)
    sum_of_values = sum(values.values) + values.baseline_value
    assert prediction == pytest.approx(sum_of_values, rel=0.01)


def test_sklearn_clf_logistic_regression(lr_clf_model, background_clf_data):
    """Test the explainer with a basic sklearn logistic regression model."""

    x_explain = background_clf_data[0]
    prediction = lr_clf_model.predict_proba(x_explain.reshape(1, -1))[0]

    explainer = TabularExplainer(model=lr_clf_model, data=background_clf_data, class_index=2)
    values = explainer.explain(x_explain)
    assert isinstance(values, InteractionValues)
    sum_of_values = sum(values.values) + values.baseline_value
    assert prediction[2] == pytest.approx(sum_of_values)

    explainer = TabularExplainer(model=lr_clf_model, data=background_clf_data, class_index=0)
    values = explainer.explain(x_explain)
    assert isinstance(values, InteractionValues)
    sum_of_values = sum(values.values) + values.baseline_value
    assert prediction[0] == pytest.approx(sum_of_values)

    # do the same with the bare explainer (only for class_label=2)
    explainer = Explainer(model=lr_clf_model, data=background_clf_data, class_index=2)
    values = explainer.explain(x_explain)
    assert isinstance(values, InteractionValues)
    sum_of_values = sum(values.values) + values.baseline_value
    assert prediction[2] == pytest.approx(sum_of_values)


def test_sklearn_reg_linear_regression(lr_reg_model, background_reg_data):
    """Test the explainer with a basic sklearn linear regression model."""

    x_explain = background_reg_data[0]
    prediction = lr_reg_model.predict(x_explain.reshape(1, -1))[0]

    explainer = TabularExplainer(model=lr_reg_model, data=background_reg_data)
    values = explainer.explain(x_explain)
    assert isinstance(values, InteractionValues)
    sum_of_values = sum(values.values) + values.baseline_value
    assert prediction == pytest.approx(sum_of_values)

    # do the same with the bare explainer
    explainer = Explainer(model=lr_reg_model, data=background_reg_data)
    values = explainer.explain(x_explain)
    assert isinstance(values, InteractionValues)
    sum_of_values = sum(values.values) + values.baseline_value
    assert prediction == pytest.approx(sum_of_values)
