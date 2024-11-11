"""This test module contains tests for the explainer module given differnt model types."""

from shapiq.explainer import Explainer


def test_torch_reg(torch_reg_model, background_reg_data):
    """Test the explainer with basic torch regression model."""

    x_explain = background_reg_data[0]

    explainer = Explainer(model=torch_reg_model, data=background_reg_data)
    values = explainer.explain(x_explain)
    print(values)
