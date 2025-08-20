"""This test module contains all tests for the product kernel explainer module of the shapiq package."""

from __future__ import annotations

from src.shapiq.explainer.product_kernel import ProductKernelExplainer


def test_bin_svc_product_kernel_explainer(bin_svc_model, background_clf_dataset_binary):
    """Test the product kernel explainer with a binary SVC model."""

    # Initialize the explainer
    explainer = ProductKernelExplainer(model=bin_svc_model, max_order=1, index="SV")

    x_explain, y_explain = background_clf_dataset_binary
    explanation = explainer.explain(x_explain[0])
    prediction = bin_svc_model.predict(x_explain)  # noqa:F841

    assert type(explanation).__name__ == "InteractionValues"

    # check init with class label
    _ = ProductKernelExplainer(model=bin_svc_model, max_order=1, min_order=0, class_index=0)

    assert True

    # compare baseline value with empty prediction
    # TODO(IsaH57): add (Issue #425)


def test_svr_product_kernel_explainer(svr_model, background_reg_data):
    """Test the product kernel explainer with a SVR model."""

    # Initialize the explainer
    explainer = ProductKernelExplainer(model=svr_model, max_order=1, index="SV")

    x_explain = background_reg_data
    explanation = explainer.explain(x_explain[0])
    prediction = svr_model.predict(x_explain)  # noqa:F841

    assert type(explanation).__name__ == "InteractionValues"

    # compare baseline value with empty prediction
    # TODO(IsaH57): add (Issue #425)


def test_gp_reg_product_kernel_explainer(gp_reg_model, background_reg_data):
    """Test the product kernel explainer with a Gaussian Process Regressor model."""

    # Initialize the explainer
    explainer = ProductKernelExplainer(model=gp_reg_model, max_order=1, index="SV")

    x_explain = background_reg_data
    explanation = explainer.explain(x_explain[0])
    prediction = gp_reg_model.predict(x_explain)  # noqa:F841

    assert type(explanation).__name__ == "InteractionValues"

    # compare baseline value with empty prediction
    # TODO(IsaH57): add (Issue #425)
