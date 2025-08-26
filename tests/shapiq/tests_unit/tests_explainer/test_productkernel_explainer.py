"""This test module contains all tests for the product kernel explainer module of the shapiq package."""

from __future__ import annotations

import pytest

from shapiq.explainer.product_kernel import ProductKernelExplainer
from shapiq.explainer.product_kernel.conversion import convert_gp_reg, convert_svm
from shapiq.explainer.product_kernel.game import ProductKernelGame, ProductKernelGame_Author
from shapiq.game_theory.exact import ExactComputer


def test_bin_svc_product_kernel_explainer(bin_svc_model, background_clf_dataset_binary):
    """Test the product kernel explainer with a binary SVC model."""

    # Initialize the explainer
    explainer = ProductKernelExplainer(model=bin_svc_model, max_order=1, index="SV")

    x_explain, _ = background_clf_dataset_binary
    explanation = explainer.explain(x_explain[0])
    prediction = bin_svc_model.predict(x_explain[0].reshape(1, -1))  # noqa:F841

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
    prediction = svr_model.predict(x_explain[0].reshape(1, -1))  # noqa:F841

    assert type(explanation).__name__ == "InteractionValues"

    # compare baseline value with empty prediction
    # TODO(IsaH57): add (Issue #425)


def test_gp_reg_product_kernel_explainer(gp_reg_model, background_reg_data):
    """Test the product kernel explainer with a Gaussian Process Regressor model."""

    # Initialize the explainer
    explainer = ProductKernelExplainer(model=gp_reg_model, max_order=1, index="SV")

    x_explain = background_reg_data
    explanation = explainer.explain(x_explain[0])
    prediction = gp_reg_model.predict(x_explain[0].reshape(1, -1))  # noqa:F841

    assert type(explanation).__name__ == "InteractionValues"

    # compare baseline value with empty prediction
    # TODO(IsaH57): add (Issue #425)


def test_svc_against_exact_computer(bin_svc_model, background_clf_dataset_binary):
    """Test the binary SVC model against the exact computer for product kernel explainer."""

    x_explain, _ = background_clf_dataset_binary

    # Initialize the exact computer
    svc_kernel_game = ProductKernelGame(
        model=convert_svm(bin_svc_model),
        n_players=bin_svc_model.n_features_in_,
        explain_point=x_explain[0],
        normalize=True,
    )
    exact_computer = ExactComputer(game=svc_kernel_game, n_players=bin_svc_model.n_features_in_)

    sv_values = exact_computer("SV").values
    sum_values = sv_values.sum()

    model_prediction = bin_svc_model.decision_function(x_explain[0].reshape(1, -1))
    model_prediction_scalar = model_prediction.item()

    assert model_prediction_scalar == pytest.approx(sum_values)


def test_svr_against_exact_computer(svr_model, background_reg_data):
    """Test the SVR model against the exact computer for product kernel explainer."""

    x_explain = background_reg_data

    # Initialize the exact computer
    svr_kernel_game = ProductKernelGame(
        model=convert_svm(svr_model),
        n_players=svr_model.n_features_in_,
        explain_point=x_explain[0],
        normalize=True,
    )
    exact_computer = ExactComputer(game=svr_kernel_game, n_players=svr_model.n_features_in_)

    sv_values = exact_computer("SV").values
    sum_values = sv_values.sum()

    model_prediction = svr_model.predict(x_explain[0].reshape(1, -1))
    model_prediction_scalar = model_prediction.item()

    assert model_prediction_scalar == pytest.approx(sum_values)


def test_game_equal_to_author(gp_reg_model, background_reg_data):
    """Test if the ProductKernelGame implementation is equal to the author's implementation."""

    x_explain = background_reg_data

    # Initialize both games
    game_author = ProductKernelGame_Author(
        model=convert_gp_reg(gp_reg_model),
        n_players=gp_reg_model.n_features_in_,
        explain_point=x_explain[0],
        normalize=False,
    )
    game_new = ProductKernelGame(
        model=convert_gp_reg(gp_reg_model),
        n_players=gp_reg_model.n_features_in_,
        explain_point=x_explain[0],
        normalize=False,
    )
    from shapiq.utils import powerset, transform_coalitions_to_array

    coalitions = list(powerset(range(gp_reg_model.n_features_in_)))
    coalitions = transform_coalitions_to_array(coalitions, gp_reg_model.n_features_in_)

    values_author = game_author.value_function(coalitions)
    values_new = game_new.value_function(coalitions)

    assert values_author == pytest.approx(values_new)


def test_gp_reg_against_exact_computer(gp_reg_model, background_reg_data):
    """Test the Gaussian Process Regression model against the exact computer for product kernel explainer."""

    x_explain = background_reg_data

    # Initialize the exact computer
    gp_reg_kernel_game = ProductKernelGame(
        model=convert_gp_reg(gp_reg_model),
        n_players=gp_reg_model.n_features_in_,
        explain_point=x_explain[0],
        normalize=False,
    )
    exact_computer = ExactComputer(game=gp_reg_kernel_game, n_players=gp_reg_model.n_features_in_)

    sv_values = exact_computer("SV").values
    sum_values = sv_values.sum()

    model_prediction = gp_reg_model.predict(x_explain[0].reshape(1, -1))
    model_prediction_scalar = model_prediction.item()

    assert model_prediction_scalar == pytest.approx(sum_values)
