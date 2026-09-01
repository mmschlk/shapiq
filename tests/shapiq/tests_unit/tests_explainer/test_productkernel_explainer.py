"""This test module contains all tests for the product kernel explainer module of the shapiq package."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessClassifier

from shapiq.explainer.product_kernel import (
    ProductKernelComputer,
    ProductKernelExplainer,
    ProductKernelModel,
)
from shapiq.explainer.product_kernel.conversion import convert_gp_reg, convert_svm
from shapiq.explainer.product_kernel.game import (
    ProductKernelGame,
)
from shapiq.game_theory.exact import ExactComputer


def test_invalid_application(bin_svc_model, background_clf_dataset_binary):
    """Test the product kernel explainer with an invalid application."""
    with pytest.raises(ValueError):
        _ = ProductKernelExplainer(model=bin_svc_model, max_order=2, index="SV")

    non_rbf_svm = copy.deepcopy(bin_svc_model)
    non_rbf_svm.kernel = "linear"

    with pytest.raises(ValueError):
        _ = ProductKernelExplainer(model=non_rbf_svm, max_order=1, index="SV")

    gaussian_process_classifier = GaussianProcessClassifier()

    with pytest.raises(TypeError):
        _ = ProductKernelExplainer(model=gaussian_process_classifier, max_order=1, index="SV")


def test_bin_svc_product_kernel_explainer(bin_svc_model, background_clf_dataset_binary):
    """Test the product kernel explainer with a binary SVC model."""

    # Initialize the explainer
    explainer = ProductKernelExplainer(model=bin_svc_model, max_order=1, index="SV")

    x_explain, _ = background_clf_dataset_binary
    explanation = explainer.explain(x_explain[0])
    prediction = bin_svc_model.decision_function(x_explain[0].reshape(1, -1))

    assert explanation.values.sum() == pytest.approx(prediction.item())


def test_svr_product_kernel_explainer(svr_model, background_reg_data):
    """Test the product kernel explainer with a SVR model."""

    # Initialize the explainer
    explainer = ProductKernelExplainer(model=svr_model, max_order=1, index="SV")

    x_explain = background_reg_data
    explanation = explainer.explain(x_explain[0])
    prediction = svr_model.predict(x_explain[0].reshape(1, -1))

    assert explanation.values.sum() == pytest.approx(prediction.item())


def test_gp_reg_product_kernel_explainer(gp_reg_model, background_reg_data):
    """Test the product kernel explainer with a Gaussian Process Regressor model."""

    # Initialize the explainer
    explainer = ProductKernelExplainer(model=gp_reg_model, max_order=1, index="SV")

    x_explain = background_reg_data
    explanation = explainer.explain(x_explain[0])
    prediction = gp_reg_model.predict(x_explain[0].reshape(1, -1))

    assert explanation.values.sum() == pytest.approx(prediction)


def test_svc_against_exact_computer(bin_svc_model, background_clf_dataset_binary):
    """Test the binary SVC model against the exact computer for product kernel explainer."""

    x_explain, _ = background_clf_dataset_binary

    # Initialize the exact computer
    svc_kernel_game = ProductKernelGame(
        model=convert_svm(bin_svc_model),
        n_players=bin_svc_model.n_features_in_,
        explain_point=x_explain[0],
        normalize=False,
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
        normalize=False,
    )
    exact_computer = ExactComputer(game=svr_kernel_game, n_players=svr_model.n_features_in_)

    sv_values = exact_computer("SV").values
    sum_values = sv_values.sum()

    model_prediction = svr_model.predict(x_explain[0].reshape(1, -1))
    model_prediction_scalar = model_prediction.item()

    assert model_prediction_scalar == pytest.approx(sum_values)


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


def test_invalid_quadrature_points(svr_model):
    """Test that a non-positive number of quadrature points is rejected."""
    with pytest.raises(ValueError, match="n_quadrature_points"):
        _ = ProductKernelExplainer(model=svr_model, n_quadrature_points=0)

    with pytest.raises(ValueError, match="n_quadrature_points"):
        _ = ProductKernelExplainer(model=svr_model, n_quadrature_points=-3)


def test_svr_quadrature_is_exact(svr_model, background_reg_data):
    """Test that the default quadrature rule reproduces the exact Shapley values."""
    x_explain = background_reg_data[0]
    n_players = svr_model.n_features_in_

    explanation = ProductKernelExplainer(model=svr_model).explain(x_explain)

    game = ProductKernelGame(
        model=convert_svm(svr_model),
        n_players=n_players,
        explain_point=x_explain,
        normalize=False,
    )
    exact = ExactComputer(game=game, n_players=n_players)("SV")

    for player in range(n_players):
        assert explanation[(player,)] == pytest.approx(exact[(player,)], abs=1e-10)


def test_fewer_quadrature_points_stay_close(svr_model, background_reg_data):
    """Test that dropping below the exactness bound degrades smoothly rather than abruptly."""
    x_explain = background_reg_data[0]
    n_players = svr_model.n_features_in_

    exact = ProductKernelExplainer(model=svr_model).explain(x_explain)
    approx = ProductKernelExplainer(model=svr_model, n_quadrature_points=2).explain(x_explain)

    scale = max(abs(exact[(player,)]) for player in range(n_players))
    for player in range(n_players):
        assert approx[(player,)] == pytest.approx(exact[(player,)], abs=0.05 * scale)


def test_single_feature_model():
    """Test a model with a single feature, where the leave-one-out product is empty."""
    rng = np.random.default_rng(0)
    x_train, alpha = rng.normal(size=(5, 1)), rng.normal(size=5)
    model = ProductKernelModel(X_train=x_train, alpha=alpha, n=5, d=1, gamma=1.0)
    x_explain = rng.normal(size=1)

    values = ProductKernelComputer(model).compute_shapley_values(x_explain)

    # with one player the Shapley value is the full marginal contribution v({0}) - v({})
    kernel = np.exp(-model.gamma * (x_train[:, 0] - x_explain[0]) ** 2)
    assert values[0] == pytest.approx(float(alpha @ (kernel - 1.0)))


def test_unsupported_kernel_type():
    """Test that the computer rejects kernels it cannot factorize."""
    model = ProductKernelModel(
        X_train=np.zeros((2, 2)), alpha=np.zeros(2), n=2, d=2, gamma=1.0, kernel_type="linear"
    )
    with pytest.raises(NotImplementedError, match="linear"):
        ProductKernelComputer(model).compute_shapley_values(np.zeros(2))
