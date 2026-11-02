"""Tests for the utility functions in the explainer module."""

from __future__ import annotations

import inspect
import sys
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVC, SVR

from shapiq import Explainer, TabularExplainer
from shapiq.explainer.product_kernel import ProductKernelExplainer
from shapiq.explainer.utils import (
    get_predict_function_and_model_type,
    has_unsupported_product_kernel,
    print_class,
)
from shapiq.tree.validation import SUPPORTED_MODELS
from tests.shapiq.conftest import (
    TABULAR_MODEL_FIXTURES,
    TABULAR_TENSORFLOW_MODEL_FIXTURES,
    TABULAR_TORCH_MODEL_FIXTURES,
    TREE_MODEL_FIXTURES,
)  # pyright: ignore[reportCallIssue]  thinks is RuntimeError
from tests.shapiq.fixtures.models import PRODUCT_KERNEL_MODEL_FIXTURES


def _utils_get_model(model, label, x_data):
    predict_function, model_type = get_predict_function_and_model_type(model, label)
    assert predict_function(model, x_data).ndim == 1  # pyright: ignore[reportCallIssue] thinks its RuntimeError
    return predict_function, model_type


@pytest.mark.external_libraries
@pytest.mark.parametrize(("model_name", "label"), TABULAR_MODEL_FIXTURES)
def test_tabular_get_predict_function_and_model_type(
    model_name,
    label,
    background_reg_dataset,
    request,
):
    """Tests whether the tabular model is recognized as a tabular model."""
    model = request.getfixturevalue(model_name)
    x_data, y = background_reg_dataset
    predict_function, model_type = _utils_get_model(model, label, x_data)
    assert model_type == "tabular"

    if label == "custom_model":
        assert np.all(predict_function(model, x_data) == y)  # pyright: ignore[reportCallIssue] thinks its RuntimeError

    if label == "sklearn.linear_model.LinearRegression":
        assert np.all(predict_function(model, x_data) == model.predict(x_data))  # pyright: ignore[reportCallIssue] thinks its RuntimeError


@pytest.mark.skipif(
    not any(pkg in sys.modules for pkg in ["tensorflow"]),
    reason="Tensorflow is not available.",
)
@pytest.mark.external_libraries
@pytest.mark.parametrize(("model_name", "label"), TABULAR_TENSORFLOW_MODEL_FIXTURES)
def test_tensorflow_get_predict_function_and_model_type(
    model_name,
    label,
    background_reg_dataset,
    request,
):
    """Tests whether the tensorflow model is recognized as a tabular model."""
    model = request.getfixturevalue(model_name)
    x_data, _ = background_reg_dataset
    _predict_function, model_type = _utils_get_model(model, label, x_data)
    assert model_type == "tabular"


@pytest.mark.external_libraries
@pytest.mark.parametrize(("model_name", "label"), TABULAR_TORCH_MODEL_FIXTURES)
def test_torch_get_predict_function_and_model_type(
    model_name,
    label,
    background_reg_dataset,
    request,
):
    """Tests whether the torch model is recognized as a tabular model."""
    model = request.getfixturevalue(model_name)
    x_data, _ = background_reg_dataset
    _predict_function, model_type = _utils_get_model(model, label, x_data)
    assert model_type == "tabular"


@pytest.mark.external_libraries
@pytest.mark.parametrize(("model_fixture", "model_class"), TREE_MODEL_FIXTURES)
def test_tree_get_predict_function_and_model_type(
    model_fixture,
    model_class,
    background_reg_dataset,
    request,
):
    """Tests whether the tree model is recognized as a tree model."""
    model = request.getfixturevalue(model_fixture)
    x_data, _y = background_reg_dataset
    predict_function, model_type = _utils_get_model(model, model_class, x_data)
    assert model_type == "tree"

    if model_class == "sklearn.tree.DecisionTreeRegressor":
        assert np.all(predict_function(model, x_data) == model.predict(x_data))  # pyright: ignore[reportCallIssue] thinks its RuntimeError


def test_all_supported_tree_models_recognized():
    """Test that all supported tree models are recognized as tree models."""
    model = Mock()
    for label in SUPPORTED_MODELS:
        _predict_function, model_type = get_predict_function_and_model_type(model, label)
        assert model_type == "tree"


def test_all_supported_product_kernel_models_recognized():
    """Test that all supported SVM models are recognized as product kernel models."""
    model = Mock()
    svm_models = [
        "sklearn.svm.SVC",
        "sklearn.svm._classes.SVC",
        "sklearn.svm.SVR",
        "sklearn.svm._classes.SVR",
        "sklearn.gaussian_process.GaussianProcessRegressor",
        "sklearn.gaussian_process._gpr.GaussianProcessRegressor",
    ]
    for label in svm_models:
        _predict_function, model_type = get_predict_function_and_model_type(model, label)
        assert model_type == "product_kernel"


@pytest.mark.parametrize(("model_fixture", "model_class"), PRODUCT_KERNEL_MODEL_FIXTURES)
def test_fitted_product_kernel_models_dispatch_to_product_kernel_explainer(
    model_fixture,
    model_class,
    background_clf_dataset_binary,
    background_reg_dataset,
    request,
):
    """Tests that fitted product kernel models are explained by the ProductKernelExplainer.

    The class path of a fitted model is its private module path (e.g.
    ``sklearn.svm._classes.SVC``), not the public one (``sklearn.svm.SVC``). Both must be
    recognized, otherwise ``shapiq.Explainer`` falls back to the ``TabularExplainer``.
    """
    model = request.getfixturevalue(model_fixture)
    is_classifier = model_class == "sklearn.svm.SVC"
    x_data, _y = background_clf_dataset_binary if is_classifier else background_reg_dataset

    # the inferred class path is the private one, which must still be recognized
    assert print_class(model) != model_class
    _predict_function, model_type = get_predict_function_and_model_type(model)
    assert model_type == "product_kernel"

    # ``shapiq.Explainer`` passes ``data`` and ``class_index`` on, which are ignored
    explainer = Explainer(model, x_data, class_index=1 if is_classifier else None)
    assert isinstance(explainer, ProductKernelExplainer)

    expected = ProductKernelExplainer(model).explain(x_data[0])
    explanation = explainer.explain(x_data[0])
    assert explanation.index == "SV"
    assert explanation.max_order == 1
    assert np.allclose(explanation.values, expected.values)


@pytest.mark.parametrize(
    "model",
    [
        SVC(kernel="linear"),
        SVR(kernel="poly"),
        GaussianProcessRegressor(),  # default kernel: ConstantKernel * RBF
    ],
)
def test_unsupported_product_kernel_models_dispatch_to_tabular_explainer(
    model,
    background_reg_dataset,
):
    """Tests that SVMs and GPs the ProductKernelExplainer cannot explain stay tabular."""
    x_data, y = background_reg_dataset
    model.fit(x_data, y > np.median(y) if isinstance(model, SVC) else y)
    assert has_unsupported_product_kernel(model)
    _predict_function, model_type = get_predict_function_and_model_type(model)
    assert model_type == "tabular"
    assert isinstance(Explainer(model, x_data), TabularExplainer)


def test_multiclass_svc_dispatches_to_tabular_explainer(background_clf_dataset):
    """Tests that a multiclass SVC, which the ProductKernelExplainer rejects, stays tabular."""
    x_data, y = background_clf_dataset
    model = SVC(kernel="rbf").fit(x_data, y)
    assert len(model.classes_) > 2
    _predict_function, model_type = get_predict_function_and_model_type(model)
    assert model_type == "tabular"


class ModelWithFalseCall:
    """A dummy model that has a __call__ method but does not match the expected signature."""

    def __call__(self, string: str, double: float):
        """A dummy call method that does not match the expected signature."""


class NonCallableModel:
    """A dummy model that does not implement a __call__ method."""


def test_exceptions_get_predict_function_and_model_type(background_reg_data):
    """Test the exceptions in get_predict_function_and_model_type."""
    # neither call nor predict functions
    model_without_call = NonCallableModel()
    with pytest.raises(TypeError):
        _, _ = get_predict_function_and_model_type(model_without_call, "non_sense_model")


def test_class_index():
    """Test the class index in get_predict_function_and_model_type."""

    def _model(x: np.ndarray):
        return np.array([[1, 2, 3, 4], [1, 2, 3, 4]])

    for i in range(4):
        pred_fun, _label = get_predict_function_and_model_type(_model, "custom_model", i)
        return_value = pred_fun(_model, np.array([[11, 22, 33, 44], [11, 22, 33, 44]]))  # pyright: ignore[reportCallIssue] thinks its RuntimeError
        assert return_value[0] == i + 1


def _valid_sig(param: inspect.Parameter):
    return param.annotation in (np.ndarray, inspect._empty, Any)
