"""Tests for the utility functions in the explainer module."""

from __future__ import annotations

import inspect
import sys
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from shapiq.explainer.tree.validation import SUPPORTED_MODELS
from shapiq.explainer.utils import get_predict_function_and_model_type
from tests.shapiq.conftest import (
    TABULAR_MODEL_FIXTURES,
    TABULAR_TENSORFLOW_MODEL_FIXTURES,
    TABULAR_TORCH_MODEL_FIXTURES,
    TREE_MODEL_FIXTURES,
)


def _utils_get_model(model, label, x_data):
    predict_function, model_type = get_predict_function_and_model_type(model, label)
    assert predict_function(model, x_data).ndim == 1
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
        assert np.all(predict_function(model, x_data) == y)

    if label == "sklearn.linear_model.LinearRegression":
        assert np.all(predict_function(model, x_data) == model.predict(x_data))


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
    predict_function, model_type = _utils_get_model(model, label, x_data)
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
    predict_function, model_type = _utils_get_model(model, label, x_data)
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
    x_data, y = background_reg_dataset
    predict_function, model_type = _utils_get_model(model, model_class, x_data)
    assert model_type == "tree"

    if model_class == "sklearn.tree.DecisionTreeRegressor":
        assert np.all(predict_function(model, x_data) == model.predict(x_data))


def test_all_supported_tree_models_recognized():
    """Test that all supported tree models are recognized as tree models."""
    model = Mock()
    for label in SUPPORTED_MODELS:
        predict_function, model_type = get_predict_function_and_model_type(model, label)
        assert model_type == "tree"


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
        pred_fun, label = get_predict_function_and_model_type(_model, "custom_model", i)
        return_value = pred_fun(_model, np.array([[11, 22, 33, 44], [11, 22, 33, 44]]))
        assert return_value[0] == i + 1


def _valid_sig(param: inspect.Parameter):
    return param.annotation in (np.ndarray, inspect._empty, Any)
