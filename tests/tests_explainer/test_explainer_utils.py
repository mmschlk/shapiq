import inspect
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from shapiq.explainer.tree.validation import SUPPORTED_MODELS
from shapiq.explainer.utils import get_predict_function_and_model_type
from tests.conftest import TABULAR_MODEL_FIXTURES, TREE_MODEL_FIXTURES


@pytest.mark.external_libraries
@pytest.mark.parametrize("model_name, label", TABULAR_MODEL_FIXTURES)
def test_tabular_get_predict_function_and_model_type(
    model_name, label, background_reg_dataset, request
):
    model = request.getfixturevalue(model_name)
    X, y = background_reg_dataset

    predict_function, model_type = get_predict_function_and_model_type(model, label)
    assert model_type == "tabular"
    assert predict_function(model, X).ndim == 1

    if label == "custom_model":
        assert np.all(predict_function(model, X) == y)

    if label == "sklearn.linear_model.LinearRegression":
        assert np.all(predict_function(model, X) == model.predict(X))


@pytest.mark.external_libraries
@pytest.mark.parametrize("model_fixture, model_class", TREE_MODEL_FIXTURES)
def test_tree_get_predict_function_and_model_type(
    model_fixture, model_class, background_reg_dataset, request
):
    model = request.getfixturevalue(model_fixture)
    X, y = background_reg_dataset

    predict_function, model_type = get_predict_function_and_model_type(model, model_class)
    assert predict_function(model, X).ndim == 1
    assert model_type == "tree"

    if model_class == "sklearn.tree.DecisionTreeRegressor":
        assert np.all(predict_function(model, X) == model.predict(X))


def test_all_supported_tree_models_recognized():
    model = Mock()
    for label in SUPPORTED_MODELS:
        predict_function, model_type = get_predict_function_and_model_type(model, label)
        assert model_type == "tree"


class ModelWithFalseCall:
    def __call__(self, string: str, double: float):
        pass


class NonCallableModel:
    pass


def test_exceptions_get_predict_function_and_model_type(background_reg_data):
    # neither call nor predict functions
    model_without_call = NonCallableModel()
    with pytest.raises(TypeError):
        _, _ = get_predict_function_and_model_type(model_without_call, "non_sense_model")

def test_class_index():
    def _model(x: np.ndarray):
        return np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
    for i in range(0, 4):
        pred_fun, label = get_predict_function_and_model_type(_model, "custom_model", i)
        return_value = pred_fun(_model, np.array([[11, 22, 33, 44], [11, 22, 33, 44]]))
        assert return_value[0] == i+1

@pytest.mark.skip("not possible to implement right now")
def test_class_index():
    def _model(x: np.ndarray):
        return np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
    # out of bounds
    with pytest.raises(TypeError):
        _, _ = get_predict_function_and_model_type(_model, "non_sense_model", 4)
    # out of bounds
    with pytest.raises(TypeError):
        _, _ = get_predict_function_and_model_type(_model, "non_sense_model", -5)


def _valid_sig(param: inspect.Parameter):
    return (
            param.annotation == np.ndarray
            or param.annotation == inspect._empty
            or param.annotation == Any
    )


def callable_check():  # todo useful addition?
    # call with false signature
    model_with_false_call = ModelWithFalseCall()
    call_signature = inspect.signature(model_with_false_call)
    if not any([_valid_sig(param) for param in call_signature.parameters.values()]):
        raise TypeError
