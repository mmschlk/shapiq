"""This module contains utility functions for the explainer module."""

import re
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

WARNING_NO_CLASS_INDEX = (
    "No class_index provided. "
    "Explaining the 2nd '1' class for classification models. "
    "Please provide the class_index to explain a different class. "
    "Disregard this warning for regression models."
)

ModelType = TypeVar("ModelType")


def get_explainers() -> dict[str, Any]:
    """Return a dictionary of all available explainer classes.

    Returns:
        A dictionary of all available explainer classes.
    """
    from shapiq.explainer.tabpfn import TabPFNExplainer
    from shapiq.explainer.tabular import TabularExplainer
    from shapiq.explainer.tree.explainer import TreeExplainer

    return {"tabular": TabularExplainer, "tree": TreeExplainer, "tabpfn": TabPFNExplainer}


def get_predict_function_and_model_type(
    model: ModelType,
    model_class: str | None = None,
    class_index: int | None = None,
) -> tuple[Callable[[ModelType, np.ndarray], np.ndarray], str]:
    """Get the predict function and model type for a given model.

    The prediction function is used in the explainer to predict the model's output for a given data
    point. The function has the following signature: ``predict_function(model, data)``.

    Args:
        model: The model to explain. Can be any model object or callable function. We try to infer
            the model type from the model object.

        model_class: The class of the model. as a string. If not provided, it will be inferred from
            the model object.

        class_index: The class index of the model to explain. Defaults to ``None``, which will set
            the class index to ``1`` per default for classification models and is ignored for
            regression models.

    Returns:
        A tuple of the predict function and the model type.
    """
    from . import tree

    if model_class is None:
        model_class = print_class(model)

    _model_type = "tabular"  # default
    _predict_function = None

    if callable(model):
        _predict_function = predict_callable

    # sklearn
    if model_class in [
        "sklearn.tree.DecisionTreeRegressor",
        "sklearn.tree._classes.DecisionTreeRegressor",
        "sklearn.tree.DecisionTreeClassifier",
        "sklearn.tree._classes.DecisionTreeClassifier",
        "sklearn.ensemble.RandomForestClassifier",
        "sklearn.ensemble._forest.RandomForestClassifier",
        "sklearn.ensemble.ExtraTreesClassifier",
        "sklearn.ensemble._forest.ExtraTreesClassifier",
        "sklearn.ensemble.RandomForestRegressor",
        "sklearn.ensemble._forest.RandomForestRegressor",
        "sklearn.ensemble.ExtraTreesRegressor",
        "sklearn.ensemble._forest.ExtraTreesRegressor",
        "sklearn.ensemble.IsolationForest",
        "sklearn.ensemble._iforest.IsolationForest",
    ]:
        _model_type = "tree"

    # lightgbm
    if model_class in [
        "lightgbm.basic.Booster",
        "lightgbm.sklearn.LGBMRegressor",
        "lightgbm.sklearn.LGBMClassifier",
    ]:
        _model_type = "tree"

    # xgboost
    if model_class == "xgboost.core.Booster":
        _predict_function = predict_xgboost
    if model_class in [
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBRegressor",
        "xgboost.sklearn.XGBClassifier",
    ]:
        _model_type = "tree"

    # pytorch
    if model_class in [
        "torch.nn.modules.container.Sequential",
        "torch.nn.modules.module.Module",
        "torch.nn.modules.container.ModuleList",
        "torch.nn.modules.container.ModuleDict",
    ]:
        _model_type = "tabular"
        _predict_function = predict_torch

    # tensorflow
    if model_class in [
        "tensorflow.python.keras.engine.sequential.Sequential",
        "tensorflow.python.keras.engine.training.Model",
        "tensorflow.python.keras.engine.functional.Functional",
        "keras.engine.sequential.Sequential",
        "keras.engine.training.Model",
        "keras.engine.functional.Functional",
        "keras.src.models.sequential.Sequential",
    ]:
        _model_type = "tabular"
        _predict_function = predict_tensorflow

    if model_class in [
        "tabpfn.classifier.TabPFNClassifier",
        "tabpfn.regressor.TabPFNRegressor",
    ]:
        _model_type = "tabpfn"

    # default extraction (sklearn api)
    if _predict_function is None and hasattr(model, "predict_proba"):
        _predict_function = predict_proba
    elif _predict_function is None and hasattr(model, "predict"):
        _predict_function = predict
    # extraction for tree models
    elif isinstance(model, tree.TreeModel):  # test scenario
        _predict_function = model.compute_empty_prediction
        _model_type = "tree"
    elif isinstance(model, list) and all([isinstance(m, tree.TreeModel) for m in model]):
        _predict_function = model[0].compute_empty_prediction
        _model_type = "tree"
    elif _predict_function is None:
        raise TypeError(
            f"`model` is of unsupported type: {model_class}.\n"
            "Please, raise a new issue at https://github.com/mmschlk/shapiq/issues if you want this model type\n"
            "to be handled automatically by shapiq.Explainer. Otherwise, use one of the supported explainers:\n"
            f'{", ".join(print_classes_nicely(get_explainers()))}'
        )

    if class_index is None:
        class_index = 1

    def _predict_function_with_class_index(model: ModelType, data: np.ndarray) -> np.ndarray:
        """A wrapper prediction function to retrieve class_index predictions for classifiers.
        Regression models are not affected by this function.

        Args:
            model: The model to predict with.
            data: The data to predict on.

        Returns:
            The model's prediction for the given data point as a vector.
        """
        predictions = _predict_function(model, data)
        if predictions.ndim == 1:
            return predictions
        elif predictions.shape[1] == 1:
            return predictions[:, 0]
        return predictions[:, class_index]

    return _predict_function_with_class_index, _model_type


def predict_callable(model: ModelType, data: np.ndarray) -> np.ndarray:
    return model(data)


def predict(model: ModelType, data: np.ndarray) -> np.ndarray:
    return model.predict(data)


def predict_proba(model: ModelType, data: np.ndarray) -> np.ndarray:
    return model.predict_proba(data)


def predict_xgboost(model: ModelType, data: np.ndarray) -> np.ndarray:
    from xgboost import DMatrix

    return model.predict(DMatrix(data))


def predict_tensorflow(model: ModelType, data: np.ndarray) -> np.ndarray:
    return model.predict(data, verbose=0)


def predict_torch(model: ModelType, data: np.ndarray) -> np.ndarray:
    import torch

    return model(torch.from_numpy(data).float()).detach().numpy()


def print_classes_nicely(obj):
    """
    Converts a list of classes into *user-readable* class names. I/O examples:
    [shapiq.explainer._base.Explainer] -> ['shapiq.Explainer']
    {'tree': shapiq.explainer.tree.explainer.TreeExplainer}  -> ['shapiq.TreeExplainer']
    {'tree': shapiq.TreeExplainer}  -> ['shapiq.TreeExplainer']
    """
    if isinstance(obj, dict):
        return [".".join([print_class(v).split(".")[i] for i in (0, -1)]) for _, v in obj.items()]
    elif isinstance(obj, list):
        return [".".join([print_class(v).split(".")[i] for i in (0, -1)]) for v in obj]


def print_class(obj):
    """
    Converts a class or class type into a *user-readable* class name. I/O examples:
    sklearn.ensemble._forest.RandomForestRegressor -> 'sklearn.ensemble._forest.RandomForestRegressor'
    type(sklearn.ensemble._forest.RandomForestRegressor) -> 'sklearn.ensemble._forest.RandomForestRegressor'
    shapiq.explainer.tree.explainer.TreeExplainer -> 'shapiq.explainer.tree.explainer.TreeExplainer'
    shapiq.TreeExplainer -> 'shapiq.explainer.tree.explainer.TreeExplainer'
    type(shapiq.TreeExplainer) -> 'shapiq.explainer.tree.explainer.TreeExplainer'
    """
    if isinstance(obj, type):
        return re.search("(?<=<class ').*(?='>)", str(obj))[0]
    else:
        return re.search("(?<=<class ').*(?='>)", str(type(obj)))[0]
