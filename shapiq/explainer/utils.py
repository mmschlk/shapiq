"""This module contains utility functions for the explainer module."""

import re
import warnings
from typing import Any, Optional

WARNING_NO_CLASS_LABEL = (
    "No class_label provided. Explaining the 2nd '1' per default. "
    "Please provide the class_label to explain a different class."
)


def get_explainers() -> dict[str, Any]:
    """Return a dictionary of all available explainer classes.

    Returns:
        A dictionary of all available explainer classes.
    """
    from shapiq.explainer.tabular import TabularExplainer
    from shapiq.explainer.tree.explainer import TreeExplainer

    return {"tabular": TabularExplainer, "tree": TreeExplainer}


def get_predict_function_and_model_type(model, model_class, class_label: Optional[int] = None):
    from . import tree

    model_predictor = ModelPredictor(class_label)
    _model_type = "tabular"  # default
    _predict_function = None

    if callable(model):
        _predict_function = model_predictor.predict_callable

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
        _predict_function = model_predictor.predict_xgboost
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
        _predict_function = model_predictor.predict_torch

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
        _predict_function = model_predictor.predict_tensorflow

    # default extraction (sklearn api)
    if _predict_function is None and hasattr(model, "predict_proba"):
        _predict_function = model_predictor.predict_proba
        if class_label is None:
            warnings.warn(WARNING_NO_CLASS_LABEL)
    elif _predict_function is None and hasattr(model, "predict"):
        _predict_function = model_predictor.predict
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

    return _predict_function, _model_type


class ModelPredictor:
    """A wrapper class for different model prediction functions.

    This class provides a unified interface for different model types to predict on data.

    Args:
        class_label: The class label to predict on. Defaults to ``None``. If ``None``, a warning is
        raised and the class label is set to ``1``.

    Raises:
        UserWarning: If no class label is provided.
    """

    def __init__(self, class_label: Optional[int] = None):
        if class_label is None:
            warnings.warn(WARNING_NO_CLASS_LABEL)
            class_label = 1
        self.class_label = class_label

    @staticmethod
    def predict_callable(model, data):
        return model(data)

    @staticmethod
    def predict(model, data):
        return model.predict(data)

    def predict_proba(self, model, data):
        return model.predict_proba(data)[:, self.class_label]

    @staticmethod
    def predict_xgboost(model, data):
        from xgboost import DMatrix

        return model.predict(DMatrix(data))

    def predict_torch(self, model, data):
        import torch

        data = torch.tensor(data, dtype=torch.float32)
        predictions = model(data).detach().numpy()
        return self._get_class_based_predictions(predictions)

    def predict_tensorflow(self, model, data):
        predictions = model.predict(data, verbose=0)
        return self._get_class_based_predictions(predictions)

    def _get_class_based_predictions(self, predictions):
        if len(predictions.shape) == 1:
            return predictions
        elif predictions.shape[1] == 1:
            return predictions[:, 0]
        return predictions[:, self.class_label]


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
