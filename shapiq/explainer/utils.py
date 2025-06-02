"""This module contains utility functions for the explainer module."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from shapiq.games.base import Game
    from shapiq.utils.custom_types import Model

WARNING_NO_CLASS_INDEX = (
    "No class_index provided. "
    "Explaining the 2nd '1' class for classification models. "
    "Please provide the class_index to explain a different class. "
    "Disregard this warning for regression models."
)


def get_explainers() -> dict[str, Any]:
    """Return a dictionary of all available explainer classes.

    Returns:
        A dictionary of all available explainer classes.

    """
    from shapiq.explainer.agnostic import AgnosticExplainer
    from shapiq.explainer.tabpfn import TabPFNExplainer
    from shapiq.explainer.tabular import TabularExplainer
    from shapiq.explainer.tree.explainer import TreeExplainer

    return {
        "tabular": TabularExplainer,
        "tree": TreeExplainer,
        "tabpfn": TabPFNExplainer,
        "game": AgnosticExplainer,
        "imputer": AgnosticExplainer,
    }


def get_predict_function_and_model_type(
    model: Model | Game | Callable[[np.ndarray], np.ndarray],
    model_class: str | None = None,
    class_index: int | None = None,
) -> tuple[Callable[[Model, np.ndarray], np.ndarray], str]:
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
    from shapiq.games.base import Game
    from shapiq.games.imputer.base import Imputer

    from .tree import TreeModel

    if model_class is None:
        model_class = print_class(model)

    _model_type = "tabular"  # default
    _predict_function = None

    if isinstance(model, Imputer) or model_class == "shapiq.games.imputer.base.Imputer":
        _predict_function = model._predict_function  # noqa: SLF001
        _model_type = "imputer"
        return _predict_function, _model_type

    if isinstance(model, Game) or model_class == "shapiq.games.base.Game":
        _predict_function = RuntimeError("Games cannot be used for prediction.")
        _model_type = "game"
        return _predict_function, _model_type

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
    elif isinstance(model, TreeModel):  # test scenario
        _predict_function = model.compute_empty_prediction
        _model_type = "tree"
    elif isinstance(model, list) and all(isinstance(m, TreeModel) for m in model):
        _predict_function = model[0].compute_empty_prediction
        _model_type = "tree"
    elif _predict_function is None:
        msg = (
            f"`model` is of unsupported type: {model_class}.\n"
            "Please, raise a new issue at https://github.com/mmschlk/shapiq/issues if you want this model type\n"
            "to be handled automatically by shapiq.Explainer. Otherwise, use one of the supported explainers:\n"
            f"{', '.join(print_classes_nicely(get_explainers()))}"
        )
        raise TypeError(msg)

    if class_index is None:
        class_index = 1

    def _predict_function_with_class_index(model: Model, data: np.ndarray) -> np.ndarray:
        """A wrapper prediction function to retrieve class_index predictions for classifiers.

        Note:
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
        if predictions.shape[1] == 1:
            return predictions[:, 0]
        return predictions[:, class_index]

    return _predict_function_with_class_index, _model_type


def predict_callable(model: Model, data: np.ndarray) -> np.ndarray:
    """Makes predictions with a model that is callable."""
    return model(data)


def predict(model: Model, data: np.ndarray) -> np.ndarray:
    """Makes predictions with a model that has a ``predict`` method."""
    return model.predict(data)


def predict_proba(model: Model, data: np.ndarray) -> np.ndarray:
    """Makes predictions with a model that has a ``predict_proba`` method."""
    return model.predict_proba(data)


def predict_xgboost(model: Model, data: np.ndarray) -> np.ndarray:
    """Makes predictions with an XGBoost model."""
    from xgboost import DMatrix

    return model.predict(DMatrix(data))


def predict_tensorflow(model: Model, data: np.ndarray) -> np.ndarray:
    """Makes predictions with a TensorFlow model."""
    return model.predict(data, verbose=0)


def predict_torch(model: Model, data: np.ndarray) -> np.ndarray:
    """Makes predictions with a PyTorch model."""
    import torch

    return model(torch.from_numpy(data).float()).detach().numpy()


def print_classes_nicely(obj: list[Any] | dict[str, Any]) -> list[str] | None:
    """Converts a collection of classes into *user-readable* class names.

    I/O examples:
        - ``[shapiq.explainer._base.Explainer]`` -> ``['shapiq.Explainer']``
        - ``{'tree': shapiq.explainer.tree.explainer.TreeExplainer}``  -> ``['shapiq.TreeExplainer']``
        - ``{'tree': shapiq.TreeExplainer}  -> ``['shapiq.TreeExplainer']``.

    Args:
        obj: The objects as a list or dictionary to convert. Can be a class or a class type.
        Can be a list or dictionary of classes or class types.

    Returns:
        The user-readable class names as a list. If the input is not a list or dictionary, returns
            ``None``.

    """
    if isinstance(obj, dict):
        return [".".join([print_class(v).split(".")[i] for i in (0, -1)]) for _, v in obj.items()]
    if isinstance(obj, list):
        return [".".join([print_class(v).split(".")[i] for i in (0, -1)]) for v in obj]
    return None


def print_class(obj: object) -> str:
    """Converts a class or class type into a *user-readable* class name.

    I/O Examples:
        - ``sklearn.ensemble._forest.RandomForestRegressor`` -> ``'sklearn.ensemble._forest.RandomForestRegressor'``
        - ``type(sklearn.ensemble._forest.RandomForestRegressor)`` -> ``'sklearn.ensemble._forest.RandomForestRegressor'``
        - ``shapiq.explainer.tree.explainer.TreeExplainer`` -> ``'shapiq.explainer.tree.explainer.TreeExplainer'``
        - ``shapiq.TreeExplainer`` -> ``'shapiq.explainer.tree.explainer.TreeExplainer'``
        - ``type(shapiq.TreeExplainer)`` -> ``'shapiq.explainer.tree.explainer.TreeExplainer'``

    Args:
        obj: The object to convert. Can be a class or a class type.

    Returns:
        The user-readable class name.

    """
    if isinstance(obj, type):
        return re.search("(?<=<class ').*(?='>)", str(obj))[0]
    return re.search("(?<=<class ').*(?='>)", str(type(obj)))[0]


def set_random_state_old(random_state: int | None, object_with_rng: object) -> None:
    """Sets the random state for all rng objects in the explainer.

    Args:
        random_state: The random state to re-initialize, Explainer, Imputer and Approximator with.
            Defaults to ``None`` which does not change the random state.
        object_with_rng: The object to set the random state for.
    """
    # TODO(mmshlk): write semantic test for this method
    if random_state is not None:
        if hasattr(object_with_rng, "_rng"):  # default attribute
            object_with_rng._rng = np.random.default_rng(random_state)  # noqa: SLF001
        # explainer can have an imputer
        if hasattr(object_with_rng, "_imputer"):
            object_with_rng._imputer._rng = np.random.default_rng(random_state)  # noqa: SLF001
        # explainer can have an approximator
        if hasattr(object_with_rng, "_approximator"):
            object_with_rng._approximator._rng = np.random.default_rng(random_state)  # noqa: SLF001
            # approximators inside an explainer can have a sampler
            if hasattr(object_with_rng._approximator, "_sampler"):  # noqa: SLF001
                object_with_rng._approximator._sampler._rng = np.random.default_rng(random_state)  # noqa: SLF001
        # appoximators can have a sampler
        if hasattr(object_with_rng, "_sampler"):
            object_with_rng._sampler._rng = np.random.default_rng(random_state)  # noqa: SLF001


def set_random_state(random_state: int | None, object_with_rng: object) -> None:
    """Sets the random state for all random number generator objects recursively.

    This function searches for attributes named "_rng" or "rng" in the given object and its
    nested attributes and re-initializes them with the specified random state.

    Args:
        random_state: The random state to use for reinitialization.
            Defaults to ``None`` which does not change any random state.
        object_with_rng: The object to inspect and modify random states for.
    """
    if random_state is None:
        return

    from shapiq.approximator._base import Approximator
    from shapiq.explainer._base import Explainer
    from shapiq.games.base import Game
    from shapiq.games.imputer.base import Imputer

    # set to avoid circular references
    visited = set()

    def _is_shapiq_object(obj: object) -> bool:
        """Check if the object is from shapiq library."""
        return isinstance(obj, (Explainer | Game | Imputer | Approximator))

    def _set_rng_recursive(obj: object) -> None:
        # avoid circular references or None objects
        if obj is None or id(obj) in visited:
            return

        visited.add(id(obj))

        # only process shapiq objects
        if not _is_shapiq_object(obj):
            return

        # set RNG attributes directly
        for attr_name in ["_rng", "rng"]:
            if hasattr(obj, attr_name):
                setattr(obj, attr_name, np.random.default_rng(random_state))

        # process child objects that might be shapiq objects
        for attr_name in dir(obj):
            # skip dunder attributes and methods
            if attr_name.startswith("__") or callable(getattr(obj.__class__, attr_name, None)):
                continue

            try:
                attr_value = getattr(obj, attr_name)
                # Recursively process object attributes that are shapiq objects
                if hasattr(attr_value, "__dict__"):
                    _set_rng_recursive(attr_value)
            except (AttributeError, TypeError):
                continue

    _set_rng_recursive(object_with_rng)
