"""Conversion functions for the tree explainer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapiq.utils.modules import safe_isinstance

from .base import TreeModel
from .conversion.lightgbm import convert_lightgbm_booster
from .conversion.sklearn import (
    convert_sklearn_forest,
    convert_sklearn_isolation_forest,
    convert_sklearn_tree,
)
from .conversion.xgboost import convert_xgboost_booster

if TYPE_CHECKING:
    from shapiq.utils.custom_types import Model

SUPPORTED_MODELS = {
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
    "lightgbm.sklearn.LGBMRegressor",
    "lightgbm.sklearn.LGBMClassifier",
    "lightgbm.basic.Booster",
    "xgboost.sklearn.XGBRegressor",
    "xgboost.sklearn.XGBClassifier",
}


def validate_tree_model(
    model: Model,
    class_label: int | None = None,
) -> TreeModel | list[TreeModel]:
    """Validate the model.

    Args:
        model: The model to validate.
        class_label: The class label of the model to explain. Only used for classification models.

    Returns:
        The validated model and the model function.

    """
    # direct returns for base tree models and dict as model
    # tree model (is already in the correct format)
    if type(model).__name__ == "TreeModel":
        tree_model = model
    # direct return if list of tree models
    elif type(model).__name__ == "list":
        # check if all elements are TreeModel
        if all(type(tree).__name__ == "TreeModel" for tree in model):
            tree_model = model
    # dict as model is parsed to TreeModel (the dict needs to have the correct format and names)
    elif type(model).__name__ == "dict":
        tree_model = TreeModel(**model)
    # transformation of common machine learning libraries to TreeModel
    # sklearn decision trees
    elif (
        safe_isinstance(model, "sklearn.tree.DecisionTreeRegressor")
        or safe_isinstance(model, "sklearn.tree._classes.DecisionTreeRegressor")
        or safe_isinstance(model, "sklearn.tree.DecisionTreeClassifier")
        or safe_isinstance(model, "sklearn.tree._classes.DecisionTreeClassifier")
    ):
        tree_model = convert_sklearn_tree(model, class_label=class_label)
    # sklearn random forests
    elif (
        safe_isinstance(model, "sklearn.ensemble.RandomForestRegressor")
        or safe_isinstance(model, "sklearn.ensemble._forest.RandomForestRegressor")
        or safe_isinstance(model, "sklearn.ensemble.RandomForestClassifier")
        or safe_isinstance(model, "sklearn.ensemble._forest.RandomForestClassifier")
        or safe_isinstance(model, "sklearn.ensemble.ExtraTreesRegressor")
        or safe_isinstance(model, "sklearn.ensemble._forest.ExtraTreesRegressor")
        or safe_isinstance(model, "sklearn.ensemble.ExtraTreesClassifier")
        or safe_isinstance(model, "sklearn.ensemble._forest.ExtraTreesClassifier")
    ):
        tree_model = convert_sklearn_forest(model, class_label=class_label)
    elif safe_isinstance(model, "sklearn.ensemble.IsolationForest") or safe_isinstance(
        model,
        "sklearn.ensemble._iforest.IsolationForest",
    ):
        tree_model = convert_sklearn_isolation_forest(model)
    elif safe_isinstance(model, "lightgbm.sklearn.LGBMRegressor") or safe_isinstance(
        model,
        "lightgbm.sklearn.LGBMClassifier",
    ):
        tree_model = convert_lightgbm_booster(model.booster_, class_label=class_label)
    elif safe_isinstance(model, "lightgbm.basic.Booster"):
        tree_model = convert_lightgbm_booster(model, class_label=class_label)
    elif safe_isinstance(model, "xgboost.sklearn.XGBRegressor") or safe_isinstance(
        model,
        "xgboost.sklearn.XGBClassifier",
    ):
        tree_model = convert_xgboost_booster(model, class_label=class_label)
    # unsupported model
    else:
        msg = f"Unsupported model type.Supported models are: {SUPPORTED_MODELS}"
        raise TypeError(msg)

    # if single tree model put it in a list
    if not isinstance(tree_model, list):
        tree_model = [tree_model]

    if len(tree_model) == 1:
        tree_model = tree_model[0]

    return tree_model
