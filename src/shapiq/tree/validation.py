"""Conversion functions for the tree explainer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import TreeModel
from .conversion import convert_tree_model

if TYPE_CHECKING:
    from shapiq.typing import Model

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
) -> list[TreeModel]:
    """Validate the model.

    Args:
        model: The model to validate.
        class_label: The class label of the model to explain. Only used for classification models.

    Returns:
        The validated model and the model function.

    Raises:
        NotImplementedError: If the model type is not supported.
    """
    tree_model = []
    # direct returns for base tree models and dict as model
    # tree model (is already in the correct format)
    if type(model).__name__ == "TreeModel":
        tree_model = [model]
    # direct return if list of tree models
    elif type(model).__name__ == "list":
        # check if all elements are TreeModel
        if all(type(tree).__name__ == "TreeModel" for tree in model):
            tree_model = model
    # dict as model is parsed to TreeModel (the dict needs to have the correct format and names)
    elif type(model).__name__ == "dict":
        tree_model = [TreeModel(**model)]
    else:
        try:
            result = convert_tree_model(model, class_label=class_label)
        except NotImplementedError as e:
            msg = f"Model type {type(model)} is not supported."
            raise TypeError(msg) from e
        tree_model = result if isinstance(result, list) else [result]
    return tree_model
