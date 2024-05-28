"""Conversion functions for the tree explainer implementation."""

from typing import Any, Optional, Union

from shapiq.utils import safe_isinstance

from .base import TreeModel
from .conversion.lightgbm import convert_lightgbm_booster
from .conversion.sklearn import convert_sklearn_forest, convert_sklearn_tree
from .conversion.xgboost import convert_xgboost_booster

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
    "lightgbm.sklearn.LGBMRegressor",
    "lightgbm.sklearn.LGBMClassifier",
    "lightgbm.basic.Booster",
}


def validate_tree_model(
    model: Any, class_label: Optional[int] = None
) -> Union[TreeModel, list[TreeModel]]:
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
        or safe_isinstance(model, "sklearn.ensemble.ExtraTreesClassifier")
        or safe_isinstance(model, "sklearn.ensemble._forest.ExtraTreesClassifier")
    ):
        tree_model = convert_sklearn_forest(model, class_label=class_label)
    elif safe_isinstance(model, "lightgbm.sklearn.LGBMRegressor") or safe_isinstance(
        model, "lightgbm.sklearn.LGBMClassifier"
    ):
        tree_model = convert_lightgbm_booster(model.booster_, class_label=class_label)
    elif safe_isinstance(model, "lightgbm.basic.Booster"):
        tree_model = convert_lightgbm_booster(model, class_label=class_label)
    elif safe_isinstance(model, "xgboost.sklearn.XGBRegressor") or safe_isinstance(
        model, "xgboost.sklearn.XGBClassifier"
    ):
        tree_model = convert_xgboost_booster(model.get_booster(), class_label=class_label)
    elif safe_isinstance(model, "xgboost.core.Booster"):
        tree_model = convert_xgboost_booster(model, class_label=class_label)
    # unsupported model
    else:
        raise TypeError("Unsupported model type." f"Supported models are: {SUPPORTED_MODELS}")

    # if single tree model put it in a list
    if not isinstance(tree_model, list):
        tree_model = [tree_model]

    if len(tree_model) == 1:
        tree_model = tree_model[0]

    return tree_model
