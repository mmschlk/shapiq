"""This module contains conversion functions for the tree explainer implementation."""

from typing import Any, Optional, Union

from shapiq.utils import safe_isinstance

from .base import TreeModel
from .conversion.sklearn import convert_sklearn_forest, convert_sklearn_tree

SUPPORTED_MODELS = {
    "sklearn.tree.DecisionTreeRegressor",
    "sklearn.tree.DecisionTreeClassifier",
    "sklearn.ensemble.RandomForestClassifier",
    "sklearn.ensemble.RandomForestRegressor",
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
    elif safe_isinstance(model, "sklearn.tree.DecisionTreeRegressor") or safe_isinstance(
        model, "sklearn.tree.DecisionTreeClassifier"
    ):
        tree_model = convert_sklearn_tree(model, class_label=class_label)
    # sklearn random forests
    elif safe_isinstance(model, "sklearn.ensemble.RandomForestRegressor") or safe_isinstance(
        model, "sklearn.ensemble.RandomForestClassifier"
    ):
        tree_model = convert_sklearn_forest(model, class_label=class_label)
    # unsupported model
    else:
        raise TypeError("Unsupported model type." f"Supported models are: {SUPPORTED_MODELS}")

    # if single tree model put it in a list
    if not isinstance(tree_model, list):
        tree_model = [tree_model]

    if len(tree_model) == 1:
        tree_model = tree_model[0]

    return tree_model
