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


def _validate_model(
    model: Any, class_label: Optional[int] = None, output_type: str = "raw"
) -> Union[TreeModel, list[TreeModel]]:
    """Validate the model.

    Args:
        model: The model to validate.
        class_label: The class label of the model to explain. Only used for classification models.
        output_type: The output type of the model. Can be "raw" (default), "probability", or "logit".  # TODO: add support for "probability" and "logit"

    Returns:
        The validated model and the model function.
    """
    # tree model (is already in the correct format)
    if type(model).__name__ == "TreeModel":
        return model
    # dict as model is parsed to TreeModel (the dict needs to have the correct format and names)
    if type(model).__name__ == "dict":
        return TreeModel(**model)
    # sklearn decision trees
    if safe_isinstance(model, "sklearn.tree.DecisionTreeRegressor") or safe_isinstance(
        model, "sklearn.tree.DecisionTreeClassifier"
    ):
        return convert_sklearn_tree(model, class_label=class_label, output_type=output_type)
    # sklearn random forests
    if safe_isinstance(model, "sklearn.ensemble.RandomForestRegressor") or safe_isinstance(
        model, "sklearn.ensemble.RandomForestClassifier"
    ):
        return convert_sklearn_forest(model, class_label=class_label, output_type=output_type)
    # unsupported model
    raise TypeError("Unsupported model type." f"Supported models are: {SUPPORTED_MODELS}")
