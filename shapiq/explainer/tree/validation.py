"""This module contains conversion functions for the tree explainer implementation."""
from typing import Any, Optional

from shapiq.utils import safe_isinstance

from .base import TreeModel
from .conversion.sklearn import convert_sklearn_tree

SUPPORTED_MODELS = {
    "sklearn.tree.DecisionTreeRegressor",
    "sklearn.tree.DecisionTreeClassifier",
}


def _validate_model(
    model: Any, class_label: Optional[int] = None, output_type: str = "raw"
) -> TreeModel:
    """Validate the model.

    Args:
        model: The model to validate.
        class_label: The class label of the model to explain. Only used for classification models.
        output_type: The output type of the model. Can be "raw" (default), "probability", or "logit".  # TODO: add support for "probability" and "logit"

    Returns:
        The validated model and the model function.
    """
    if isinstance(model, TreeModel):
        return model
    if safe_isinstance(model, "sklearn.tree.DecisionTreeRegressor") or safe_isinstance(
        model, "sklearn.tree.DecisionTreeClassifier"
    ):
        if safe_isinstance(model, "sklearn.tree.DecisionTreeClassifier") and class_label is None:
            class_label = 1
        return convert_sklearn_tree(model, class_label=class_label)
    else:
        raise TypeError("Unsupported model type." f"Supported models are: {SUPPORTED_MODELS}")
