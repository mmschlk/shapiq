"""This module contains conversion functions for the tree explainer implementation."""
from typing import Any, Optional

from explainer.tree.conversion import convert_sklearn_tree, TreeModel
from shapiq.utils import safe_isinstance

SUPPORTED_MODELS = {
    "sklearn.tree.DecisionTreeRegressor",
    "sklearn.tree.DecisionTreeClassifier",
}


def _validate_model(
    model: Any,
    class_label: Optional[int] = None,
) -> TreeModel:
    """Validate the model.

    Args:
        model: The model to validate.
        class_label: The class label of the model to explain. Only used for classification models.

    Returns:
        The validated model.
    """
    if safe_isinstance(model, "sklearn.tree.DecisionTreeRegressor") or safe_isinstance(
        model, "sklearn.tree.DecisionTreeClassifier"
    ):
        if safe_isinstance(model, "sklearn.tree.DecisionTreeClassifier") and class_label is None:
            class_label = 1
        converted_model = convert_sklearn_tree(model, class_label=class_label)
    else:
        raise TypeError("Unsupported model type." f"Supported models are: {SUPPORTED_MODELS}")

    return converted_model
