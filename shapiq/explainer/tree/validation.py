"""This module contains conversion functions for the tree explainer implementation."""

import warnings
from typing import Any, Optional, Union

from shapiq.utils import safe_isinstance

from .base import TreeModel, convert_tree_output_type
from .conversion.sklearn import convert_sklearn_forest, convert_sklearn_tree

SUPPORTED_MODELS = {
    "sklearn.tree.DecisionTreeRegressor",
    "sklearn.tree.DecisionTreeClassifier",
    "sklearn.ensemble.RandomForestClassifier",
    "sklearn.ensemble.RandomForestRegressor",
}


def validate_tree_model(
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
    if output_type not in ["raw", "probability", "logit"]:
        raise ValueError(
            "Invalid output type. Supported output types are: 'raw', 'probability', 'logit'."
        )

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

    # adapt output type if necessary
    if output_type != "raw":
        # check if the output type of the tree model is the same as the requested output type
        trees_to_adapt = []
        for i, tree in enumerate(tree_model):
            if tree.original_output_type != output_type:
                trees_to_adapt.append(i)
        if trees_to_adapt:
            warn_flag = False
            for i in trees_to_adapt:
                tree_model[i], warn_flag = convert_tree_output_type(tree_model[i], output_type)
                warn_flag += warn_flag
            # at least one tree model was adapted (invalid probability values were adjusted in
            # logit transformation)
            if warn_flag:
                warnings.warn(
                    UserWarning(
                        "Invalid probability values (i.e. p=0 or p=1) were numerically adjusted "
                        "in logit transformation (+/- inf is set to +/- 14)."
                    )
                )

    if len(tree_model) == 1:
        tree_model = tree_model[0]

    return tree_model
