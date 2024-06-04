"""Functions for converting scikit-learn decision trees to the format used by
shapiq."""

from typing import Optional

import numpy as np

from shapiq.utils import safe_isinstance
from shapiq.utils.types import Model

from ..base import TreeModel


def convert_sklearn_forest(
    tree_model: Model,
    class_label: Optional[int] = None,
) -> list[TreeModel]:
    """Transforms a scikit-learn random forest to the format used by shapiq.

    Args:
        tree_model: The scikit-learn random forest model to convert.
        class_label: The class label of the model to explain. Only used for classification models.
            Defaults to ``1``.

    Returns:
        The converted random forest model.
    """
    scaling = 1.0 / len(tree_model.estimators_)
    return [
        convert_sklearn_tree(tree, scaling=scaling, class_label=class_label)
        for tree in tree_model.estimators_
    ]


def convert_sklearn_tree(
    tree_model: Model, class_label: Optional[int] = None, scaling: float = 1.0
) -> TreeModel:
    """Convert a scikit-learn decision tree to the format used by shapiq.

    Args:
        tree_model: The scikit-learn decision tree model to convert.
        class_label: The class label of the model to explain. Only used for classification models.
            Defaults to ``1``.
        scaling: The scaling factor for the tree values.

    Returns:
        The converted decision tree model.
    """
    output_type = "raw"
    tree_values = tree_model.tree_.value.copy() * scaling
    # set class label if not given and model is a classifier
    if (
        safe_isinstance(tree_model, "sklearn.tree.DecisionTreeClassifier")
        or safe_isinstance(tree_model, "sklearn.tree._classes.DecisionTreeClassifier")
    ) and class_label is None:
        class_label = 1

    if class_label is not None:
        # turn node values into probabilities
        if len(tree_values.shape) == 3:
            tree_values = tree_values[:, 0, :]
        tree_values = tree_values / np.sum(tree_values, axis=1, keepdims=True)
        tree_values = tree_values[:, class_label]
        output_type = "probability"
    tree_values = tree_values.flatten()
    return TreeModel(
        children_left=tree_model.tree_.children_left,
        children_right=tree_model.tree_.children_right,
        features=tree_model.tree_.feature,
        thresholds=tree_model.tree_.threshold,
        values=tree_values,
        node_sample_weight=tree_model.tree_.weighted_n_node_samples,
        empty_prediction=None,  # compute empty prediction later
        original_output_type=output_type,
    )
