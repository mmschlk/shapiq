"""This module contains functions for converting scikit-learn decision trees to the format used by
 shapiq."""

from typing import Union, Optional

import numpy as np

try:
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
except ImportError:
    pass

from .base import TreeModel


def convert_sklearn_tree(
    tree_model: Union["DecisionTreeRegressor", "DecisionTreeClassifier"],
    class_label: int = 0,
    scaling: float = 1.0,
) -> TreeModel:
    """Convert a scikit-learn decision tree to the format used by shapiq.

    Args:
        tree_model: The scikit-learn decision tree model to convert.
        class_label: The class label of the model to explain. Only used for classification models.
            Defaults to 0.
        scaling: The scaling factor for the tree values.

    Returns:
        The converted decision tree model.
    """
    tree_values = tree_model.tree_.value.copy() * scaling
    if class_label is not None:
        # turn node values into probabilities
        if len(tree_values.shape) == 3:
            tree_values = tree_values / np.sum(tree_values, axis=2, keepdims=True)
            tree_values = tree_values[:, 0, class_label]
        else:
            tree_values = tree_values / np.sum(tree_values, axis=1, keepdims=True)
            tree_values = tree_values[:, class_label]
    tree_values = tree_values.flatten()
    return TreeModel(
        children_left=tree_model.tree_.children_left,
        children_right=tree_model.tree_.children_right,
        features=tree_model.tree_.feature,
        thresholds=tree_model.tree_.threshold,
        values=tree_values,
        node_sample_weight=tree_model.tree_.weighted_n_node_samples,
        empty_prediction=None,  # compute empty prediction later
    )
