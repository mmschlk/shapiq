"""Functions for converting scikit-learn decision trees to the format used by shapiq."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.explainer.tree.base import TreeModel
from shapiq.utils import safe_isinstance

if TYPE_CHECKING:
    from shapiq.utils.custom_types import Model


def convert_sklearn_forest(
    tree_model: Model,
    class_label: int | None = None,
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
    tree_model: Model,
    class_label: int | None = None,
    scaling: float = 1.0,
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
    tree_values = tree_model.tree_.value.copy()
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
    tree_values *= scaling
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


def average_path_length(isolation_forest: Model) -> float:
    """Compute the average path length of the isolation forest.

    Args:
        isolation_forest: The isolation forest model.

    Returns:
        The average path length of the isolation forest.

    """
    from sklearn.ensemble._iforest import _average_path_length

    max_samples = isolation_forest._max_samples  # noqa: SLF001
    return _average_path_length([max_samples])


def convert_sklearn_isolation_forest(
    tree_model: Model,
) -> list[TreeModel]:
    """Transforms a scikit-learn isolation forest to the format used by shapiq.

    Args:
        tree_model: The scikit-learn isolation forest model to convert.

    Returns:
        The converted isolation forest model.

    """
    scaling = 1.0 / len(tree_model.estimators_)

    return [
        convert_isolation_tree(tree, features, scaling=scaling)
        for tree, features in zip(
            tree_model.estimators_,
            tree_model.estimators_features_,
            strict=False,
        )
    ]


def convert_isolation_tree(
    tree_model: Model,
    tree_features: np.ndarray,
    scaling: float = 1.0,
) -> TreeModel:
    """Convert a scikit-learn decision tree to the format used by shapiq.

    Args:
        tree_model: The scikit-learn decision tree model to convert.
        tree_features: The features used in the tree.
        scaling: The scaling factor for the tree values.

    Returns:
        The converted decision tree model.

    """
    output_type = "raw"
    features_updated, values_updated = isotree_value_traversal(
        tree_model.tree_,
        tree_features,
        normalize=False,
        scaling=1.0,
    )
    values_updated = values_updated * scaling
    values_updated = values_updated.flatten()

    return TreeModel(
        children_left=tree_model.tree_.children_left,
        children_right=tree_model.tree_.children_right,
        features=features_updated,
        thresholds=tree_model.tree_.threshold,
        values=values_updated,
        node_sample_weight=tree_model.tree_.weighted_n_node_samples,
        empty_prediction=None,  # compute empty prediction later
        original_output_type=output_type,
    )


def isotree_value_traversal(
    tree: Model,
    tree_features: np.ndarray,
    *,
    normalize: bool = False,
    scaling: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Traverse the tree and calculate the average path length for each node.

    Args:
        tree: The tree to traverse.
        tree_features: The features used in the tree.
        normalize: Whether to normalize the values.
        scaling: The scaling factor for the values.

    Returns:
        The updated features and values.

    """
    from sklearn.ensemble._iforest import _average_path_length

    features = tree.feature.copy()
    corrected_values = tree.value.copy()
    if safe_isinstance(tree, "sklearn.tree._tree.Tree"):

        def _recalculate_value(tree: Model, i: int, level: int = 0) -> float:
            if tree.children_left[i] == -1 and tree.children_right[i] == -1:
                value = level + _average_path_length(np.array([tree.n_node_samples[i]]))[0]
                corrected_values[i, 0] = value
                return value * tree.n_node_samples[i]
            value_left = _recalculate_value(tree, tree.children_left[i], level + 1)
            value_right = _recalculate_value(tree, tree.children_right[i], level + 1)
            corrected_values[i, 0] = (value_left + value_right) / tree.n_node_samples[i]
            return value_left + value_right

        _recalculate_value(tree, 0, 0)
        if normalize:
            corrected_values = (corrected_values.T / corrected_values.sum(1)).T
        corrected_values = corrected_values * scaling
        # re-number the features if each tree gets a different set of features
        features = np.where(features >= 0, tree_features[features], features)
    return features, corrected_values
