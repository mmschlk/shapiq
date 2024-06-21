"""Functions for converting scikit-learn decision trees to the format used by
shapiq."""

from typing import Optional

import numpy as np

from sklearn.ensemble._iforest import _average_path_length

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

def average_path_length(isolation_forest):
    max_samples = isolation_forest._max_samples
    average_path_length = _average_path_length([max_samples]) # NOTE: _average_path_length func is equivalent to equation 1 in Isolation Forest paper Lui2008
    return average_path_length

def compute_anomaly_score(tree_model, average_path_length):
    # Basic implementation based on scores and normalized using average path length
    depths = tree_model.tree_.compute_node_depths()
    depths = depths / average_path_length
    depths = 1 - depths
    scores = depths
    return scores

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
    avg_length = average_path_length(tree_model)

    return [
        convert_isolation_tree(tree, scaling=scaling, average_path_length=avg_length)
        for tree in tree_model.estimators_
    ]

def convert_isolation_tree(
            tree_model: Model, class_label: Optional[int] = None, scaling: float = 1.0, average_path_length: float = 1.0 # TODO fix default value
) -> TreeModel:
    """Convert a scikit-learn isolation tree to the format used by shapiq.

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
    depths = compute_anomaly_score(tree_model, average_path_length)
    depths = depths * scaling

    return TreeModel(
        children_left=tree_model.tree_.children_left,
        children_right=tree_model.tree_.children_right,
        features=tree_model.tree_.feature,
        thresholds=tree_model.tree_.threshold,
        values=tree_values,
        # values=depths,
        node_sample_weight=tree_model.tree_.weighted_n_node_samples,
        empty_prediction=None,  # compute empty prediction later
        original_output_type=output_type,
    )

def convert_sklearn_isolation_forest_shap(
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
        convert_isolation_tree_shap_isotree(tree, features, scaling=scaling)
        for tree, features in zip(tree_model.estimators_, tree_model.estimators_features_)
    ]


def convert_isolation_tree_shap_isotree(
            tree_model: Model, tree_features, class_label: Optional[int] = None, scaling: float = 1.0, average_path_length: float = 1.0 # TODO fix default value
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
    # tree_values = tree_model.tree_.value.copy()
    features_updated, values_updated = isotree_value_traversal(tree_model.tree_, tree_features, normalize=False, scaling=1.0)

    values_updated = values_updated * scaling
    values_updated = values_updated.flatten()

    return TreeModel(
        children_left=tree_model.tree_.children_left,
        children_right=tree_model.tree_.children_right,
        features=features_updated,
        thresholds=tree_model.tree_.threshold,
        # values=tree_values,
        values=values_updated,
        node_sample_weight=tree_model.tree_.weighted_n_node_samples,
        empty_prediction=None,  # compute empty prediction later
        original_output_type=output_type,
    )

def isotree_value_traversal(tree, tree_features, normalize=False, scaling=1.0, data=None, data_missing=None):
    features = tree.feature.copy()
    corrected_values = tree.value.copy()
    if safe_isinstance(tree, "sklearn.tree._tree.Tree"):

        def _recalculate_value(tree, i , level):
            if tree.children_left[i] == -1 and tree.children_right[i] == -1:
                value = level + _average_path_length(np.array([tree.n_node_samples[i]]))[0]
                corrected_values[i, 0] =  value
                return value * tree.n_node_samples[i]
            else:
                value_left = _recalculate_value(tree, tree.children_left[i] , level + 1)
                value_right = _recalculate_value(tree, tree.children_right[i] , level + 1)
                corrected_values[i, 0] =  (value_left + value_right) / tree.n_node_samples[i]
                return value_left + value_right

        _recalculate_value(tree, 0, 0)
        if normalize:
            corrected_values = (corrected_values.T / corrected_values.sum(1)).T
        corrected_values = corrected_values * scaling
        # re-number the features if each tree gets a different set of features
        features = np.where(features >= 0, tree_features[features], features)
    return features, corrected_values