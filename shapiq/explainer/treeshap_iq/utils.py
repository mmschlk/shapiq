"""This module contains utility functions for dealing with trees or tree structures."""

import numpy as np

__all__ = ["get_conditional_sample_weights", "compute_empty_prediction"]


def get_conditional_sample_weights(
    sample_count: np.ndarray[int],
    parent_array: np.ndarray[int],
) -> np.ndarray[float]:
    """Get the conditional sample weights for a tree at each decision node.

    The conditional sample weights are the probabilities of going left or right at each decision
    node. The probabilities are computed by the number of instances going through each node
    divided by the number of instances going through the parent node. The conditional sample
    weights of the root node is 1.

    Args:
        sample_count: The counts of the instances going through each node.
        parent_array: The parent array denoting the id of the parent node for each node in the tree.
            The parent of the root node is -1 or otherwise specified.

    Returns:
        The conditional sample weights of the nodes.

    Examples:
        >>> parent_array = np.asarray([-1, 0, 1, 1, 0, 4, 4])  # binary tree with depth 2
        >>> sample_count = np.asarray([100, 70, 50, 20, 30, 15, 15])
        >>> get_conditional_sample_weights(sample_count, parent_array)
        >>> [1., 0.7, 0.71428571, 0.28571429, 0.3, 0.5, 0.5]
    """
    conditional_sample_weights = np.zeros_like(sample_count, dtype=float)
    conditional_sample_weights[0] = 1
    parent_sample_count = sample_count[parent_array[1:]]
    conditional_sample_weights[1:] = sample_count[1:] / parent_sample_count
    return conditional_sample_weights


def compute_empty_prediction(
    leaf_values: np.ndarray[float], leaf_sample_weights: np.ndarray[float]
) -> float:
    """Compute the empty prediction of a tree model.

    The empty prediction is the weighted average of the leaf node values.

    Args:
        leaf_values: The values of the leaf nodes in the tree.
        leaf_sample_weights: The sample weights of the leaf nodes in the tree.

    Returns:
        The empty prediction of the tree model.
    """
    return np.sum(leaf_values * leaf_sample_weights) / np.sum(leaf_sample_weights)
