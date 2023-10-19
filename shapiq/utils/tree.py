"""This module contains utility functions for dealing with trees or tree structures."""

import numpy as np


__all__ = [
    "get_parent_array",
    "get_conditional_sample_weights",
]


def get_parent_array(
        children_left: np.ndarray[int],
        children_right: np.ndarray[int]
) -> np.ndarray[int]:
    """Combines the left and right children of a tree to a parent array. The parent of the root
    node is -1.

    Args:
        children_left: The left children of each node in a tree. Leaf nodes are -1.
        children_right: The right children of each node in a tree. Leaf nodes are -1.

    Returns:
        The parent array of the tree. The parent of the root node is -1.
    """
    parent_array = np.full_like(children_left, -1)
    non_leaf_indices = np.logical_or(children_left != -1, children_right != -1)
    parent_array[children_left[non_leaf_indices]] = np.where(non_leaf_indices)[0]
    parent_array[children_right[non_leaf_indices]] = np.where(non_leaf_indices)[0]
    return parent_array


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
    """
    conditional_sample_weights = np.zeros_like(sample_count, dtype=float)
    conditional_sample_weights[0] = 1
    parent_sample_count = sample_count[parent_array[1:]]
    conditional_sample_weights[1:] = sample_count[1:] / parent_sample_count
    return conditional_sample_weights
