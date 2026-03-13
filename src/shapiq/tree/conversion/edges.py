"""Conversion functions to parse a :class:`~shapiq.tree.base.TreeModel` into the :class:`~shapiq.tree.base.EdgeTree` format."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import binom

from shapiq.tree.base import EdgeTree

if TYPE_CHECKING:
    from shapiq.typing import FloatVector, IntVector


def create_edge_tree(
    children_left: IntVector,
    children_right: IntVector,
    features: IntVector,
    node_sample_weight: FloatVector,
    values: FloatVector,
    n_nodes: int,
    n_features: int,
    max_interaction: int,
    subset_updates_pos_store: dict[int, dict[int, IntVector]],
) -> EdgeTree:
    """Extract edge information recursively from the tree and return an ``EdgeTree``.

    Parses the tree recursively to create an edge-based representation. Pre-calculates
    the ``p_e`` and ``p_e_ancestors`` values for interaction subsets up to order
    ``max_interaction``.

    Args:
        children_left: The left children of each node. Leaf nodes are denoted with ``-1``.
        children_right: The right children of each node. Leaf nodes are denoted with ``-1``.
        features: The feature used for splitting at each node. Leaf nodes have the value ``-2``.
        node_sample_weight: The sample weights of each node in the tree.
        values: The output values at the leaf nodes of the tree.
        n_nodes: The number of nodes in the tree.
        n_features: The number of features in the dataset.
        max_interaction: The maximum interaction order to compute. An order of ``1`` corresponds
            to Shapley values; higher orders compute Shapley interaction values up to that order.
        subset_updates_pos_store: A dictionary mapping interaction order to a per-feature
            dictionary of index arrays indicating which interaction subsets must be incremented
            when a new occurrence of that feature is seen on a root-to-leaf path.

    Returns:
        A populated ``EdgeTree`` containing the edge-based representation of the tree,
        including parent/ancestor arrays, edge weights, empty predictions, and
        pre-computed interaction height counts.

    """
    # variables to be filled with recursive function
    parents = np.full(n_nodes, -1, dtype=int)
    ancestors: np.ndarray = np.full(n_nodes, -1, dtype=int)

    ancestor_nodes: dict[int, np.ndarray] = {}

    p_e_values: np.ndarray = np.ones(n_nodes, dtype=float)
    p_e_storages: np.ndarray = np.ones((n_nodes, n_features), dtype=float)
    split_weights: np.ndarray = np.ones(n_nodes, dtype=float)
    empty_predictions: np.ndarray = np.zeros(n_nodes, dtype=float)
    edge_heights: np.ndarray = np.full_like(children_left, -1, dtype=int)
    max_depth: list[int] = [0]
    interaction_height_store = {
        i: np.zeros((n_nodes, int(binom(n_features, i))), dtype=int)
        for i in range(1, max_interaction + 1)
    }

    features_last_seen_in_tree: dict[int, int] = {}
    last_feature_node_in_path: np.ndarray = np.full_like(
        children_left, fill_value=False, dtype=bool
    )

    def recursive_search(
        node_id: int = 0,
        depth: int = 0,
        prod_weight: float = 1.0,
        seen_features: np.ndarray | None = None,
    ) -> int:
        """Traverse the tree recursively and collect edge information for each node.

        Args:
            node_id: The current node id. Defaults to ``0`` (the root).
            depth: The depth of the current node in the tree. Defaults to ``0``.
            prod_weight: The cumulative product of split-probability weights on the path
                from the root to the current node. Defaults to ``1.0``.
            seen_features: An integer array of length ``n_features`` mapping each feature
                index to the node id where that feature was last observed on the current
                root-to-node path, or ``-1`` if not yet seen. Initialized to all ``-1``
                at the root on the first call (``None``).

        Returns:
            The edge height of the current node (the number of distinct features seen on
            the longest root-to-leaf path through this node's subtree).

        """
        # if root node, initialize seen_features and p_e_storage
        if seen_features is None:
            # map feature_id to ancestor node_id
            seen_features = np.full(n_features, -1, dtype=int)

        # update the maximum depth of the tree
        max_depth[0] = max(max_depth[0], depth)

        # set the parents of the children nodes
        left_child, right_child = children_left[node_id], children_right[node_id]
        is_leaf = left_child == -1
        if not is_leaf:
            parents[left_child], parents[right_child] = node_id, node_id
            features_last_seen_in_tree[int(features[node_id])] = node_id

        # if root_node, step into the tree and end recursion
        if node_id == 0:
            edge_heights_left = recursive_search(
                int(left_child),
                depth + 1,
                prod_weight,
                seen_features.copy(),
            )
            edge_heights_right = recursive_search(
                int(right_child),
                depth + 1,
                prod_weight,
                seen_features.copy(),
            )
            edge_heights[node_id] = max(edge_heights_left, edge_heights_right)
            return edge_heights[node_id]  # final return ending the recursion

        # node is not root node follow the path and compute weights

        ancestor_nodes[node_id] = seen_features.copy()

        # get the feature id of the current node
        feature_id = features[parents[node_id]]

        # Assume it is the last occurrence of feature
        last_feature_node_in_path[node_id] = True

        # compute prod_weight with node samples
        n_sample = node_sample_weight[node_id]
        n_parent = node_sample_weight[parents[node_id]]
        weight = n_sample / n_parent
        split_weights[node_id] = weight
        prod_weight *= weight

        # calculate the p_e value of the current node
        p_e = 1 / weight

        # copy parent height information
        for order in range(1, max_interaction + 1):
            interaction_height_store[order][node_id] = interaction_height_store[order][
                parents[node_id]
            ].copy()
        # correct if feature was seen before
        if seen_features[feature_id] > -1:  # feature has been seen before in the path
            ancestor_id = seen_features[feature_id]  # get ancestor node with same feature
            ancestors[node_id] = ancestor_id  # store ancestor node
            last_feature_node_in_path[ancestor_id] = False  # correct previous assumption
            p_e *= p_e_values[ancestor_id]  # add ancestor weight to p_e
        else:
            for order in range(1, max_interaction + 1):
                indices_to_update = subset_updates_pos_store[order][int(feature_id)]
                interaction_height_store[order][node_id][indices_to_update] += 1

        # store the p_e value of the current node
        p_e_values[node_id] = p_e
        p_e_storages[node_id] = p_e_storages[parents[node_id]].copy()
        p_e_storages[node_id][feature_id] = p_e

        # update seen features with current node
        seen_features[feature_id] = node_id

        # update the edge heights
        if not is_leaf:  # if node is not a leaf, continue recursion
            edge_heights_left = recursive_search(
                int(left_child),
                depth + 1,
                prod_weight,
                seen_features.copy(),
            )
            edge_heights_right = recursive_search(
                int(right_child),
                depth + 1,
                prod_weight,
                seen_features.copy(),
            )
            edge_heights[node_id] = max(edge_heights_left, edge_heights_right)
        else:  # if node is a leaf, end recursion
            edge_heights[node_id] = np.sum(seen_features > -1)
            empty_predictions[node_id] = prod_weight * values[node_id]
        return edge_heights[node_id]  # return upwards in the recursion

    _ = recursive_search()
    return EdgeTree(
        parents=parents,
        ancestors=ancestors,
        ancestor_nodes=ancestor_nodes,
        p_e_values=p_e_values,
        p_e_storages=p_e_storages,
        split_weights=split_weights,
        empty_predictions=empty_predictions,
        edge_heights=edge_heights,
        max_depth=max_depth[0],
        last_feature_node_in_path=last_feature_node_in_path,
        interaction_height_store=interaction_height_store,
    )
