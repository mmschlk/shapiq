"""This module contains the conversion functions to parse a tree model into the edge representation.
The edge representation is used by the TreeSHAP-IQ algorithm to compute the interaction values of a
tree-based model."""
import numpy as np
from scipy.special import binom

from ..base import EdgeTree


def create_edge_tree(
    children_left: np.ndarray[int],
    children_right: np.ndarray[int],
    features: np.ndarray[int],
    node_sample_weight: np.ndarray[float],
    values: np.ndarray[float],
    n_nodes: int,
    n_features: int,
    max_interaction: int,
    subset_updates_pos_store: dict[int, dict[int, np.ndarray[int]]],
):
    """Extracts edge information recursively from the tree information.

    Parses the tree recursively to create an edge-based representation of the tree. It
    precalculates the p_e and p_e_ancestors of the interaction subsets up to order
    'max_interaction'.

    Args:
        children_left (np.ndarray[int]): The left children of each node. Leaf nodes are denoted
            with -1.
        children_right (np.ndarray[int]): The right children of each node. Leaf nodes are denoted
            with -1.
        features (np.ndarray[int]): The feature used for splitting at each node. Leaf nodes have
            the value -2.
        node_sample_weight (np.ndarray[float]): The sample weights of the tree.
        values (np.ndarray[float]): The output values at the leaf values of the tree.
        n_nodes (int): The number of nodes in the tree.
        n_features (int): The number of features of the dataset.
        max_interaction (int, optional): The maximum interaction order to be computed. An
            interaction order of 1 corresponds to the Shapley value. Any value higher than 1
            computes the Shapley interactions values up to that order. Defaults to 1 (i.e. SV).
        subset_updates_pos_store (dict[int, np.ndarray[int]]): A dictionary containing the
            interaction subsets for each feature given an interaction order.

    Returns:
        EdgeTree: A dataclass containing the edge information of the tree.
    """
    # variables to be filled with recursive function
    parents = np.full(n_nodes, -1, dtype=int)
    ancestors: np.ndarray[int] = np.full(n_nodes, -1, dtype=int)

    ancestor_nodes: dict[int, np.ndarray[int]] = {}

    p_e_values: np.ndarray[float] = np.ones(n_nodes, dtype=float)
    p_e_storages: np.ndarray[float] = np.ones((n_nodes, n_features), dtype=float)
    split_weights: np.ndarray[float] = np.ones(n_nodes, dtype=float)
    empty_predictions: np.ndarray[float] = np.zeros(n_nodes, dtype=float)
    edge_heights: np.ndarray[int] = np.full_like(children_left, -1, dtype=int)
    max_depth: list[int] = [0]
    interaction_height_store = {
        i: np.zeros((n_nodes, int(binom(n_features, i))), dtype=int)
        for i in range(1, max_interaction + 1)
    }

    features_last_seen_in_tree: dict[int, int] = {}

    last_feature_node_in_path: np.ndarray[bool] = np.full_like(children_left, False, dtype=bool)

    def recursive_search(
        node_id: int = 0,
        depth: int = 0,
        prod_weight: float = 1.0,
        seen_features: np.ndarray[int] = None,
    ):
        """Traverses the tree recursively and collects all relevant information.

        Args:
            node_id (int): The current node id.
            depth (int): The depth of the current node.
            prod_weight (float): The product of the node weights on the path to the current
                node.
            seen_features (np.ndarray[int]): The features seen on the path to the current node.
                Maps the feature id to the node id where the feature was last seen on the way.

        Returns:
            int: The edge height of the current node.
        """
        # if root node, initialize seen_features and p_e_storage
        if seen_features is None:
            seen_features: np.ndarray[int] = np.full(
                n_features, -1, dtype=int
            )  # maps feature_id to ancestor node_id

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
                int(left_child), depth + 1, prod_weight, seen_features.copy()
            )
            edge_heights_right = recursive_search(
                int(right_child), depth + 1, prod_weight, seen_features.copy()
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
                int(left_child), depth + 1, prod_weight, seen_features.copy()
            )
            edge_heights_right = recursive_search(
                int(right_child), depth + 1, prod_weight, seen_features.copy()
            )
            edge_heights[node_id] = max(edge_heights_left, edge_heights_right)
        else:  # if node is a leaf, end recursion
            edge_heights[node_id] = np.sum(seen_features > -1)
            empty_predictions[node_id] = prod_weight * values[node_id]
        return edge_heights[node_id]  # return upwards in the recursion

    _ = recursive_search()
    edge_tree = EdgeTree(
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
    return edge_tree
