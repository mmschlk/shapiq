"""Conversion functions to parse a :class:`~shapiq.tree.base.TreeModel` into the :class:`~shapiq.tree.base.EdgeTree` format."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.tree.base import EdgeTree

from .cext import create_edge_tree_arrays  # ty: ignore[unresolved-import]

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
    """Create an ``EdgeTree`` using the C++ extension.

    Parses the tree to create the edge-based representation used by TreeSHAP-IQ and
    pre-calculates edge weights, ancestor references, empty predictions, and interaction
    height counts up to ``max_interaction``.
    """
    (
        parents,
        ancestors,
        ancestor_nodes_dense,
        p_e_values,
        p_e_storages,
        split_weights,
        empty_predictions,
        edge_heights,
        max_depth,
        last_feature_node_in_path,
        interaction_height_store,
    ) = create_edge_tree_arrays(
        np.asarray(children_left, dtype=np.int64),
        np.asarray(children_right, dtype=np.int64),
        np.asarray(features, dtype=np.int64),
        np.asarray(node_sample_weight, dtype=float),
        np.asarray(values, dtype=float),
        n_nodes,
        n_features,
        max_interaction,
        {
            order: {
                feature: np.asarray(indices, dtype=np.int64)
                for feature, indices in feature_positions.items()
            }
            for order, feature_positions in subset_updates_pos_store.items()
        },
    )
    # TreeSHAP-IQ precomputes interaction ancestors for every non-root node, including leaves.
    # A constant single-node tree has no non-root nodes and does not need ancestor entries.
    ancestor_nodes = {}
    if children_left[0] != -1:
        ancestor_nodes = {node_id: ancestor_nodes_dense[node_id] for node_id in range(1, n_nodes)}
    return EdgeTree(
        parents=parents,
        ancestors=ancestors,
        ancestor_nodes=ancestor_nodes,
        p_e_values=p_e_values,
        p_e_storages=p_e_storages,
        split_weights=split_weights,
        empty_predictions=empty_predictions,
        edge_heights=edge_heights,
        max_depth=max_depth,
        last_feature_node_in_path=last_feature_node_in_path,
        interaction_height_store=interaction_height_store,
    )
