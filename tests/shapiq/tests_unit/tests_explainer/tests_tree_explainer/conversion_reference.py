"""Pure-Python reference converters for testing C++ tree conversion paths."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import binom

from shapiq.tree.base import EdgeTree, TreeModel

if TYPE_CHECKING:
    from shapiq.typing import FloatVector, IntVector


def create_edge_tree_python(
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
    """Reference implementation of EdgeTree construction used to test the C++ converter."""
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

    if children_left[0] == -1:
        empty_predictions[0] = values[0]
        edge_heights[0] = 0
        return EdgeTree(
            parents=parents,
            ancestors=ancestors,
            ancestor_nodes=ancestor_nodes,
            p_e_values=p_e_values,
            p_e_storages=p_e_storages,
            split_weights=split_weights,
            empty_predictions=empty_predictions,
            edge_heights=edge_heights,
            max_depth=0,
            last_feature_node_in_path=np.full_like(children_left, fill_value=True, dtype=bool),
            interaction_height_store=interaction_height_store,
        )

    last_feature_node_in_path: np.ndarray = np.full_like(
        children_left, fill_value=False, dtype=bool
    )

    def recursive_search(
        node_id: int = 0,
        depth: int = 0,
        prod_weight: float = 1.0,
        seen_features: np.ndarray | None = None,
    ) -> int:
        if seen_features is None:
            seen_features = np.full(n_features, -1, dtype=int)

        max_depth[0] = max(max_depth[0], depth)
        left_child, right_child = children_left[node_id], children_right[node_id]
        is_leaf = left_child == -1
        if not is_leaf:
            parents[left_child], parents[right_child] = node_id, node_id

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
            return edge_heights[node_id]

        ancestor_nodes[node_id] = seen_features.copy()
        feature_id = features[parents[node_id]]
        last_feature_node_in_path[node_id] = True

        weight = node_sample_weight[node_id] / node_sample_weight[parents[node_id]]
        split_weights[node_id] = weight
        prod_weight *= weight
        p_e = 1 / weight

        for order in range(1, max_interaction + 1):
            interaction_height_store[order][node_id] = interaction_height_store[order][
                parents[node_id]
            ].copy()

        if seen_features[feature_id] > -1:
            ancestor_id = seen_features[feature_id]
            ancestors[node_id] = ancestor_id
            last_feature_node_in_path[ancestor_id] = False
            p_e *= p_e_values[ancestor_id]
        else:
            for order in range(1, max_interaction + 1):
                indices_to_update = subset_updates_pos_store[order][int(feature_id)]
                interaction_height_store[order][node_id][indices_to_update] += 1

        p_e_values[node_id] = p_e
        p_e_storages[node_id] = p_e_storages[parents[node_id]].copy()
        p_e_storages[node_id][feature_id] = p_e
        seen_features[feature_id] = node_id

        if not is_leaf:
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
        else:
            edge_heights[node_id] = np.sum(seen_features > -1)
            empty_predictions[node_id] = prod_weight * values[node_id]
        return edge_heights[node_id]

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


def _get_float_feature_nan_treatments(model_json: dict[str, Any]) -> dict[int, str]:
    float_features = model_json.get("features_info", {}).get("float_features", [])
    return {
        int(feature["feature_index"]): feature.get("nan_value_treatment", "AsIs")
        for feature in float_features
    }


def _leaf_values_for_class(
    leaf_values: list[float],
    leaf_count: int,
    class_label: int | None,
) -> np.ndarray:
    class_count = len(leaf_values) // leaf_count
    if len(leaf_values) % leaf_count != 0 or class_count < 1:
        msg = "CatBoost leaf_values length is incompatible with the number of leaves."
        raise ValueError(msg)
    if class_count == 1:
        return np.asarray(leaf_values, dtype=np.float32)
    if class_label is None:
        class_label = 1
    if class_label < 0 or class_label >= class_count:
        msg = f"class_label must be in [0, {class_count - 1}], got {class_label}."
        raise ValueError(msg)
    return np.asarray(
        [leaf_values[leaf_index * class_count + class_label] for leaf_index in range(leaf_count)],
        dtype=np.float32,
    )


def _catboost_oblivious_tree_to_treemodel(
    tree: dict[str, Any],
    nan_treatments: dict[int, str],
    class_label: int | None,
    *,
    scaling: float = 1.0,
    bias: float = 0.0,
) -> TreeModel:
    splits = tree.get("splits", [])
    depth = len(splits)
    leaf_count = 1 << depth
    internal_count = leaf_count - 1
    node_count = internal_count + leaf_count

    children_left = np.full(node_count, -1, dtype=np.int64)
    children_right = np.full(node_count, -1, dtype=np.int64)
    children_missing = np.full(node_count, -1, dtype=np.int64)
    features = np.full(node_count, -2, dtype=np.int64)
    thresholds = np.full(node_count, np.nan, dtype=np.float32)
    values = np.zeros(node_count, dtype=np.float32)
    node_sample_weight = np.zeros(node_count, dtype=np.float32)

    leaf_values = _leaf_values_for_class(tree.get("leaf_values", []), leaf_count, class_label)
    leaf_weights = np.asarray(tree.get("leaf_weights", [1.0] * leaf_count), dtype=np.float32)
    if leaf_weights.size != leaf_count:
        msg = "CatBoost leaf_weights length is incompatible with the number of leaves."
        raise ValueError(msg)

    def build_node(level: int, leaf_index: int) -> int:
        if level == depth:
            node_id = internal_count + leaf_index
            values[node_id] = scaling * leaf_values[leaf_index] + bias
            node_sample_weight[node_id] = leaf_weights[leaf_index]
            return node_id

        node_id = (1 << level) - 1 + leaf_index
        split = splits[level]
        if split.get("split_type") != "FloatFeature":
            msg = (
                "Only CatBoost JSON models with FloatFeature splits are supported. "
                f"Got split_type={split.get('split_type')!r}."
            )
            raise NotImplementedError(msg)
        feature_id = int(split["float_feature_index"])
        left_child = build_node(level + 1, leaf_index)
        right_child = build_node(level + 1, leaf_index | (1 << level))
        nan_treatment = nan_treatments.get(feature_id, "AsIs")

        children_left[node_id] = left_child
        children_right[node_id] = right_child
        children_missing[node_id] = right_child if nan_treatment == "AsTrue" else left_child
        features[node_id] = feature_id
        thresholds[node_id] = float(split["border"])
        node_sample_weight[node_id] = (
            node_sample_weight[left_child] + node_sample_weight[right_child]
        )
        return node_id

    if depth == 0:
        values[0] = scaling * leaf_values[0] + bias
        node_sample_weight[0] = leaf_weights[0]
    else:
        build_node(level=0, leaf_index=0)

    return TreeModel(
        children_left=children_left,
        children_right=children_right,
        children_missing=children_missing,
        features=features,
        thresholds=thresholds,
        values=values,
        node_sample_weight=node_sample_weight,
        original_output_type="raw",
    )


def parse_catboost_json_model_python(
    model_json: dict[str, Any],
    class_label: int | None = None,
) -> list[TreeModel]:
    """Reference CatBoost JSON converter used to test the C++ converter."""
    oblivious_trees = model_json.get("oblivious_trees")
    if oblivious_trees is None:
        msg = "Expected CatBoost JSON model with an 'oblivious_trees' entry."
        raise ValueError(msg)
    nan_treatments = _get_float_feature_nan_treatments(model_json)
    scale_and_bias = model_json.get("scale_and_bias", [1.0, [0.0]])
    scaling = float(scale_and_bias[0])
    bias_values = scale_and_bias[1]
    effective_class_label = 1 if class_label is None and len(bias_values) > 1 else class_label
    effective_class_label = 0 if effective_class_label is None else effective_class_label
    bias = float(bias_values[effective_class_label]) / len(oblivious_trees)
    return [
        _catboost_oblivious_tree_to_treemodel(
            tree,
            nan_treatments,
            class_label,
            scaling=scaling,
            bias=bias,
        )
        for tree in oblivious_trees
    ]
