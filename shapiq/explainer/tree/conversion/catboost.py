"""Functions for converting catboost decision trees to the format used by shapiq."""

from __future__ import annotations

import numpy as np

from ....utils import safe_isinstance
from ....utils.custom_types import Model
from ..base import TreeModel


def convert_catboost(
    tree_model: Model,
    class_label: int | None = None,
) -> list[TreeModel]:
    """Transforms models from the catboost package to the format used by shapiq.
     Note: part of this implementation is taken and adapted from the shap package, where it can be found in shap/explainers/_tree.py.

    Args:
        tree_model: The catboost model to convert.
        class_label: The class label of the model to explain. Only used for classification models.

    Returns:
        The converted catboost model.

    """
    output_type = "raw"

    # workaround to get the single trees in the ensemble
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(delete=True) as tmp_dir:
        tmp_file = Path(tmp_dir) / "model.json"
        tree_model.save_model(tmp_file, format="json")
        with Path.open(tmp_file, encoding="utf-8") as fh:
            loaded_cb_model = json.load(fh)

    num_trees = len(loaded_cb_model["oblivious_trees"])

    trees = []
    for tree_index in range(num_trees):
        leaf_values_json = loaded_cb_model["oblivious_trees"][tree_index]["leaf_values"]
        leaf_values = [0] * (len(leaf_values_json) - 1) + leaf_values_json
        leaf_values = np.array(leaf_values)

        children_left = [i * 2 + 1 for i in range(len(leaf_values_json) - 1)]
        children_left += [-1] * len(leaf_values_json)

        children_right = [i * 2 for i in range(1, len(leaf_values_json))]
        children_right += [-1] * len(leaf_values_json)

        total_nodes = len(children_right)

        # added to assign each node a weight. Before it was only the leafs
        leaf_weights_json = loaded_cb_model["oblivious_trees"][tree_index]["leaf_weights"]
        leaf_weights = [0] * (total_nodes - len(leaf_weights_json)) + leaf_weights_json
        leaf_weights[0] = sum(leaf_weights_json)
        for index in range(len(leaf_weights_json) - 2, 0, -1):
            leaf_weights[index] = (
                leaf_weights[2 * index + 1] + leaf_weights[2 * index + 2]
            )  # each node weight is the sum of its children

        # split features and borders go from leafs to the root
        split_features_index_json = []
        borders_json = []

        for elem in loaded_cb_model["oblivious_trees"][tree_index]["splits"]:
            split_type = elem.get("split_type")
            if split_type == "FloatFeature":
                split_feature_index = elem.get("float_feature_index")
                borders_json.append(elem["border"])
            elif split_type == "OneHotFeature":
                split_feature_index = elem.get("cat_feature_index")
                borders_json.append(elem["value"])
            else:
                split_feature_index = elem.get("ctr_target_border_idx")
                borders_json.append(elem["border"])
            split_features_index_json.append(split_feature_index)

        split_features_index = []
        for counter, feature_index in enumerate(split_features_index_json[::-1]):
            split_features_index += [feature_index] * (2**counter)
        split_features_index += [-2] * (total_nodes - len(split_features_index))

        borders = []
        for counter, border in enumerate(borders_json[::-1]):
            borders += [border] * (2**counter)
        borders += [0] * (total_nodes - len(borders))

        if (safe_isinstance(tree_index, "catboost.core.Classsifier")) and class_label is not None:
            class_label = 0

        if class_label is not None:
            # turn leaf values into probabilities
            leaf_values = np.maximum(leaf_values, 0)  # because some values are negative
            if np.sum(leaf_values) > 0:
                leaf_values = leaf_values / np.sum(leaf_values)
            output_type = "probability"

        trees.append(
            TreeModel(
                children_left=np.array(children_left),
                children_right=np.array(children_right),
                features=np.array(split_features_index),
                thresholds=np.array(borders),
                values=leaf_values.reshape((-1, 1)),
                node_sample_weight=np.array(leaf_weights),
                empty_prediction=None,  # compute empty prediction later
                original_output_type=output_type,
            )
        )

    return trees
