"""Functions for converting catboost decision trees to the format used by shapiq."""

from __future__ import annotations

import numpy as np

from ....utils.custom_types import Model
from ..base import TreeModel


def convert_catboost(
    tree_model: Model,
) -> list[TreeModel]:
    """Transforms models from the catboost package to the format used by shapiq.

    Args:
        tree_model: The catboost model to convert.

    Returns:
        The converted catboost model.

    """
    output_type = "raw"

    # Logic from shap package
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(delete=True) as tmp_dir:
        tmp_file = Path(tmp_dir) / "model.json"
        tree_model.save_model(tmp_file, format="json")
        with Path.open(tmp_file, encoding="utf-8") as fh:
            loaded_cb_model = json.load(fh)

    # load the CatBoost oblivious trees specific parameters
    num_trees = len(loaded_cb_model["oblivious_trees"])

    trees = []
    for tree_index in range(num_trees):
        leaf_weights = loaded_cb_model["oblivious_trees"][tree_index]["leaf_weights"]
        leaf_weights_unraveled = [0] * (len(leaf_weights) - 1) + leaf_weights
        leaf_weights_unraveled[0] = sum(leaf_weights)
        for index in range(len(leaf_weights) - 2, 0, -1):
            leaf_weights_unraveled[index] = (
                leaf_weights_unraveled[2 * index + 1] + leaf_weights_unraveled[2 * index + 2]
            )

        leaf_values = loaded_cb_model["oblivious_trees"][tree_index]["leaf_values"]
        leaf_values_unraveled = [0] * (len(leaf_values) - 1) + leaf_values

        children_left = [i * 2 + 1 for i in range(len(leaf_values) - 1)]
        children_left += [-1] * len(leaf_values)

        children_right = [i * 2 for i in range(1, len(leaf_values))]
        children_right += [-1] * len(leaf_values)

        children_default = [i * 2 + 1 for i in range(len(leaf_values) - 1)]
        children_default += [-1] * len(leaf_values)

        # load the split features and borders
        # split features and borders go from leafs to the root
        split_features_index = []
        borders = []

        # split features and borders go from leafs to the root
        for elem in loaded_cb_model["oblivious_trees"][tree_index]["splits"]:
            split_type = elem.get("split_type")
            if split_type == "FloatFeature":
                split_feature_index = elem.get("float_feature_index")
                borders.append(elem["border"])
            elif split_type == "OneHotFeature":
                split_feature_index = elem.get("cat_feature_index")
                borders.append(elem["value"])
            else:
                split_feature_index = elem.get("ctr_target_border_idx")
                borders.append(elem["border"])
            split_features_index.append(split_feature_index)

        split_features_index_unraveled = []
        for counter, feature_index in enumerate(split_features_index[::-1]):
            split_features_index_unraveled += [feature_index] * (2**counter)
        split_features_index_unraveled += [0] * len(leaf_values)

        borders_unraveled = []
        for counter, border in enumerate(borders[::-1]):
            borders_unraveled += [border] * (2**counter)
        borders_unraveled += [0] * len(leaf_values)

        trees.append(
            TreeModel(
                children_left=np.array(children_left),
                children_right=np.array(children_right),
                features=np.array(split_features_index_unraveled),
                thresholds=np.array(borders_unraveled),
                values=np.array(leaf_values_unraveled).reshape((-1, 1)),
                node_sample_weight=np.array(leaf_weights_unraveled),
                empty_prediction=None,  # compute empty prediction later
                original_output_type=output_type,
            )
        )

    return trees
